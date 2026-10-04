#include "cdpr/plant.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace cdpr {

Simulator::Simulator(const Params& P_, const Vec3& p0, const Mat83& anchors0, const Vec4& ugv_yaw0)
    : P(P_), core(P_) {
  mixer = quad_mixer(P.drone);
  Jd_inv = P.drone.J.inverse();
  Jp_inv = P.platform.J.inverse();
  platform.p = p0;
  for (int k = 0; k < ND; ++k) {
    drone[k].p = anchors0.row(k).transpose() - P.drone.r_att;
    refs.drone_p.row(k) = drone[k].p;
  }
  for (int k = 0; k < NU; ++k) {
    ugv.row(k) << anchors0(ND + k, 0), anchors0(ND + k, 1), ugv_yaw0(k);
    refs.ugv_goal.row(k) = anchors0.row(ND + k).head<2>();
  }
  refs.platform.p = p0;
}

Truth Simulator::truth() const {
  Truth t;
  t.p = platform.p, t.v = platform.v, t.R = platform.R, t.w = platform.w;
  t.drone = drone;
  t.drone_acc = drone_acc;
  t.ugv = ugv;
  t.ugv_fair.col(2).setConstant(P.ugv.h_att);
  for (int k = 0; k < NU; ++k) {
    t.wheel(k, 0) = (ugv_vw(k, 0) - 0.5 * P.ugv.track * ugv_vw(k, 1)) / P.ugv.wheel_r;
    t.wheel(k, 1) = (ugv_vw(k, 0) + 0.5 * P.ugv.track * ugv_vw(k, 1)) / P.ugv.wheel_r;
  }
  t.L = L, t.Ldot = Ldot, t.T = T;
  return t;
}

static void true_anchors(const Simulator& s, Mat83& a, Mat83& va) {
  for (int k = 0; k < ND; ++k) {
    const Rigid& d = s.drone[k];
    a.row(k) = d.p + d.R * s.P.drone.r_att;
    va.row(k) = d.v + d.R * d.w.cross(s.P.drone.r_att);
  }
  for (int k = 0; k < NU; ++k) {
    const double th = s.ugv(k, 2);
    a.row(ND + k) << s.ugv(k, 0), s.ugv(k, 1), s.P.ugv.h_att;
    va.row(ND + k) << s.ugv_vw(k, 0) * std::cos(th), s.ugv_vw(k, 0) * std::sin(th), 0.0;
  }
}

void Simulator::initialise() {
  Mat83 a, va;
  true_anchors(*this, a, va);
  platform.p = refs.platform.p;
  platform.R = refs.platform.R;
  const CableGeom g = cable_geometry(platform.p, platform.R, P.platform.b, a);
  Vec8 lo, hi;
  tension_bounds(P, g.u, lo, hi);
  Vec6 w;
  w << Vec3(0, 0, P.platform.m * P.g) - w_known.head<3>(), -w_known.tail<3>();
  const TdResult td = tension_distribution(g.W, w, lo, hi, P.tension.t_ref, Vec8::Constant(P.tension.t_ref),
                                           adaptive_lambda(g.W, P.tension), 0.0);
  for (int i = 0; i < NC; ++i) L(i) = unstretched_length(P.cable, g.d(i), td.t(i));
  Ldot.setZero();
  T = td.t;
  for (int k = 0; k < ND; ++k) {
    // each drone starts trimmed: thrust axis along m g e3 + t u, heading as commanded
    const Vec3 F = Vec3(0, 0, P.drone.m * P.g) + td.t(k) * g.u.row(k).transpose();
    const Vec3 b3 = F.normalized();
    const Vec3 b2 = b3.cross(Vec3(std::cos(refs.drone_yaw(k)), std::sin(refs.drone_yaw(k)), 0)).normalized();
    drone[k].R << b2.cross(b3), b2, b3;
    drone[k].p = a.row(k).transpose() - drone[k].R * P.drone.r_att;
    drone[k].v.setZero();
    drone[k].w.setZero();
    refs.drone_p.row(k) = a.row(k).transpose() - P.drone.r_att;
    rotor_w.row(k).setConstant(std::sqrt(std::clamp(F.norm() / 4.0, 0.0, P.drone.f_max) / P.drone.kf));
  }
  core.reset(truth());
  core.platform.t_prev = td.t;
  core.cmd.winch.L_cmd = L;
  core.cmd.winch.T_ff = td.t;
  time = 0.0;
}

void Simulator::step() {
  const double dt = P.dt;
  const Vec3 e3(0, 0, 1);
  Mat83 a, va;
  true_anchors(*this, a, va);
  const CableGeom g = cable_geometry(platform.p, platform.R, P.platform.b, a);
  const Vec8 d_dot = cable_rates(g, platform.v, platform.R * platform.w, va);
  T = cable_tension(P.cable, L, Ldot, g.d, d_dot);
  const Vec8 half_w = 0.5 * P.cable.rho * P.g * L;

  // forces on the drones at the current rotor speeds (the IMU needs the acceleration)
  std::array<Vec3, ND> F_d, tau_d;
  for (int k = 0; k < ND; ++k) {
    Vec3 F_rot, tau_b;
    quad_wrench(P.drone, mixer, drone[k], rotor_w.row(k), F_rot, tau_b);
    const Vec3 F_cable = -T(k) * g.u.row(k).transpose() - half_w(k) * e3;
    F_d[k] = F_rot + F_cable - P.drone.m * P.g * e3;
    tau_d[k] = tau_b + P.drone.r_att.cross(drone[k].R.transpose() * F_cable);
    drone_acc.row(k) = F_d[k] / P.drone.m;
  }

  core.step(truth(), refs, w_known);

  // actuators
  for (int k = 0; k < ND; ++k) {
    const Vec4 w_cmd = (core.cmd.rotor_f.row(k).transpose() / P.drone.kf).cwiseSqrt();
    const double w_max = std::sqrt(P.drone.f_max / P.drone.kf);
    const Vec4 w_now = rotor_w.row(k);
    rotor_w.row(k) = (w_now + (w_cmd - w_now) * (dt / P.drone.tau)).cwiseMax(0.0).cwiseMin(w_max);
  }
  winch_step(P.winch, P.cable, core.cmd.winch, T, L, Ldot, dt);

  // rigid bodies (semi-implicit Euler)
  for (int k = 0; k < ND; ++k) {
    Rigid& d = drone[k];
    d.v += dt * F_d[k] / P.drone.m;
    d.p += dt * d.v;
    d.w += dt * Jd_inv * (tau_d[k] - d.w.cross(P.drone.J * d.w));
    d.R = d.R * expSO3(d.w * dt);
  }
  Vec3 f = w_known.head<3>() + ext_wrench.head<3>() - (P.platform.m * P.g + half_w.sum()) * e3;
  Vec3 tau_w = w_known.tail<3>() + ext_wrench.tail<3>();
  for (int i = 0; i < NC; ++i) {
    const Vec3 Fc = T(i) * g.u.row(i).transpose();
    f += Fc;
    tau_w += g.rb.row(i).transpose().cross(Fc);
  }
  platform.v += dt * f / P.platform.m;
  platform.p += dt * platform.v;
  platform.w += dt * Jp_inv * (platform.R.transpose() * tau_w - platform.w.cross(P.platform.J * platform.w));
  platform.R = platform.R * expSO3(platform.w * dt);

  // ground robots: rolling without slipping, wheel velocity servos, cable pulling at the fairlead on the axle axis
  const UgvParams& u = P.ugv;
  for (int k = 0; k < NU && !freeze_ugv; ++k) {
    const double th = ugv(k, 2);
    const Vec3 head(std::cos(th), std::sin(th), 0), left(-std::sin(th), std::cos(th), 0);
    const Vec3 Fc = -T(ND + k) * g.u.row(ND + k).transpose() - half_w(ND + k) * e3;
    const double m_eff = u.m + 2.0 * u.wheel_J / (u.wheel_r * u.wheel_r);
    const double I_eff = u.Iz + 0.5 * u.wheel_J * u.track * u.track / (u.wheel_r * u.wheel_r);
    double& v = ugv_vw(k, 0);
    double& om = ugv_vw(k, 1);
    const double wl = (v - 0.5 * u.track * om) / u.wheel_r, wr = (v + 0.5 * u.track * om) / u.wheel_r;
    const double tl = u.wheel_kd * (core.cmd.wheel_w(k, 0) - wl), tr = u.wheel_kd * (core.cmd.wheel_w(k, 1) - wr);
    const double F_long = Fc.dot(head);
    double F_tyre;
    if (std::abs(tl) < u.wheel_tau_max && std::abs(tr) < u.wheel_tau_max) {
      // unsaturated servo: stiff, so integrate its damping implicitly
      const double v_c = 0.5 * u.wheel_r * (core.cmd.wheel_w(k, 0) + core.cmd.wheel_w(k, 1));
      const double om_c = u.wheel_r * (core.cmd.wheel_w(k, 1) - core.cmd.wheel_w(k, 0)) / u.track;
      const double kv = 2.0 * u.wheel_kd / (u.wheel_r * u.wheel_r);
      const double ko = 0.5 * u.wheel_kd * u.track * u.track / (u.wheel_r * u.wheel_r);
      const double v_new = (v + dt * (kv * v_c + F_long) / m_eff) / (1.0 + dt * kv / m_eff);
      F_tyre = kv * (v_c - v_new);
      v = v_new;
      om = (om + dt * ko * om_c / I_eff) / (1.0 + dt * ko / I_eff);
    } else {
      const double tlc = std::clamp(tl, -u.wheel_tau_max, u.wheel_tau_max);
      const double trc = std::clamp(tr, -u.wheel_tau_max, u.wheel_tau_max);
      F_tyre = (tlc + trc) / u.wheel_r;
      v += dt * (F_tyre + F_long) / m_eff;
      om += dt * (trc - tlc) * 0.5 * u.track / u.wheel_r / I_eff;
    }
    ugv(k, 0) += dt * v * std::cos(th);
    ugv(k, 1) += dt * v * std::sin(th);
    ugv(k, 2) = wrap_pi(th + dt * om);
    // friction the tyres must supply (drive force + the sideways cable pull the wheels resist) against mu N
    const double N = u.m * P.g + Fc.z();
    traction_use(k) = N > 1e-6 ? std::hypot(F_tyre, Fc.dot(left)) / (u.mu * N) : 1e9;
  }
  time += dt;
}

void Simulator::run(int n) {
  for (int i = 0; i < n; ++i) step();
}

// ---------------------------------------------------------------------------------------------------- linearisation
static void rigid_diff(const Rigid& a, const Rigid& r, VecX& d, int& o) {
  d.segment<3>(o) = a.p - r.p;
  d.segment<3>(o + 3) = a.v - r.v;
  d.segment<3>(o + 6) = logSO3(r.R.transpose() * a.R);
  d.segment<3>(o + 9) = a.w - r.w;
  o += 12;
}

static void rigid_perturb(Rigid& a, const VecX& d, int& o) {
  a.p += d.segment<3>(o);
  a.v += d.segment<3>(o + 3);
  a.R = a.R * expSO3(d.segment<3>(o + 6));
  a.w += d.segment<3>(o + 9);
  o += 12;
}

VecX Simulator::diff(const Simulator& r) const {
  VecX d(state_dim());
  int o = 0;
  rigid_diff(platform, r.platform, d, o);
  for (int k = 0; k < ND; ++k) {
    rigid_diff(drone[k], r.drone[k], d, o);
    d.segment<4>(o) = (rotor_w.row(k) - r.rotor_w.row(k)).transpose() * 1e-3;  // rotor speed in krad/s
    o += 4;
  }
  d.segment<8>(o) = L - r.L, o += 8;
  d.segment<8>(o) = Ldot - r.Ldot, o += 8;
  d.segment<6>(o) = core.platform.e_int - r.core.platform.e_int, o += 6;
  for (int k = 0; k < ND; ++k) d.segment<3>(o) = core.drone[k].e_int - r.core.drone[k].e_int, o += 3;
  d.segment<8>(o) = core.platform.t_prev - r.core.platform.t_prev, o += 8;
  for (int i = P.hybrid ? ND : 0; i < NC; ++i) d(o++) = (core.platform.dL(i) - r.core.platform.dL(i)) * 1e3;  // mm
  if (!freeze_ugv)
    for (int k = 0; k < NU; ++k) {
      d.segment<3>(o) = (ugv.row(k) - r.ugv.row(k)).transpose();
      d(o + 2) = wrap_pi(d(o + 2));
      d.segment<2>(o + 3) = (ugv_vw.row(k) - r.ugv_vw.row(k)).transpose();
      d(o + 5) = core.ugv[k].v - r.core.ugv[k].v;
      d(o + 6) = core.ugv[k].om - r.core.ugv[k].om;
      o += 7;
    }
  return d;
}

void Simulator::perturb(const VecX& d) {
  int o = 0;
  rigid_perturb(platform, d, o);
  for (int k = 0; k < ND; ++k) {
    rigid_perturb(drone[k], d, o);
    rotor_w.row(k) += d.segment<4>(o).transpose() * 1e3;
    o += 4;
  }
  L += d.segment<8>(o), o += 8;
  Ldot += d.segment<8>(o), o += 8;
  core.platform.e_int += d.segment<6>(o), o += 6;
  for (int k = 0; k < ND; ++k) core.drone[k].e_int += d.segment<3>(o), o += 3;
  core.platform.t_prev += d.segment<8>(o), o += 8;
  for (int i = P.hybrid ? ND : 0; i < NC; ++i) core.platform.dL(i) += d(o++) * 1e-3;
  if (!freeze_ugv)
    for (int k = 0; k < NU; ++k) {
      ugv.row(k) += d.segment<3>(o).transpose();
      ugv_vw.row(k) += d.segment<2>(o + 3).transpose();
      core.ugv[k].v += d(o + 5);
      core.ugv[k].om += d(o + 6);
      o += 7;
    }
}

MatX Simulator::monodromy(int n_steps, double eps) {
  if (!P.use_truth) throw std::runtime_error("monodromy needs use_truth (a deterministic loop)");
  const int period = std::lcm(std::lcm(P.div_drone, P.div_platform), P.div_ugv);
  if (core.k % period != 0 || n_steps % period != 0)
    throw std::runtime_error("monodromy: start on, and span, a multiple of the slowest control period");
  const int n = state_dim();
  Simulator nominal = *this;
  nominal.run(n_steps);
  MatX A(n, n);
  for (int j = 0; j < n; ++j) {
    VecX dx = VecX::Zero(n);
    dx(j) = eps;
    Simulator plus = *this, minus = *this;
    plus.perturb(dx);
    minus.perturb(-dx);
    plus.run(n_steps);
    minus.run(n_steps);
    A.col(j) = (plus.diff(nominal) - minus.diff(nominal)) / (2.0 * eps);
  }
  return A;
}

}  // namespace cdpr
