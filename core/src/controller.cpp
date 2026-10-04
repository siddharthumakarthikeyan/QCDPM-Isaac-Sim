#include "cdpr/controller.hpp"

#include <algorithm>

namespace cdpr {

// ---------------------------------------------------------------------------------------------------- drone
Vec4 DroneController::step(const DroneParams& d, const Mat44& mixer_inv, double g, const Rigid& s, const Vec3& p_ref,
                           const Vec3& v_ref, const Vec3& a_ref, double yaw_ref, const Vec3& F_ext, double dt) {
  const Vec3 kp = d.kp * d.m, kd = d.kd * d.m, ki = d.ki * d.m;
  const Vec3 ep = s.p - p_ref, ev = s.v - v_ref;
  const Vec3 lim = d.i_lim * ki.cwiseMax(1e-9).cwiseInverse();
  e_int = (e_int + ep * dt).cwiseMax(-lim).cwiseMin(lim);
  Vec3 F = -kp.cwiseProduct(ep) - kd.cwiseProduct(ev) - ki.cwiseProduct(e_int) + d.m * (a_ref + Vec3(0, 0, g));
  if (d.cable_ff) F -= F_ext;
  // tilt limit: keep the vertical component, shrink the horizontal one
  F.z() = std::max(F.z(), 0.2 * d.m * g);
  const double h = std::hypot(F.x(), F.y()), h_max = F.z() * std::tan(d.max_tilt);
  if (h > h_max) F.head<2>() *= h_max / h;
  const Vec3 b3 = F.normalized();
  const Vec3 b1c(std::cos(yaw_ref), std::sin(yaw_ref), 0.0);
  const Vec3 b2 = b3.cross(b1c).normalized();
  Mat3 Rd;
  Rd << b2.cross(b3), b2, b3;
  const double thrust = F.dot(s.R.col(2));
  const Vec3 eR = rot_error(Rd, s.R);
  Vec3 M = -d.kR.cwiseProduct(eR) - d.kW.cwiseProduct(s.w) + s.w.cross(d.J * s.w);
  if (d.cable_ff) M -= d.r_att.cross(s.R.transpose() * F_ext);  // the fairlead is offset from the centre of mass
  Vec4 u;
  u << thrust, M;
  return (mixer_inv * u).cwiseMax(0.0).cwiseMin(d.f_max);
}

// ---------------------------------------------------------------------------------------------------- ground robot
Vec2 UgvController::step(const UgvParams& u, const Vec3& pose, const Vec2& goal, const Vec2& goal_vel,
                         double yaw_goal, bool velocity_mode, double cmd_v, double cmd_om, double dt) {
  double v_t = cmd_v, om_t = cmd_om;
  if (!velocity_mode) {
    const Vec2 e = goal - pose.head<2>();
    Vec2 vd = goal_vel + u.k_pos * e;
    double sp = vd.norm();
    if (sp > u.v_max) vd *= u.v_max / sp, sp = u.v_max;
    double e_head = wrap_pi(std::atan2(vd.y(), vd.x()) - pose(2));
    const bool back = std::abs(e_head) > M_PI / 2;  // drive backwards if that needs less turning
    if (back) e_head = wrap_pi(e_head + M_PI);
    v_t = sp * std::cos(e_head) * (back ? -1.0 : 1.0);
    om_t = u.k_head * e_head;
    if (e.norm() < u.pos_tol && goal_vel.norm() <= 1e-3) {
      v_t = 0.0;
      om_t = std::isnan(yaw_goal) ? 0.0 : u.k_yaw * wrap_pi(yaw_goal - pose(2));
    }
  }
  v += std::clamp(v_t - v, -u.a_lin * dt, u.a_lin * dt);
  om += std::clamp(om_t - om, -u.a_ang * dt, u.a_ang * dt);
  const Vec2 wheel((v - 0.5 * u.track * om) / u.wheel_r, (v + 0.5 * u.track * om) / u.wheel_r);
  return wheel.cwiseMax(-u.wheel_w_max).cwiseMin(u.wheel_w_max);
}

// ---------------------------------------------------------------------------------------------------- platform
PlatformRef PlatformController::governed(const PlatformGains& k, const PlatformRef& ref, double dt) {
  if (!gov_init) gov = ref, gov_init = true;
  // translation: the velocity follows the commanded one with bounded acceleration, the position is pulled back on
  Vec3 dv = ref.v + k.k_gov * (ref.p - gov.p) - gov.v;
  const double dv_max = k.a_max * dt;
  if (dv.norm() > dv_max) dv *= dv_max / dv.norm();
  gov.a = dv / dt + ref.a;
  gov.v += dv;
  gov.p += gov.v * dt;
  // rotation, same law on SO(3) (rates in the governed frame)
  const Vec3 e = logSO3(gov.R.transpose() * ref.R);
  Vec3 dw = gov.R.transpose() * ref.R * ref.w + k.k_gov * e - gov.w;
  const double dw_max = k.alpha_max * dt;
  if (dw.norm() > dw_max) dw *= dw_max / dw.norm();
  gov.w += dw;
  gov.R = gov.R * expSO3(gov.w * dt);
  return gov;
}

WinchCmd PlatformController::step(const Params& P, const PlatformRef& ref_cmd, const Estimate& est,
                                  const Vec6& w_known, double dt) {
  const PlatformRef ref = P.gains.a_max > 0 ? governed(P.gains, ref_cmd, dt) : ref_cmd;
  const PlatformParams& pl = P.platform;
  const PlatformGains& k = P.gains;
  // tracking errors on SE(3)
  const Vec3 e_p = ref.p - est.p, e_v = ref.v - est.v;
  const Vec3 e_R = rot_error(ref.R, est.R);
  const Vec3 w_d = est.R.transpose() * ref.R * ref.w;  // reference rate seen from the body
  const Vec3 e_W = est.w - w_d;
  e_int.head<3>() += e_p * dt;
  e_int.tail<3>() -= e_R * dt;
  const double ll = k.i_lim / std::max(k.ki_lin, 1e-9), lr = k.i_lim / std::max(k.ki_rot, 1e-9);
  e_int.head<3>() = e_int.head<3>().cwiseMax(-ll).cwiseMin(ll);  // bounded authority, so bounded wind-up
  e_int.tail<3>() = e_int.tail<3>().cwiseMax(-lr).cwiseMin(lr);
  const Vec3 a = ref.a + k.kp_lin * e_p + k.kd_lin * e_v + k.ki_lin * e_int.head<3>();
  const Vec3 alpha = -k.kp_rot * e_R - k.kd_rot * e_W + k.ki_rot * e_int.tail<3>() - est.w.cross(w_d);
  // wrench the cables have to produce (world, about the centre of mass)
  Vec6 w;
  w.head<3>() = pl.m * (a + Vec3(0, 0, P.g)) - w_known.head<3>();
  w.tail<3>() = est.R * (pl.J * alpha + est.w.cross(pl.J * est.w)) - w_known.tail<3>();
  if (P.dob_ff) w -= est.d_hat;
  wrench = w;

  // cable geometry at the reference pose with the anchors where they are now
  const CableGeom g = cable_geometry(ref.p, ref.R, pl.b, est.anchors);
  Vec8 lo, hi;
  tension_bounds(P, g.u, lo, hi);
  td = tension_distribution(g.W, w, lo, hi, P.tension.t_ref, t_prev, adaptive_lambda(g.W, P.tension, &sigma_min),
                            P.tension.w_rate);
  t_prev = t_des = td.t;
  t_cap = hi;

  WinchCmd c;
  const Vec8 rate = cable_rates(g, ref.v, ref.R * ref.w, est.anchor_vel);
  // commanded twist for the tension-mode winches: the reference twist plus a bounded correction of the pose error
  Vec3 dv = k.k_twist * e_p, dw = -k.k_twist * (est.R * e_R);
  if (dv.norm() > k.twist_max) dv *= k.twist_max / dv.norm();
  const double w_max = k.twist_max / std::max(pl.b.row(0).norm(), 1e-6);
  if (dw.norm() > w_max) dw *= w_max / dw.norm();
  const Vec8 rate_c = cable_rates(g, ref.v + dv, ref.R * ref.w + dw, est.anchor_vel);
  const CableGeom g_ik = cable_geometry(ref.p + k.k_ik * e_int.head<3>(),
                                        ref.R * expSO3(ref.R.transpose() * est.R * (k.k_ik * e_int.tail<3>())), pl.b,
                                        est.anchors);
  for (int i = 0; i < NC; ++i) {
    c.tension_mode(i) = P.hybrid && i < ND;
    if (!c.tension_mode(i)) {
      const double eT = est.T(i) - td.t(i);
      const double eT_db = eT > k.adm_band ? eT - k.adm_band : (eT < -k.adm_band ? eT + k.adm_band : 0.0);
      const double guard = est.T(i) < k.t_guard ? -k.k_guard * (k.t_guard - est.T(i)) : 0.0;
      dL(i) = std::clamp(dL(i) + (k.k_adm * eT_db + guard) * dt, -k.adm_max, k.adm_max);
    }
    c.L_cmd(i) = unstretched_length(P.cable, g_ik.d(i), td.t(i)) + (c.tension_mode(i) ? 0.0 : dL(i));
    c.Ld_cmd(i) = c.tension_mode(i) ? rate_c(i) : rate(i);
    c.T_ff(i) = td.t(i);
  }
  return c;
}

// ---------------------------------------------------------------------------------------------------- core
Core::Core(const Params& P_) : P(P_), sensors(P_.sensors, P_.dt, P_.div_drone) {
  mixer_inv = quad_mixer(P.drone).inverse();
  platform.t_prev.setConstant(P.tension.t_ref);
}

static void fill_anchors(const Params& P, Estimate& e) {
  for (int k = 0; k < ND; ++k) {
    const Rigid& d = e.drone[k];
    e.anchors.row(k) = d.p + d.R * P.drone.r_att;
    e.anchor_vel.row(k) = d.v + d.R * d.w.cross(P.drone.r_att);
  }
  for (int k = 0; k < NU; ++k) {
    const double th = e.ugv(k, 2);
    e.anchors.row(ND + k) = Vec3(e.ugv(k, 0), e.ugv(k, 1), 0.0).transpose() + e.ugv_fair.row(k);
    e.anchor_vel.row(ND + k) << e.ugv_vw(k, 0) * std::cos(th), e.ugv_vw(k, 0) * std::sin(th), 0.0;
  }
}

static void copy_truth(const Params& P, const Truth& t, Estimate& e) {
  e.p = t.p, e.v = t.v, e.R = t.R, e.w = t.w;
  e.d_hat.setZero();
  e.drone = t.drone;
  e.ugv = t.ugv;
  e.ugv_fair = t.ugv_fair;
  for (int k = 0; k < NU; ++k) {
    e.ugv_vw(k, 0) = 0.5 * P.ugv.wheel_r * (t.wheel(k, 0) + t.wheel(k, 1));
    e.ugv_vw(k, 1) = P.ugv.wheel_r * (t.wheel(k, 1) - t.wheel(k, 0)) / P.ugv.track;
  }
  e.L = t.L, e.Ldot = t.Ldot, e.T = t.T;
  fill_anchors(P, e);
}

void Core::reset(const Truth& t) {
  copy_truth(P, t, est);
  for (int k = 0; k < ND; ++k) {
    drone_filter[k].init(t.drone[k].p, yaw_of(t.drone[k].R));
    drone_filter[k].R = t.drone[k].R;
    drone_filter[k].v = t.drone[k].v;
  }
  for (int k = 0; k < NU; ++k) ugv_filter[k].x = t.ugv.row(k);
  observer.init(t.p, t.R);
  observer.v = t.v;
  observer.w = t.w;
  k = 0;
}

void Core::estimate(const Truth& t, const Vec6& w_known) {
  if (P.use_truth) {
    copy_truth(P, t, est);
    return;
  }
  const SensorParams& s = P.sensors;
  sensors.winch(t, est.L, est.Ldot, est.T);

  for (int i = 0; i < ND; ++i) {
    DroneEskf& f = drone_filter[i];
    Vec3 acc_m, gyro_m;
    if (sensors.imu(i, t.drone[i], t.drone_acc.row(i), P.g, acc_m, gyro_m))
      f.predict(acc_m, gyro_m, P.dt * P.div_drone, P.g, P.observer, s);
    if (k % s.div_drone_fix == 0) {
      f.update_position(sensors.drone_fix(t.drone[i].p), s.drone_fix_sigma);
      f.update_yaw(sensors.drone_yaw(yaw_of(t.drone[i].R)), s.drone_yaw_sigma);
    }
    est.drone[i].p = f.p, est.drone[i].v = f.v, est.drone[i].R = f.R, est.drone[i].w = f.w_body;
  }
  for (int i = 0; i < NU; ++i) {
    UgvEkf& f = ugv_filter[i];
    if (k % P.div_ugv == 0) {
      const Vec2 w = sensors.wheels(t.wheel.row(i));
      f.predict(w(0), w(1), P.dt * P.div_ugv, P.ugv, s.wheel_sigma);
      const double al = std::min(1.0, P.dt * P.div_ugv / P.observer.fair_tau);
      est.ugv_fair.row(i) += al * (sensors.fairlead(t.ugv_fair.row(i)).transpose() - est.ugv_fair.row(i));
    }
    if (k % s.div_ugv_fix == 0) f.update(sensors.ugv_fix(t.ugv.row(i)), s.ugv_fix_sigma, s.ugv_yaw_sigma);
    est.ugv.row(i) = f.x;
    est.ugv_vw.row(i) << f.v, f.om;
  }
  fill_anchors(P, est);

  // platform: tension-driven prediction, corrected by the cable lengths and by the tags
  const ObserverParams& o = P.observer;
  observer.predict(P, est.T, est.L, est.anchors, w_known, P.dt);
  if (k % P.div_platform == 0) {
    for (int i = 0; i < NC; ++i) {
      const Vec3 a = est.anchors.row(i);
      const Vec3 b = P.platform.b.row(i);
      const double u_z = (a - observer.p - observer.R * b).normalized().z();
      d_meas(i) = chord_length(P.cable, est.L(i), est.T(i), u_z, P.g, o.sag);
      if (est.T(i) > o.t_slack)
        observer.update_cable(P, i, d_meas(i), a, i < ND ? o.sigma_cable_drone : o.sigma_cable_ugv);
    }
  }
  if (k % s.div_tag == 0) {
    // one camera per frame, round robin over the eight robots; the tag pose is measured in the robot's frame and
    // carried to the world with that robot's own estimated pose
    const int r = tag_robot++ % NC;
    Vec3 pr_true, pr_est, p_rel;
    Mat3 Rr_true, Rr_est, R_rel;
    if (r < ND) {
      pr_true = t.drone[r].p, Rr_true = t.drone[r].R;
      pr_est = est.drone[r].p, Rr_est = est.drone[r].R;
    } else {
      const int j = r - ND;
      pr_true = Vec3(t.ugv(j, 0), t.ugv(j, 1), 0.0) + t.ugv_fair.row(j).transpose();
      pr_est = est.anchors.row(r);
      Rr_true = Eigen::AngleAxisd(t.ugv(j, 2), Vec3::UnitZ()).toRotationMatrix();
      Rr_est = Eigen::AngleAxisd(est.ugv(j, 2), Vec3::UnitZ()).toRotationMatrix();
    }
    sensors.tag(pr_true, Rr_true, t.p, t.R, p_rel, R_rel);
    tag_p = pr_est + Rr_est * p_rel;
    observer.update_tag(tag_p, Rr_est * R_rel, o.sigma_tag_p, o.sigma_tag_th, o.gate);
  }
  est.p = observer.p, est.v = observer.v, est.R = observer.R, est.w = observer.w;
  est.d_hat = observer.disturbance_world();
}

void Core::step(const Truth& t, const Refs& refs, const Vec6& w_known) {
  estimate(t, w_known);
  if (k % P.div_platform == 0) cmd.winch = platform.step(P, refs.platform, est, w_known, P.dt * P.div_platform);
  if (k % P.div_drone == 0) {
    for (int i = 0; i < ND; ++i) {
      const Vec3 b = P.platform.b.row(i);
      const Vec3 a = est.anchors.row(i);
      const Vec3 u = (a - est.p - est.R * b).normalized();
      const Vec3 F_ext = -est.T(i) * u - Vec3(0, 0, 0.5 * P.cable.rho * est.L(i) * P.g);
      cmd.rotor_f.row(i) = drone[i].step(P.drone, mixer_inv, P.g, est.drone[i], refs.drone_p.row(i),
                                         refs.drone_v.row(i), refs.drone_a.row(i), refs.drone_yaw(i), F_ext,
                                         P.dt * P.div_drone);
    }
  }
  if (k % P.div_ugv == 0) {
    for (int i = 0; i < NU; ++i)
      cmd.wheel_w.row(i) = ugv[i].step(P.ugv, est.ugv.row(i), refs.ugv_goal.row(i), refs.ugv_goal_vel.row(i),
                                       refs.ugv_yaw(i), refs.ugv_velocity_mode(i) != 0, refs.ugv_cmd_v(i),
                                       refs.ugv_cmd_om(i), P.dt * P.div_ugv);
  }
  ++k;
}

}  // namespace cdpr
