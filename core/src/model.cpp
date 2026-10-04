#include "cdpr/model.hpp"

#include <algorithm>

namespace cdpr {

// ---------------------------------------------------------------------------------------------------- cables
CableGeom cable_geometry(const Vec3& p, const Mat3& R, const Mat83& b, const Mat83& anchors) {
  CableGeom g;
  for (int i = 0; i < NC; ++i) {
    const Vec3 rb = R * b.row(i).transpose();
    const Vec3 dv = anchors.row(i).transpose() - p - rb;
    const double d = std::max(dv.norm(), 1e-9);
    const Vec3 u = dv / d;
    g.rb.row(i) = rb;
    g.u.row(i) = u;
    g.d(i) = d;
    g.W.col(i) << u, rb.cross(u);
  }
  return g;
}

Vec8 cable_rates(const CableGeom& g, const Vec3& v, const Vec3& w_world, const Mat83& anchor_vel) {
  Vec8 r;
  for (int i = 0; i < NC; ++i) {
    const Vec3 rb = g.rb.row(i);
    const Vec3 va = v + w_world.cross(rb);
    r(i) = g.u.row(i).dot(anchor_vel.row(i) - va.transpose());
  }
  return r;
}

Vec8 cable_tension(const CableParams& c, const Vec8& L, const Vec8& Ldot, const Vec8& d, const Vec8& d_dot) {
  Vec8 T;
  for (int i = 0; i < NC; ++i) {
    const double stretch = d(i) - L(i);
    const double t = c.EA / L(i) * stretch + c.cEA / L(i) * (d_dot(i) - Ldot(i));
    T(i) = stretch > 0.0 ? std::max(t, 0.0) : 0.0;
  }
  return T;
}

double chord_length(const CableParams& c, double L, double T, double u_z, double g, bool sag) {
  const double s = L * (1.0 + T / c.EA);  // stretched arc length
  if (!sag || T < 1e-6) return s;
  // parabolic sag under the weight component normal to the chord: chord = s (1 - (w_n s)^2 / (24 T^2))
  const double wn = c.rho * g * std::sqrt(std::max(0.0, 1.0 - u_z * u_z));
  const double k = wn * s / T;
  return s * (1.0 - k * k / 24.0);
}

double forward_kinematics(const Vec8& d, const Mat83& b, const Mat83& anchors, Vec3& p, Mat3& R, int max_iter) {
  double rms = 0.0;
  double mu = 1e-9;
  for (int it = 0; it < max_iter; ++it) {
    const CableGeom g = cable_geometry(p, R, b, anchors);
    Eigen::Matrix<double, 8, 6> J;
    for (int i = 0; i < NC; ++i) {
      const Vec3 u = g.u.row(i);
      J.block<1, 3>(i, 0) = -u.transpose();
      J.block<1, 3>(i, 3) = u.transpose() * R * hat(b.row(i).transpose());
    }
    const Vec8 r = d - g.d;
    rms = std::sqrt(r.squaredNorm() / NC);
    Eigen::Matrix<double, 6, 6> A = J.transpose() * J;
    A.diagonal().array() += mu;
    const Vec6 dx = A.ldlt().solve(J.transpose() * r);
    p += dx.head<3>();
    R = R * expSO3(dx.tail<3>());
    if (dx.norm() < 1e-12) break;
  }
  return rms;
}

// ---------------------------------------------------------------------------------------------------- winches
void winch_step(const WinchParams& w, const CableParams& c, const WinchCmd& cmd, const Vec8& T, Vec8& L, Vec8& Ldot,
                double dt) {
  for (int i = 0; i < NC; ++i) {
    const double e = cmd.L_cmd(i) - L(i);
    const double e_dot = cmd.Ld_cmd(i) - Ldot(i);
    const double damp = w.m_eq * 2.0 * w.zeta * w.wn * e_dot;
    const double F = -cmd.T_ff(i) + (cmd.tension_mode(i) ? damp : w.m_eq * w.wn * w.wn * e + damp);
    // motor envelope: the available force falls linearly to zero at the no-load speed when driving along the motion
    const bool driving = (F > 0) == (Ldot(i) > 0) && F != 0.0 && Ldot(i) != 0.0;
    const double F_avail = driving ? w.F_max * std::clamp(1.0 - std::abs(Ldot(i)) / w.v_max, 0.0, 1.0) : w.F_max;
    const double F_motor = std::clamp(F, -F_avail, F_avail);
    const double acc = (F_motor + T(i) - w.c_v * Ldot(i) - w.F_c * std::tanh(Ldot(i) / 1e-3)) / w.m_eq;
    Ldot(i) += acc * dt;
    L(i) += Ldot(i) * dt;
    if (L(i) < c.Lmin || L(i) > c.Lmax) {
      L(i) = std::clamp(L(i), c.Lmin, c.Lmax);
      Ldot(i) = 0.0;
    }
  }
}

// ---------------------------------------------------------------------------------------------------- quadrotor
Mat44 quad_mixer(const DroneParams& d) {
  // rotor order: front-left, rear-left, rear-right, front-right
  const double a = d.arm / std::sqrt(2.0);
  const Vec4 xs(a, -a, -a, a), ys(a, a, -a, -a), spin(1, -1, 1, -1);
  Mat44 M;
  M.row(0).setOnes();
  M.row(1) = ys.transpose();
  M.row(2) = -xs.transpose();
  M.row(3) = -d.km * spin.transpose();
  return M;
}

void quad_wrench(const DroneParams& d, const Mat44& mixer, const Rigid& s, const Vec4& rotor_w, Vec3& F_world,
                 Vec3& tau_body) {
  const Vec4 f = d.kf * rotor_w.array().square();
  const Vec4 u = mixer * f;
  const Vec3 v_body = s.R.transpose() * s.v;
  const Vec3 F_body(-d.drag_lin * v_body.x(), -d.drag_lin * v_body.y(), u(0));
  F_world = s.R * F_body - d.drag_quad * s.v.norm() * s.v;
  tau_body = u.tail<3>();
}

// ---------------------------------------------------------------------------------------------------- limits
double drone_tension_cap(const Vec3& u, double m, double g, double theta_max, double f_max) {
  // u points platform -> drone, so the cable pulls the drone with -t u and the thrust must be m g e3 + t u.
  const double mg = m * g;
  const double uxy = std::hypot(u.x(), u.y()), uz = u.z();
  const double tan_th = std::tan(theta_max);
  const double den = uxy - uz * tan_th;
  const double t_tilt = den > 1e-9 ? mg * tan_th / den : std::numeric_limits<double>::infinity();
  const double disc = mg * uz * mg * uz - mg * mg + f_max * f_max;
  const double t_thrust = -mg * uz + std::sqrt(std::max(disc, 0.0));
  return std::min(t_tilt, t_thrust);
}

double ugv_tension_cap(const Vec3& u, const UgvParams& p, double g, double margin) {
  // The cable pulls the fairlead with -t u: horizontally t |u_xy| towards the platform, vertically -t u_z (upwards
  // when the platform is above the fairlead, u_z < 0 here means the anchor is below the attachment point).
  const double mg = p.m * g;
  const double uxy = std::hypot(u.x(), u.y()), uz = u.z();
  double cap = std::numeric_limits<double>::infinity();
  // sliding: t |u_xy| <= mu (m g + t u_z)
  const double den_s = uxy - margin * p.mu * uz;
  if (den_s > 1e-9) cap = std::min(cap, margin * p.mu * mg / den_s);
  // tipping about the edge of the support polygon: t |u_xy| h <= (m g + t u_z) a
  const double a = margin * p.tip_radius;
  const double den_t = uxy * p.h_att - uz * a;
  if (den_t > 1e-9) cap = std::min(cap, mg * a / den_t);
  return cap;
}

void tension_bounds(const Params& P, const Mat83& u, Vec8& lo, Vec8& hi) {
  const TensionParams& t = P.tension;
  lo.setConstant(t.t_min);
  for (int i = P.hybrid ? ND : 0; i < NC; ++i) lo(i) = std::max(t.t_min, t.t_min_length);
  hi.setConstant(t.t_max);
  for (int i = 0; i < ND; ++i)
    if (t.drone_caps)
      hi(i) = std::min(hi(i), drone_tension_cap(u.row(i), P.drone.m, P.g, t.tilt_cap, t.thrust_cap));
  for (int i = 0; i < NU; ++i)
    if (t.ugv_caps) hi(ND + i) = std::min(hi(ND + i), ugv_tension_cap(u.row(ND + i), P.ugv, P.g, t.ground_margin));
  for (int i = 0; i < NC; ++i) hi(i) = std::max(hi(i), lo(i));
}

// ---------------------------------------------------------------------------------------------------- tensions
int solve_box_qp(const Mat8& H, const Vec8& g, const Vec8& lo, const Vec8& hi, Vec8& x, int* n_active) {
  using SmallM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 8, 8>;
  using SmallV = Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 8, 1>;
  x = x.cwiseMax(lo).cwiseMin(hi);
  int state[NC] = {0};  // 0 free, -1 at the lower bound, +1 at the upper bound
  int it = 0;
  for (; it < 64; ++it) {
    int idx[NC], nf = 0;
    for (int i = 0; i < NC; ++i)
      if (state[i] == 0) idx[nf++] = i;
    if (nf > 0) {
      // minimiser over the free variables with the others held at their bounds
      SmallM Hff(nf, nf);
      SmallV rhs(nf);
      const Vec8 Hx_fixed = [&] {
        Vec8 xb = x;
        for (int k = 0; k < nf; ++k) xb(idx[k]) = 0.0;
        return Vec8(H * xb);
      }();
      for (int a = 0; a < nf; ++a) {
        rhs(a) = -g(idx[a]) - Hx_fixed(idx[a]);
        for (int c = 0; c < nf; ++c) Hff(a, c) = H(idx[a], idx[c]);
      }
      const SmallV y = Hff.llt().solve(rhs);
      // longest step towards it that stays inside the box
      double alpha = 1.0;
      int block = -1, side = 0;
      for (int a = 0; a < nf; ++a) {
        const int i = idx[a];
        const double dx = y(a) - x(i);
        if (dx > 1e-14 && x(i) + dx > hi(i)) {
          const double al = (hi(i) - x(i)) / dx;
          if (al < alpha) alpha = al, block = i, side = 1;
        } else if (dx < -1e-14 && x(i) + dx < lo(i)) {
          const double al = (lo(i) - x(i)) / dx;
          if (al < alpha) alpha = al, block = i, side = -1;
        }
      }
      for (int a = 0; a < nf; ++a) x(idx[a]) += alpha * (y(a) - x(idx[a]));
      if (block >= 0) {
        state[block] = side;
        x(block) = side > 0 ? hi(block) : lo(block);
        continue;
      }
    }
    // the free variables are optimal: release the bound with the most wrong-signed multiplier, if any
    const Vec8 grad = H * x + g;
    int worst = -1;
    double worst_val = 1e-9;
    for (int i = 0; i < NC; ++i) {
      const double viol = state[i] < 0 ? -grad(i) : (state[i] > 0 ? grad(i) : 0.0);
      if (viol > worst_val) worst_val = viol, worst = i;
    }
    if (worst < 0) break;
    state[worst] = 0;
  }
  if (n_active) {
    *n_active = 0;
    for (int i = 0; i < NC; ++i) *n_active += state[i] != 0;
  }
  return it + 1;
}

double adaptive_lambda(const Mat68& W, const TensionParams& t, double* sigma_min) {
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(W * W.transpose(), Eigen::EigenvaluesOnly);
  const double s = std::sqrt(std::max(es.eigenvalues()(0), 0.0));
  if (sigma_min) *sigma_min = s;
  if (s >= t.sigma_eps) return t.lambda_min;
  const double x = s / t.sigma_eps;
  return t.lambda_min + (1.0 - x * x) * (t.lambda - t.lambda_min);
}

TdResult tension_distribution(const Mat68& W, const Vec6& w, const Vec8& lo, const Vec8& hi, double t_ref,
                              const Vec8& t_prev, double lambda, double w_rate) {
  // cost scaled by lambda:  lambda |t - t_ref|^2 + lambda w_rate |t - t_prev|^2 + |W t - w|^2
  TdResult r;
  Mat8 H = W.transpose() * W;
  H.diagonal().array() += lambda * (1.0 + w_rate);
  const Vec8 g = -(W.transpose() * w) - lambda * (Vec8::Constant(t_ref) + w_rate * t_prev);
  const Vec8 t_free = H.llt().solve(-g);
  r.t = t_prev;
  r.iterations = solve_box_qp(H, g, lo, hi, r.t, &r.n_active);
  r.residual = W * r.t - w;
  const double res_free = (W * t_free - w).norm();
  r.feasible = r.residual.norm() <= 1.5 * res_free + 0.1;
  return r;
}

}  // namespace cdpr
