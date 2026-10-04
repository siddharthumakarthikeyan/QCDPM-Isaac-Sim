#include "cdpr/estimator.hpp"

namespace cdpr {

// ---------------------------------------------------------------------------------------------------- drone
// error state: [dp, dv, dtheta (body), dbg, dba]
void DroneEskf::init(const Vec3& p0, double yaw0) {
  p = p0;
  v.setZero();
  R = Eigen::AngleAxisd(yaw0, Vec3::UnitZ()).toRotationMatrix();
  bg.setZero();
  ba.setZero();
  Eigen::Matrix<double, 15, 1> d;
  d << Vec3::Constant(1e-6), Vec3::Constant(1e-2), Vec3::Constant(1e-4), Vec3::Constant(2.5e-5),
      Vec3::Constant(2.5e-3);
  P = d.asDiagonal();
  initialised = true;
}

void DroneEskf::predict(const Vec3& acc_m, const Vec3& gyro_m, double dt, double g, const ObserverParams& o,
                        const SensorParams& s) {
  const Vec3 w = gyro_m - bg;
  const Vec3 a_b = acc_m - ba;
  const Vec3 a = R * a_b - Vec3(0, 0, g);
  M15 F = M15::Identity();
  F.block<3, 3>(0, 3) = Mat3::Identity() * dt;
  F.block<3, 3>(3, 6) = -R * hat(a_b) * dt;
  F.block<3, 3>(3, 12) = -R * dt;
  F.block<3, 3>(6, 6) = expSO3(-w * dt);
  F.block<3, 3>(6, 9) = -Mat3::Identity() * dt;
  p += v * dt + 0.5 * a * dt * dt;
  v += a * dt;
  R = R * expSO3(w * dt);
  w_body = w;
  acc_world = a;
  P = F * P * F.transpose();
  const double qv = o.drone_acc_sigma * dt, qth = o.drone_gyro_sigma * dt;
  P.diagonal().segment<3>(3).array() += qv * qv;
  P.diagonal().segment<3>(6).array() += qth * qth;
  P.diagonal().segment<3>(9).array() += s.gyro_rw * s.gyro_rw * dt;
  P.diagonal().segment<3>(12).array() += s.acc_rw * s.acc_rw * dt;
}

void DroneEskf::inject(const Eigen::Matrix<double, 15, 1>& dx) {
  p += dx.segment<3>(0);
  v += dx.segment<3>(3);
  R = R * expSO3(dx.segment<3>(6));
  bg += dx.segment<3>(9);
  ba += dx.segment<3>(12);
}

void DroneEskf::scalar_update(const Eigen::Matrix<double, 1, 15>& H, double r, double var) {
  const Eigen::Matrix<double, 15, 1> PHt = P * H.transpose();
  const double S = H * PHt + var;
  const Eigen::Matrix<double, 15, 1> K = PHt / S;
  P -= K * PHt.transpose();
  P = 0.5 * (P + P.transpose()).eval();
  inject(K * r);
}

void DroneEskf::update_position(const Vec3& z, double sigma) {
  for (int j = 0; j < 3; ++j) {
    Eigen::Matrix<double, 1, 15> H = Eigen::Matrix<double, 1, 15>::Zero();
    H(j) = 1.0;
    scalar_update(H, z(j) - p(j), sigma * sigma);
  }
}

void DroneEskf::update_yaw(double z, double sigma) {
  // yaw = atan2(R10, R00); a body-frame perturbation dtheta changes the first column by -R hat(e1) dtheta
  const Mat3 A = -R * hat(Vec3::UnitX());
  const double n2 = R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0);
  Eigen::Matrix<double, 1, 15> H = Eigen::Matrix<double, 1, 15>::Zero();
  H.segment<3>(6) = (R(0, 0) * A.row(1) - R(1, 0) * A.row(0)) / n2;
  scalar_update(H, wrap_pi(z - yaw_of(R)), sigma * sigma);
}

// ---------------------------------------------------------------------------------------------------- ground robot
void UgvEkf::predict(double wl, double wr, double dt, const UgvParams& u, double wheel_sigma) {
  v = 0.5 * u.wheel_r * (wl + wr);
  om = u.wheel_r * (wr - wl) / u.track;
  const double c = std::cos(x(2)), s = std::sin(x(2));
  Mat3 F = Mat3::Identity();
  F(0, 2) = -v * s * dt;
  F(1, 2) = v * c * dt;
  x += Vec3(v * c, v * s, om) * dt;
  x(2) = wrap_pi(x(2));
  // encoder noise (white, per sample), slip in proportion to the motion and the slow drag the encoders cannot see
  // (both random walks)
  const double ev = u.wheel_r * wheel_sigma / std::sqrt(2.0) * dt;
  const double eo = u.wheel_r * wheel_sigma * std::sqrt(2.0) / u.track * dt;
  const double qv = (slip_ratio * slip_ratio * v * v + slip_xy * slip_xy) * dt;
  const double qo = (slip_ratio * slip_ratio * om * om + slip_yaw * slip_yaw) * dt;
  P = F * P * F.transpose();
  P(0, 0) += ev * ev + qv;
  P(1, 1) += ev * ev + qv;
  P(2, 2) += eo * eo + qo;
}

void UgvEkf::update(const Vec3& z, double sigma_xy, double sigma_yaw) {
  for (int j = 0; j < 3; ++j) {
    const double r = j < 2 ? z(j) - x(j) : wrap_pi(z(2) - x(2));
    const double var = j < 2 ? sigma_xy * sigma_xy : sigma_yaw * sigma_yaw;
    const Vec3 PHt = P.col(j);
    const Vec3 K = PHt / (P(j, j) + var);
    P -= K * PHt.transpose();
    P = 0.5 * (P + P.transpose()).eval();
    x += K * r;
    x(2) = wrap_pi(x(2));
  }
}

// ---------------------------------------------------------------------------------------------------- platform
// error state: [dp, dv, dtheta (body), dOmega, d(df), d(dtau)]
void PlatformObserver::init(const Vec3& p0, const Mat3& R0) {
  p = p0;
  R = R0;
  v.setZero();
  w.setZero();
  df.setZero();
  dtau.setZero();
  Eigen::Matrix<double, 18, 1> d;
  d << Vec3::Constant(1e-6), Vec3::Constant(1e-2), Vec3::Constant(1e-5), Vec3::Constant(1e-2), Vec3::Constant(25.0),
      Vec3::Constant(1.0);
  P = d.asDiagonal();
}

void PlatformObserver::predict(const Params& P_, const Vec8& T, const Vec8& L, const Mat83& anchors,
                               const Vec6& w_known, double dt) {
  const PlatformParams& pl = P_.platform;
  const CableGeom g = cable_geometry(p, R, pl.b, anchors);
  const Mat3 Rt = R.transpose();
  const Mat3 Jinv = pl.J.inverse();

  // cable wrench and its sensitivity to the pose. The tensions are measured, so only the direction of each cable
  // force depends on the state: d(T u)/dx_a = -(T/d)(I - u u^T) (geometric stiffness).
  Vec3 f = w_known.head<3>() + df - Vec3(0, 0, pl.m * P_.g + 0.5 * P_.cable.rho * P_.g * L.sum());
  Vec3 tau_b = Rt * w_known.tail<3>() + dtau;
  Mat3 Fp = Mat3::Zero(), Fth = Mat3::Zero(), Mp = Mat3::Zero(), Mth = hat(Rt * w_known.tail<3>());
  for (int i = 0; i < NC; ++i) {
    const Vec3 u = g.u.row(i);
    const Vec3 b = pl.b.row(i);
    const Mat3 G = (T(i) / g.d(i)) * (Mat3::Identity() - u * u.transpose());
    const Mat3 hb = hat(b);
    f += T(i) * u;
    tau_b += T(i) * b.cross(Rt * u);
    Fp -= G;
    Fth += G * R * hb;
    Mp -= hb * Rt * G;
    Mth += T(i) * hb * hat(Rt * u) + hb * Rt * G * R * hb;
  }
  const Vec3 a = f / pl.m;
  const Vec3 w_dot = Jinv * (tau_b - w.cross(pl.J * w));

  M18 A = M18::Zero();
  A.block<3, 3>(0, 3).setIdentity();
  A.block<3, 3>(3, 0) = Fp / pl.m;
  A.block<3, 3>(3, 6) = Fth / pl.m;
  A.block<3, 3>(3, 12) = Mat3::Identity() / pl.m;
  A.block<3, 3>(6, 6) = -hat(w);
  A.block<3, 3>(6, 9).setIdentity();
  A.block<3, 3>(9, 0) = Jinv * Mp;
  A.block<3, 3>(9, 6) = Jinv * Mth;
  A.block<3, 3>(9, 9) = Jinv * (hat(pl.J * w) - hat(w) * pl.J);
  A.block<3, 3>(9, 15) = Jinv;
  const M18 Phi = M18::Identity() + A * dt;

  p += v * dt + 0.5 * a * dt * dt;
  v += a * dt;
  R = R * expSO3(w * dt);
  w += w_dot * dt;

  const ObserverParams& o = P_.observer;
  P = Phi * P * Phi.transpose();
  P.diagonal().segment<3>(3).array() += o.q_acc * o.q_acc * dt;
  P.diagonal().segment<3>(9).array() += o.q_alpha * o.q_alpha * dt;
  P.diagonal().segment<3>(12).array() += o.q_df * o.q_df * dt;
  P.diagonal().segment<3>(15).array() += o.q_dtau * o.q_dtau * dt;
}

void PlatformObserver::scalar_update(const Eigen::Matrix<double, 1, 18>& H, double r, double var, double gate) {
  const Eigen::Matrix<double, 18, 1> PHt = P * H.transpose();
  const double S = H * PHt + var;
  if (gate > 0 && r * r > gate * gate * S) {
    ++rejected;
    return;
  }
  const Eigen::Matrix<double, 18, 1> K = PHt / S;
  P -= K * PHt.transpose();
  P = 0.5 * (P + P.transpose()).eval();
  const Eigen::Matrix<double, 18, 1> dx = K * r;
  p += dx.segment<3>(0);
  v += dx.segment<3>(3);
  R = R * expSO3(dx.segment<3>(6));
  w += dx.segment<3>(9);
  df += dx.segment<3>(12);
  dtau += dx.segment<3>(15);
}

void PlatformObserver::update_cable(const Params& P_, int i, double d_meas, const Vec3& anchor, double sigma) {
  const Vec3 b = P_.platform.b.row(i);
  const Vec3 dv = anchor - p - R * b;
  const double d = dv.norm();
  const Vec3 u = dv / d;
  Eigen::Matrix<double, 1, 18> H = Eigen::Matrix<double, 1, 18>::Zero();
  H.segment<3>(0) = -u.transpose();
  H.segment<3>(6) = u.transpose() * R * hat(b);
  scalar_update(H, d_meas - d, sigma * sigma, P_.observer.gate);
}

void PlatformObserver::update_tag(const Vec3& p_meas, const Mat3& R_meas, double sigma_p, double sigma_th,
                                  double gate) {
  for (int j = 0; j < 3; ++j) {
    Eigen::Matrix<double, 1, 18> H = Eigen::Matrix<double, 1, 18>::Zero();
    H(j) = 1.0;
    scalar_update(H, p_meas(j) - p(j), sigma_p * sigma_p, gate);
  }
  for (int j = 0; j < 3; ++j) {
    Eigen::Matrix<double, 1, 18> H = Eigen::Matrix<double, 1, 18>::Zero();
    H(6 + j) = 1.0;
    scalar_update(H, logSO3(R.transpose() * R_meas)(j), sigma_th * sigma_th, gate);
  }
}

Vec6 PlatformObserver::disturbance_world() const {
  Vec6 d;
  d << df, R * dtau;
  return d;
}

// ---------------------------------------------------------------------------------------------------- sensors
SensorSuite::SensorSuite(const SensorParams& s, double dt, int div_imu_)
    : P(s), dt_imu(dt * div_imu_), rate_imu(1.0 / (dt * div_imu_)), div_imu(div_imu_), rng(s.seed) {
  for (int k = 0; k < ND; ++k) {
    bg[k] = P.gyro_bias0 * n3();
    ba[k] = P.acc_bias0 * n3();
    acc_sum[k].setZero();
    gyr_sum[k].setZero();
  }
}

bool SensorSuite::imu(int k, const Rigid& s, const Vec3& acc_world, double g, Vec3& acc_m, Vec3& gyro_m) {
  acc_sum[k] += s.R.transpose() * (acc_world + Vec3(0, 0, g));  // specific force in the body frame
  gyr_sum[k] += s.w;
  if (++count[k] < div_imu) return false;
  bg[k] += P.gyro_rw * std::sqrt(dt_imu) * n3();
  ba[k] += P.acc_rw * std::sqrt(dt_imu) * n3();
  acc_m = acc_sum[k] / count[k] + ba[k] + P.acc_nd * std::sqrt(rate_imu) * n3();
  gyro_m = gyr_sum[k] / count[k] + bg[k] + P.gyro_nd * std::sqrt(rate_imu) * n3();
  acc_sum[k].setZero();
  gyr_sum[k].setZero();
  count[k] = 0;
  return true;
}

Vec3 SensorSuite::drone_fix(const Vec3& p) { return p + P.drone_fix_sigma * n3(); }
double SensorSuite::drone_yaw(double yaw) { return yaw + P.drone_yaw_sigma * n(); }
Vec2 SensorSuite::wheels(const Vec2& w) { return w + P.wheel_sigma * Vec2(n(), n()); }
Vec3 SensorSuite::ugv_fix(const Vec3& x) {
  return Vec3(x(0) + P.ugv_fix_sigma * n(), x(1) + P.ugv_fix_sigma * n(), x(2) + P.ugv_yaw_sigma * n());
}

Vec3 SensorSuite::fairlead(const Vec3& r) { return r + P.fair_sigma * n3(); }

void SensorSuite::winch(const Truth& t, Vec8& L, Vec8& Ldot, Vec8& T) {
  for (int i = 0; i < NC; ++i) {
    L(i) = t.L(i) + P.enc_sigma * n();
    Ldot(i) = t.Ldot(i) + P.enc_rate_sigma * n();
    T(i) = std::max(0.0, t.T(i) + P.load_sigma * n());
  }
}

void SensorSuite::tag(const Vec3& p_robot, const Mat3& R_robot, const Vec3& p, const Mat3& R, Vec3& p_rel,
                      Mat3& R_rel) {
  p_rel = R_robot.transpose() * (p - p_robot) + P.tag_sigma_p * n3();
  R_rel = R_robot.transpose() * R * expSO3(P.tag_sigma_th * n3());
}

}  // namespace cdpr
