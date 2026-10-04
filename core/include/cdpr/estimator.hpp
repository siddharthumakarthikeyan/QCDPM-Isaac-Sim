// State estimation from the sensors the robots carry.
//
//   DroneEskf        IMU-driven error-state Kalman filter (p, v, R, gyro bias, accel bias), corrected by a position
//                    and heading fix.
//   UgvEkf           planar pose from wheel odometry, corrected by a pose fix (lidar localisation).
//   PlatformObserver error-state Kalman filter on the platform's rigid-body dynamics. It is driven by the measured
//                    cable tensions (load cells) and corrected by the cable lengths (winch encoders + the elastic
//                    law) and by AprilTag poses from the robot cameras. It also estimates the external wrench on
//                    the platform (payload, tool, model error).
//   SensorSuite      turns true states into measurements: rates, white noise, biases. Used by the C++ plant and by
//                    the Isaac bridge, so both closed loops see the same sensors.
#pragma once

#include <random>

#include "cdpr/model.hpp"

namespace cdpr {

class DroneEskf {
 public:
  using M15 = Eigen::Matrix<double, 15, 15>;
  Vec3 p = Vec3::Zero(), v = Vec3::Zero(), bg = Vec3::Zero(), ba = Vec3::Zero();
  Mat3 R = Mat3::Identity();
  Vec3 w_body = Vec3::Zero();     // bias-corrected body rate (latest sample)
  Vec3 acc_world = Vec3::Zero();  // latest acceleration estimate
  M15 P = M15::Identity();
  bool initialised = false;

  void init(const Vec3& p0, double yaw0);
  void predict(const Vec3& acc_m, const Vec3& gyro_m, double dt, double g, const ObserverParams& o,
               const SensorParams& s);
  void update_position(const Vec3& z, double sigma);
  void update_yaw(double z, double sigma);

 private:
  void inject(const Eigen::Matrix<double, 15, 1>& dx);
  void scalar_update(const Eigen::Matrix<double, 1, 15>& H, double r, double var);
};

class UgvEkf {
 public:
  Vec3 x = Vec3::Zero();  // x, y, yaw
  double v = 0.0, om = 0.0;
  // Unmodelled slip [m/sqrt(s)], [rad/sqrt(s)]. Kept small on purpose: the fix then enters with a gain of about
  // 0.02, so the anchor estimate does not jitter (a length-controlled cable turns 0.1 mm into 3 N).
  double slip_xy = 4e-4, slip_yaw = 1e-3;
  double slip_ratio = 0.0;   // extra slip random walk per unit speed (off: it makes the anchors jitter while driving)
  Mat3 P = Mat3::Identity() * 1e-8;  // the robots start at surveyed positions
  void predict(double wl, double wr, double dt, const UgvParams& u, double wheel_sigma);
  void update(const Vec3& z, double sigma_xy, double sigma_yaw);
};

class PlatformObserver {
 public:
  using M18 = Eigen::Matrix<double, 18, 18>;
  Vec3 p = Vec3::Zero(), v = Vec3::Zero(), w = Vec3::Zero();  // w: body rates
  Mat3 R = Mat3::Identity();
  Vec3 df = Vec3::Zero();    // external force, world [N]
  Vec3 dtau = Vec3::Zero();  // external torque, body [N m]
  M18 P = M18::Identity();
  int rejected = 0;          // measurements refused by the innovation gate

  void init(const Vec3& p0, const Mat3& R0);
  // T: measured tensions, anchors: estimated fairlead positions, w_known: known external wrench (world, about CoM)
  void predict(const Params& P_, const Vec8& T, const Vec8& L, const Mat83& anchors, const Vec6& w_known, double dt);
  // chord length of cable i (from the encoder and the load cell)
  void update_cable(const Params& P_, int i, double d_meas, const Vec3& anchor, double sigma);
  void update_tag(const Vec3& p_meas, const Mat3& R_meas, double sigma_p, double sigma_th, double gate);
  Vec6 disturbance_world() const;

 private:
  void scalar_update(const Eigen::Matrix<double, 1, 18>& H, double r, double var, double gate);
};

// ---------------------------------------------------------------------------------------------------- sensors
struct Truth {
  // platform
  Vec3 p = Vec3::Zero(), v = Vec3::Zero(), w = Vec3::Zero();  // w: body rates
  Mat3 R = Mat3::Identity();
  // drones
  std::array<Rigid, ND> drone;
  Mat43 drone_acc = Mat43::Zero();  // world acceleration of each drone's centre of mass
  // ground robots: x, y, yaw and wheel speeds (left, right)
  Mat43 ugv = Mat43::Zero();
  Mat42 wheel = Mat42::Zero();
  // fairlead relative to the point on the ground under the chassis origin (world axes). The base rocks on its
  // casters under the cable pull, which moves the mast top by a few millimetres.
  Mat43 ugv_fair = Mat43::Zero();
  // winches / cables
  Vec8 L = Vec8::Constant(1.0), Ldot = Vec8::Zero(), T = Vec8::Zero();
};

class SensorSuite {
 public:
  explicit SensorSuite(const SensorParams& s, double dt, int div_imu);
  // IMU: accumulate every physics step; `ready` when a sample (averaged over the period) is available
  bool imu(int k, const Rigid& s, const Vec3& acc_world, double g, Vec3& acc_m, Vec3& gyro_m);
  Vec3 drone_fix(const Vec3& p);
  double drone_yaw(double yaw);
  Vec2 wheels(const Vec2& w);
  Vec3 ugv_fix(const Vec3& x);
  Vec3 fairlead(const Vec3& r);  // from the base's inclinometer
  void winch(const Truth& t, Vec8& L, Vec8& Ldot, Vec8& T);
  // platform pose relative to a robot's camera frame (= the robot body frame; the gimbal is ideal)
  void tag(const Vec3& p_robot, const Mat3& R_robot, const Vec3& p, const Mat3& R, Vec3& p_rel, Mat3& R_rel);

 private:
  double n() { return P.noise ? gauss(rng) : 0.0; }
  Vec3 n3() { return Vec3(n(), n(), n()); }
  SensorParams P;
  double dt_imu, rate_imu;
  int div_imu;
  std::mt19937_64 rng;
  std::normal_distribution<double> gauss{0.0, 1.0};
  std::array<Vec3, ND> bg, ba, acc_sum, gyr_sum;
  std::array<int, ND> count{};
};

}  // namespace cdpr
