// Controllers, from the single robot to the whole machine.
//
//   DroneController     geometric SE(3) tracking with the cable force (load cell x estimated direction) fed forward
//   UgvController       go-to-goal law of the differential-drive base with command shaping
//   PlatformController  SE(3) tracking wrench -> bounded tension distribution -> winch commands (hybrid: drone
//                       winches regulate tension, ground winches regulate length)
//   Core                sensors -> estimators -> controllers at their own rates. One call per physics step.
#pragma once

#include "cdpr/estimator.hpp"

namespace cdpr {

struct PlatformRef {
  Vec3 p = Vec3::Zero(), v = Vec3::Zero(), a = Vec3::Zero();
  Mat3 R = Mat3::Identity();
  Vec3 w = Vec3::Zero();  // angular velocity of the reference, in the reference frame
};

struct Refs {
  PlatformRef platform;
  Mat43 drone_p = Mat43::Zero(), drone_v = Mat43::Zero(), drone_a = Mat43::Zero();
  Vec4 drone_yaw = Vec4::Zero();
  Mat42 ugv_goal = Mat42::Zero(), ugv_goal_vel = Mat42::Zero();
  Vec4 ugv_yaw = Vec4::Constant(kNaN);           // NaN: keep the heading on arrival
  Eigen::Matrix<int, 4, 1> ugv_velocity_mode = Eigen::Matrix<int, 4, 1>::Zero();
  Vec4 ugv_cmd_v = Vec4::Zero(), ugv_cmd_om = Vec4::Zero();
};

struct Estimate {
  Vec3 p = Vec3::Zero(), v = Vec3::Zero(), w = Vec3::Zero();
  Mat3 R = Mat3::Identity();
  Vec6 d_hat = Vec6::Zero();  // external wrench on the platform (world)
  std::array<Rigid, ND> drone;
  Mat43 ugv = Mat43::Zero();  // x, y, yaw
  Mat42 ugv_vw = Mat42::Zero();
  Mat43 ugv_fair = Mat43::Zero();
  Mat83 anchors = Mat83::Zero(), anchor_vel = Mat83::Zero();
  Vec8 L = Vec8::Constant(1.0), Ldot = Vec8::Zero(), T = Vec8::Zero();
};

class DroneController {
 public:
  Vec3 e_int = Vec3::Zero();
  // F_ext: estimated external force on the drone (world). Returns the four rotor thrust commands.
  Vec4 step(const DroneParams& d, const Mat44& mixer_inv, double g, const Rigid& s, const Vec3& p_ref,
            const Vec3& v_ref, const Vec3& a_ref, double yaw_ref, const Vec3& F_ext, double dt);
};

class UgvController {
 public:
  double v = 0.0, om = 0.0;  // shaped commands
  // Returns wheel speed targets (left, right).
  Vec2 step(const UgvParams& u, const Vec3& pose, const Vec2& goal, const Vec2& goal_vel, double yaw_goal,
            bool velocity_mode, double cmd_v, double cmd_om, double dt);
};

class PlatformController {
 public:
  Vec6 e_int = Vec6::Zero();
  Vec8 t_prev = Vec8::Constant(15.0);
  Vec8 dL = Vec8::Zero();  // admittance trim of the length-mode cables [m]
  // outputs of the last step
  Vec6 wrench = Vec6::Zero();
  Vec8 t_des = Vec8::Zero(), t_cap = Vec8::Zero();
  TdResult td;
  double sigma_min = 0.0;  // smallest singular value of the structure matrix

  WinchCmd step(const Params& P, const PlatformRef& ref, const Estimate& est, const Vec6& w_known, double dt);
  // acceleration-limited copy of the commanded reference (state of the governor)
  PlatformRef governed(const PlatformGains& k, const PlatformRef& ref, double dt);
  PlatformRef gov;
  bool gov_init = false;
};

struct Commands {
  WinchCmd winch;
  Mat44 rotor_f = Mat44::Zero();  // one row per drone
  Mat42 wheel_w = Mat42::Zero();  // one row per ground robot: left, right [rad/s]
};

class Core {
 public:
  explicit Core(const Params& P);
  // Set the estimators to a known starting state (the robots start at rest at known positions).
  void reset(const Truth& t);
  // One physics step: sample the sensors, update the estimators, run whichever controllers are due.
  void step(const Truth& t, const Refs& refs, const Vec6& w_known);

  Params P;
  Estimate est;
  Commands cmd;
  PlatformController platform;
  std::array<DroneController, ND> drone;
  std::array<UgvController, NU> ugv;
  std::array<DroneEskf, ND> drone_filter;
  std::array<UgvEkf, NU> ugv_filter;
  PlatformObserver observer;
  SensorSuite sensors;
  long k = 0;
  int tag_robot = 0;
  Vec3 tag_p = Vec3::Zero();    // last tag-derived platform position (world), for logging
  Vec8 d_meas = Vec8::Zero();   // last chord lengths derived from the encoders and load cells

 private:
  void estimate(const Truth& t, const Vec6& w_known);
  Mat44 mixer_inv;
};

}  // namespace cdpr
