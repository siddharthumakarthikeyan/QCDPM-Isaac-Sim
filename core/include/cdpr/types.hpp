// Types, parameters and SO(3) helpers shared by the model, the estimators and the controllers.
//
// Conventions (same as the simulation): SI units, world frame z up, R maps body -> world, cables 0-3 run from the
// platform's top corners to the drones, cables 4-7 from the bottom corners to the ground robots. Angular velocity
// is stored in the body frame (Omega) inside this library; the Python bridge converts from PhysX's world frame.
#pragma once

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <limits>

namespace cdpr {

constexpr int NC = 8;  // cables
constexpr int ND = 4;  // drones
constexpr int NU = 4;  // ground robots

using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Vec8 = Eigen::Matrix<double, 8, 1>;
using VecX = Eigen::VectorXd;
using Mat3 = Eigen::Matrix3d;
using Mat4 = Eigen::Matrix4d;
using Mat8 = Eigen::Matrix<double, 8, 8>;
using Mat68 = Eigen::Matrix<double, 6, 8>;
using Mat83 = Eigen::Matrix<double, 8, 3>;  // one row per cable
using Mat43 = Eigen::Matrix<double, 4, 3>;  // one row per robot
using Mat42 = Eigen::Matrix<double, 4, 2>;
using Mat44 = Eigen::Matrix<double, 4, 4>;
using MatX = Eigen::MatrixXd;

const double kNaN = std::numeric_limits<double>::quiet_NaN();

// ---------------------------------------------------------------------------------------------------- SO(3)
inline Mat3 hat(const Vec3& a) {
  Mat3 m;
  m << 0, -a.z(), a.y(), a.z(), 0, -a.x(), -a.y(), a.x(), 0;
  return m;
}

inline Mat3 expSO3(const Vec3& w) {
  const double th = w.norm();
  const Mat3 K = hat(w);
  if (th < 1e-8) return Mat3::Identity() + K + 0.5 * K * K;
  return Mat3::Identity() + std::sin(th) / th * K + (1.0 - std::cos(th)) / (th * th) * K * K;
}

inline Vec3 logSO3(const Mat3& R) {
  const Eigen::AngleAxisd aa(R);
  return aa.angle() * aa.axis();
}

// Attitude error of Lee et al. (2010): 0.5 vee(Rd^T R - R^T Rd), in the body frame of R.
inline Vec3 rot_error(const Mat3& Rd, const Mat3& R) {
  const Mat3 E = Rd.transpose() * R - R.transpose() * Rd;
  return 0.5 * Vec3(E(2, 1), E(0, 2), E(1, 0));
}

inline double wrap_pi(double a) { return std::atan2(std::sin(a), std::cos(a)); }
inline double yaw_of(const Mat3& R) { return std::atan2(R(1, 0), R(0, 0)); }

struct Rigid {
  Vec3 p = Vec3::Zero();
  Vec3 v = Vec3::Zero();
  Mat3 R = Mat3::Identity();
  Vec3 w = Vec3::Zero();  // body frame
};

// ---------------------------------------------------------------------------------------------------- parameters
struct CableParams {
  double EA = 1.0e5;    // axial stiffness [N]
  double cEA = 30.0;    // Kelvin-Voigt damping [N s]
  double rho = 0.0026;  // linear density [kg/m]
  double Lmin = 0.10, Lmax = 8.0;
};

struct WinchParams {  // in cable coordinates: m_eq L_ddot = F_motor + T - c_v L_dot - F_c sgn(L_dot)
  double m_eq = 5.0, c_v = 5.0, F_c = 1.0, F_max = 150.0, v_max = 1.0;
  double wn = 50.0, zeta = 0.9;  // length servo
};

struct PlatformParams {
  double m = 3.0;
  Mat3 J = Mat3::Identity() * 0.045;
  Mat83 b = Mat83::Zero();  // cable attachment points, platform frame
};

struct DroneParams {
  double m = 2.0;
  Mat3 J = Vec3(0.025, 0.025, 0.045).asDiagonal();
  double kf = 2.0e-5, km = 0.016, f_max = 20.0, tau = 0.025;
  double drag_lin = 0.30, drag_quad = 0.06, arm = 0.225;
  Vec3 r_att = Vec3(0, 0, -0.08);  // fairlead in the body frame
  // geometric SE(3) controller (gains per unit mass for kp, kd, ki)
  Vec3 kp = Vec3(6, 6, 8), kd = Vec3(4.5, 4.5, 5.5), ki = Vec3(0.5, 0.5, 1.0);
  double i_lim = 6.0;  // integral authority [N]
  Vec3 kR = Vec3(10, 10, 4), kW = Vec3(0.8, 0.8, 0.5);
  double max_tilt = 0.785;
  bool cable_ff = true;
};

struct UgvParams {
  double m = 20.0, Iz = 0.71;                 // whole robot
  double wheel_r = 0.10, track = 0.44, wheel_J = 0.005;
  double wheel_kd = 50.0, wheel_tau_max = 15.0, wheel_w_max = 15.0;
  double a_lin = 1.0, a_ang = 2.0;            // command shaping
  double h_att = 0.35;                        // fairlead height above the ground
  double mu = 0.8;                            // tyre-ground friction (dynamic)
  double tip_radius = 0.148;                  // inradius of the support polygon (wheels + casters)
  double v_max = 0.6, k_pos = 1.0, k_head = 2.0, k_yaw = 1.5, pos_tol = 0.03;  // go-to-goal law
};

struct TensionParams {
  double t_min = 5.0, t_max = 60.0, t_ref = 15.0;
  // The wrench weight is 1/lambda. lambda is negligible while the structure matrix is well conditioned and rises to
  // this value as its smallest singular value falls below sigma_eps (damped least squares near singular layouts).
  double lambda = 2.5e-3, lambda_min = 1e-6, sigma_eps = 0.1;
  double w_rate = 0.0;         // weight on |t - t_prev|^2 (continuity)
  double t_min_length = 5.0;   // lower bound planned for the length-mode cables (their tension is not regulated
                               // directly, so they get more margin against going slack)
  bool drone_caps = true;      // upper bounds from the drone tilt and thrust limits
  double tilt_cap = 0.733, thrust_cap = 68.0;
  bool ugv_caps = true;        // upper bounds from ground-robot traction and tip-over
  double ground_margin = 0.8;  // use this fraction of the traction / tip-over limit
};

struct PlatformGains {
  double kp_lin = 49, kd_lin = 12.6, ki_lin = 171.5;
  double kp_rot = 49, kd_rot = 12.6, ki_rot = 171.5;
  double i_lim = 2.0;  // integral authority [m/s^2 | rad/s^2]
  // Tension-mode winches: reel at the rate of the commanded twist = reference twist + k_twist x pose error
  // (limited to twist_max), so the winch damping opposes only departures from the corrective motion.
  double k_twist = 0.0, twist_max = 0.05;
  // Length-mode winches: the inverse-kinematics target pose is the reference shifted by k_ik x integral of the pose
  // error, so a steady error in the directions those cables hold stiffly is removed kinematically.
  double k_ik = 0.0;
  // Length-mode winches: slow admittance on the load cell, dL_dot = k_adm (T - t_des), |dL| <= adm_max. The cable
  // stays a stiff length servo above this loop's bandwidth and follows its tension set-point below it, so a
  // millimetre of anchor error no longer becomes several newtons of tension error.
  double k_adm = 0.0, adm_max = 0.01;
  double adm_band = 0.0;  // dead band [N]: tension errors inside it are left to the cable dynamics
  // Slack guard: a length-mode cable measured below t_guard is reeled in at k_guard (t_guard - T) on top of that.
  double t_guard = 0.0, k_guard = 0.0;
  // Reference governor: the platform reference is followed with bounded acceleration, so a velocity step in the
  // commanded reference does not become a tension impulse.
  double a_max = 1.5, alpha_max = 3.0, k_gov = 10.0;
};

struct SensorParams {
  bool noise = true;
  unsigned seed = 7;
  // IMU on every drone, sampled at the flight-controller rate
  double gyro_nd = 4.9e-5, acc_nd = 6.9e-4;  // noise density [rad/s/sqrt(Hz)], [m/s^2/sqrt(Hz)]
  double gyro_rw = 1e-5, acc_rw = 1e-4;      // bias random walk
  double gyro_bias0 = 3e-3, acc_bias0 = 2e-2;
  // position + heading fix of each drone (RTK / motion-capture class)
  int div_drone_fix = 20;
  double drone_fix_sigma = 0.005, drone_yaw_sigma = 0.0087;
  // ground robots: wheel encoders at the base rate, planar pose fix from lidar localisation
  double wheel_sigma = 0.02;
  double fair_sigma = 4e-4;  // fairlead position from the base inclinometer (0.1 deg on a 0.2 m mast) [m]
  int div_ugv_fix = 67;
  double ugv_fix_sigma = 0.005, ugv_yaw_sigma = 0.0052;
  // winches: drum encoder and load cell
  double enc_sigma = 3e-5, enc_rate_sigma = 1e-3, load_sigma = 0.1;
  // platform pose from an AprilTag seen by one robot camera at a time (round robin), relative to that robot
  int div_tag = 33;
  double tag_sigma_p = 0.003, tag_sigma_th = 0.0052;
};

struct ObserverParams {
  // platform observer process noise (continuous densities)
  double q_acc = 1.0, q_alpha = 2.0, q_df = 6.0, q_dtau = 0.4;
  double sigma_cable_drone = 0.01, sigma_cable_ugv = 0.004;  // cable-length measurement (incl. anchor error)
  double sigma_tag_p = 0.02, sigma_tag_th = 0.01;
  double t_slack = 1.0;  // a cable below this tension carries no length information
  bool sag = false;      // parabolic sag correction of the chord length
  double gate = 5.0;     // innovation gate [sigma]
  double fair_tau = 0.1;  // low-pass on the ground robots' fairlead offset [s]
  // drone filter
  double drone_acc_sigma = 0.05, drone_gyro_sigma = 2e-3;  // per-sample input noise used by the filter
};

struct Params {
  double g = 9.81, dt = 1e-3;
  int div_drone = 2, div_platform = 4, div_ugv = 10;
  bool hybrid = true;      // drone winches regulate tension, ground winches regulate length
  bool use_truth = false;  // bypass the estimators (controllers read the true state)
  bool dob_ff = true;      // feed the estimated disturbance wrench forward
  PlatformParams platform;
  CableParams cable;
  WinchParams winch;
  DroneParams drone;
  UgvParams ugv;
  TensionParams tension;
  PlatformGains gains;
  SensorParams sensors;
  ObserverParams observer;
};

}  // namespace cdpr
