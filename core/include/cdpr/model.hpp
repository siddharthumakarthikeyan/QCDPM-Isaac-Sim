// Physical model of the mobile CDPR: cable kinematics and statics, cable and winch dynamics, quadrotor and
// differential-drive dynamics, actuator-limit tension caps, tension distribution. docs/model/cdpr_model.tex has
// the derivations; equation numbers in the comments refer to it.
#pragma once

#include "cdpr/types.hpp"

namespace cdpr {

// ---------------------------------------------------------------------------------------------------- cables
struct CableGeom {
  Mat83 u;    // unit vectors, platform attachment point -> anchor (world)
  Mat83 rb;   // R b_i (world)
  Vec8 d;     // chord lengths
  Mat68 W;    // structure matrix: columns [u_i ; (R b_i) x u_i]
};

CableGeom cable_geometry(const Vec3& p, const Mat3& R, const Mat83& b, const Mat83& anchors);

// d_dot_i = u_i . (v_anchor_i - v - omega_world x R b_i)
Vec8 cable_rates(const CableGeom& g, const Vec3& v, const Vec3& w_world, const Mat83& anchor_vel);

// Tension-only Kelvin-Voigt cable.
Vec8 cable_tension(const CableParams& c, const Vec8& L, const Vec8& Ldot, const Vec8& d, const Vec8& d_dot);

// Chord length of a cable with unstretched length L carrying tension T (inverse of the elastic law); with `sag`
// the parabolic-sag shortening of a cable at elevation sin = u_z is subtracted.
double chord_length(const CableParams& c, double L, double T, double u_z, double g, bool sag);
inline double unstretched_length(const CableParams& c, double d, double T) { return d / (1.0 + T / c.EA); }

// Pose from the eight chord lengths (Gauss-Newton on SE(3), Levenberg damping). Returns the rms residual [m].
double forward_kinematics(const Vec8& d, const Mat83& b, const Mat83& anchors, Vec3& p, Mat3& R, int max_iter = 20);

// ---------------------------------------------------------------------------------------------------- winches
struct WinchCmd {
  Vec8 L_cmd = Vec8::Constant(1.0);
  Vec8 Ld_cmd = Vec8::Zero();
  Vec8 T_ff = Vec8::Zero();
  Eigen::Matrix<int, 8, 1> tension_mode = Eigen::Matrix<int, 8, 1>::Zero();  // 1: regulate tension, 0: length
};

// One explicit step of the drum dynamics with the length / tension servo and the motor envelope.
void winch_step(const WinchParams& w, const CableParams& c, const WinchCmd& cmd, const Vec8& T, Vec8& L, Vec8& Ldot,
                double dt);

// ---------------------------------------------------------------------------------------------------- quadrotor
Mat44 quad_mixer(const DroneParams& d);  // [thrust, Mx, My, Mz] = mixer * rotor thrusts

// Rotor thrusts + drag -> world force and body torque (gravity and cable not included).
void quad_wrench(const DroneParams& d, const Mat44& mixer, const Rigid& s, const Vec4& rotor_w, Vec3& F_world,
                 Vec3& tau_body);

// ---------------------------------------------------------------------------------------------------- limits
// Largest tension a drone cable may carry before the drone exceeds its tilt or thrust limit (eq. caps-drone).
double drone_tension_cap(const Vec3& u, double m, double g, double theta_max, double f_max);
// Largest tension a ground-robot cable may carry before the robot slides or tips (eq. caps-ugv).
double ugv_tension_cap(const Vec3& u, const UgvParams& p, double g, double margin);

// ---------------------------------------------------------------------------------------------------- tensions
struct TdResult {
  Vec8 t = Vec8::Zero();
  Vec6 residual = Vec6::Zero();  // W t - w
  bool feasible = false;         // the bounds cost (almost) no wrench accuracy
  int iterations = 0;
  int n_active = 0;
};

// min 0.5 x^T H x + g^T x  s.t. lo <= x <= hi  (H symmetric positive definite), primal active set.
// x is the starting point on entry. Returns the number of iterations.
int solve_box_qp(const Mat8& H, const Vec8& g, const Vec8& lo, const Vec8& hi, Vec8& x, int* n_active = nullptr);

// min |t - t_ref|^2 + w_rate |t - t_prev|^2 + (1/lambda) |W t - w|^2  s.t. lo <= t <= hi.
TdResult tension_distribution(const Mat68& W, const Vec6& w, const Vec8& lo, const Vec8& hi, double t_ref,
                              const Vec8& t_prev, double lambda, double w_rate);

// Damping for the distribution: lambda_min while sigma_min(W) >= sigma_eps, rising smoothly to lambda at sigma_min = 0.
double adaptive_lambda(const Mat68& W, const TensionParams& t, double* sigma_min = nullptr);

// Per-cable tension bounds at a given cable geometry.
void tension_bounds(const Params& P, const Mat83& u, Vec8& lo, Vec8& hi);

}  // namespace cdpr
