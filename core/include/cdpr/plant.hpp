// The coupled plant (platform + 8 cables and winches + 4 quadrotors with rotor dynamics + 4 differential-drive
// robots) closed with Core, integrated with semi-implicit Euler at the physics rate. It is the Isaac-free
// reference for the same controller that runs in Isaac Sim, and it provides the linearised closed-loop map used
// for the stability analysis.
#pragma once

#include "cdpr/controller.hpp"

namespace cdpr {

class Simulator {
 public:
  // anchors0: fairlead positions (rows 0-3 drones, 4-7 ground robots); ugv_yaw0: headings of the ground robots
  Simulator(const Params& P, const Vec3& p0, const Mat83& anchors0, const Vec4& ugv_yaw0);
  // Start in static equilibrium at the current reference (call after setting w_known).
  void initialise();
  void step();
  void run(int n);

  // Jacobian of the closed-loop map over n_steps physics steps about the current state, in error-state
  // coordinates, by central differences. Requires use_truth (deterministic loop). With the ground robots frozen
  // the state is [platform 12 | drones 4 x (12 + 4 rotors) | winches 16 | integrators 6 + 12 | t_prev 8 | trims of the length-mode cables].
  MatX monodromy(int n_steps, double eps = 1e-6);
  VecX diff(const Simulator& ref) const;
  void perturb(const VecX& dx);
  int state_dim() const { return 12 + ND * 16 + 16 + 6 + 12 + 8 + (P.hybrid ? NU : NC) + (freeze_ugv ? 0 : NU * 7); }

  Truth truth() const;

  Params P;
  Core core;
  Refs refs;
  // true state
  Rigid platform;
  std::array<Rigid, ND> drone;
  Mat44 rotor_w = Mat44::Zero();  // one row per drone
  Mat43 ugv = Mat43::Zero();      // x, y, yaw
  Mat42 ugv_vw = Mat42::Zero();   // forward speed, yaw rate
  Vec8 L = Vec8::Constant(1.0), Ldot = Vec8::Zero(), T = Vec8::Zero();
  Mat43 drone_acc = Mat43::Zero();
  // inputs
  Vec6 w_known = Vec6::Zero();     // wrench on the platform that the controller is told about (tool load cell)
  Vec6 ext_wrench = Vec6::Zero();  // wrench on the platform that it is not told about
  bool freeze_ugv = false;
  // diagnostics
  Vec4 traction_use = Vec4::Zero();  // friction demand / friction available at each ground robot
  double time = 0.0;

 private:
  Mat44 mixer;
  Mat3 Jd_inv, Jp_inv;
};

}  // namespace cdpr
