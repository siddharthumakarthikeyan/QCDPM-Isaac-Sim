// Python bindings of the C++ core (module cdpr_core).
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cdpr/planning.hpp"
#include "cdpr/plant.hpp"

namespace py = pybind11;
using namespace cdpr;

#define RW(cls, name) def_readwrite(#name, &cls::name)

PYBIND11_MODULE(cdpr_core, m) {
  m.doc() = "Mobile CDPR: model, estimation and control (C++ core)";

  py::class_<CableParams>(m, "CableParams").def(py::init<>())
      .RW(CableParams, EA).RW(CableParams, cEA).RW(CableParams, rho).RW(CableParams, Lmin).RW(CableParams, Lmax);
  py::class_<WinchParams>(m, "WinchParams").def(py::init<>())
      .RW(WinchParams, m_eq).RW(WinchParams, c_v).RW(WinchParams, F_c).RW(WinchParams, F_max).RW(WinchParams, v_max)
      .RW(WinchParams, wn).RW(WinchParams, zeta);
  py::class_<PlatformParams>(m, "PlatformParams").def(py::init<>())
      .RW(PlatformParams, m).RW(PlatformParams, J).RW(PlatformParams, b);
  py::class_<DroneParams>(m, "DroneParams").def(py::init<>())
      .RW(DroneParams, m).RW(DroneParams, J).RW(DroneParams, kf).RW(DroneParams, km).RW(DroneParams, f_max)
      .RW(DroneParams, tau).RW(DroneParams, drag_lin).RW(DroneParams, drag_quad).RW(DroneParams, arm)
      .RW(DroneParams, r_att).RW(DroneParams, kp).RW(DroneParams, kd).RW(DroneParams, ki).RW(DroneParams, i_lim)
      .RW(DroneParams, kR).RW(DroneParams, kW).RW(DroneParams, max_tilt).RW(DroneParams, cable_ff);
  py::class_<UgvParams>(m, "UgvParams").def(py::init<>())
      .RW(UgvParams, m).RW(UgvParams, Iz).RW(UgvParams, wheel_r).RW(UgvParams, track).RW(UgvParams, wheel_J)
      .RW(UgvParams, wheel_kd).RW(UgvParams, wheel_tau_max).RW(UgvParams, wheel_w_max).RW(UgvParams, a_lin)
      .RW(UgvParams, a_ang).RW(UgvParams, h_att).RW(UgvParams, mu).RW(UgvParams, tip_radius).RW(UgvParams, v_max)
      .RW(UgvParams, k_pos).RW(UgvParams, k_head).RW(UgvParams, k_yaw).RW(UgvParams, pos_tol);
  py::class_<TensionParams>(m, "TensionParams").def(py::init<>())
      .RW(TensionParams, t_min).RW(TensionParams, t_max).RW(TensionParams, t_ref).def_readwrite("lam", &TensionParams::lambda).RW(TensionParams, lambda_min).RW(TensionParams, sigma_eps)
      .RW(TensionParams, w_rate).RW(TensionParams, t_min_length).RW(TensionParams, drone_caps).RW(TensionParams, tilt_cap)
      .RW(TensionParams, thrust_cap).RW(TensionParams, ugv_caps).RW(TensionParams, ground_margin);
  py::class_<PlatformGains>(m, "PlatformGains").def(py::init<>())
      .RW(PlatformGains, kp_lin).RW(PlatformGains, kd_lin).RW(PlatformGains, ki_lin).RW(PlatformGains, kp_rot)
      .RW(PlatformGains, kd_rot).RW(PlatformGains, ki_rot).RW(PlatformGains, i_lim).RW(PlatformGains, k_twist)
      .RW(PlatformGains, twist_max).RW(PlatformGains, k_ik).RW(PlatformGains, k_adm)
      .RW(PlatformGains, adm_max).RW(PlatformGains, adm_band).RW(PlatformGains, t_guard)
      .RW(PlatformGains, k_guard).RW(PlatformGains, a_max).RW(PlatformGains, alpha_max).RW(PlatformGains, k_gov);
  py::class_<SensorParams>(m, "SensorParams").def(py::init<>())
      .RW(SensorParams, noise).RW(SensorParams, seed).RW(SensorParams, gyro_nd).RW(SensorParams, acc_nd)
      .RW(SensorParams, gyro_rw).RW(SensorParams, acc_rw).RW(SensorParams, gyro_bias0).RW(SensorParams, acc_bias0)
      .RW(SensorParams, div_drone_fix).RW(SensorParams, drone_fix_sigma).RW(SensorParams, drone_yaw_sigma)
      .RW(SensorParams, wheel_sigma).RW(SensorParams, fair_sigma).RW(SensorParams, div_ugv_fix).RW(SensorParams, ugv_fix_sigma)
      .RW(SensorParams, ugv_yaw_sigma).RW(SensorParams, enc_sigma).RW(SensorParams, enc_rate_sigma)
      .RW(SensorParams, load_sigma).RW(SensorParams, div_tag).RW(SensorParams, tag_sigma_p)
      .RW(SensorParams, tag_sigma_th);
  py::class_<ObserverParams>(m, "ObserverParams").def(py::init<>())
      .RW(ObserverParams, q_acc).RW(ObserverParams, q_alpha).RW(ObserverParams, q_df).RW(ObserverParams, q_dtau)
      .RW(ObserverParams, sigma_cable_drone).RW(ObserverParams, sigma_cable_ugv).RW(ObserverParams, sigma_tag_p)
      .RW(ObserverParams, sigma_tag_th).RW(ObserverParams, t_slack).RW(ObserverParams, sag).RW(ObserverParams, gate)
      .RW(ObserverParams, fair_tau).RW(ObserverParams, drone_acc_sigma).RW(ObserverParams, drone_gyro_sigma);
  py::class_<Params>(m, "Params").def(py::init<>())
      .RW(Params, g).RW(Params, dt).RW(Params, div_drone).RW(Params, div_platform).RW(Params, div_ugv)
      .RW(Params, hybrid).RW(Params, use_truth).RW(Params, dob_ff).RW(Params, platform).RW(Params, cable)
      .RW(Params, winch).RW(Params, drone).RW(Params, ugv).RW(Params, tension).RW(Params, gains)
      .RW(Params, sensors).RW(Params, observer);

  py::class_<Rigid>(m, "Rigid").def(py::init<>()).RW(Rigid, p).RW(Rigid, v).RW(Rigid, R).RW(Rigid, w);
  py::class_<Truth>(m, "Truth").def(py::init<>())
      .RW(Truth, p).RW(Truth, v).RW(Truth, w).RW(Truth, R).RW(Truth, drone).RW(Truth, drone_acc).RW(Truth, ugv)
      .RW(Truth, wheel).RW(Truth, ugv_fair).RW(Truth, L).RW(Truth, Ldot).RW(Truth, T)
      // drones in one call: positions, velocities, rotations (list of 3x3), body rates
      .def("set_drones", [](Truth& t, const Mat43& p, const Mat43& v, const std::array<Mat3, ND>& R, const Mat43& w) {
        for (int k = 0; k < ND; ++k) t.drone[k].p = p.row(k), t.drone[k].v = v.row(k), t.drone[k].R = R[k],
                                     t.drone[k].w = w.row(k);
      });
  py::class_<PlatformRef>(m, "PlatformRef").def(py::init<>())
      .RW(PlatformRef, p).RW(PlatformRef, v).RW(PlatformRef, a).RW(PlatformRef, R).RW(PlatformRef, w);
  py::class_<Refs>(m, "Refs").def(py::init<>())
      .RW(Refs, platform).RW(Refs, drone_p).RW(Refs, drone_v).RW(Refs, drone_a).RW(Refs, drone_yaw)
      .RW(Refs, ugv_goal).RW(Refs, ugv_goal_vel).RW(Refs, ugv_yaw).RW(Refs, ugv_velocity_mode)
      .RW(Refs, ugv_cmd_v).RW(Refs, ugv_cmd_om);
  py::class_<Estimate>(m, "Estimate")
      .RW(Estimate, p).RW(Estimate, v).RW(Estimate, w).RW(Estimate, R).RW(Estimate, d_hat).RW(Estimate, drone)
      .RW(Estimate, ugv).RW(Estimate, ugv_vw).RW(Estimate, ugv_fair).RW(Estimate, anchors).RW(Estimate, anchor_vel).RW(Estimate, L)
      .RW(Estimate, Ldot).RW(Estimate, T);
  py::class_<WinchCmd>(m, "WinchCmd")
      .RW(WinchCmd, L_cmd).RW(WinchCmd, Ld_cmd).RW(WinchCmd, T_ff).RW(WinchCmd, tension_mode);
  py::class_<Commands>(m, "Commands").RW(Commands, winch).RW(Commands, rotor_f).RW(Commands, wheel_w);
  py::class_<TdResult>(m, "TdResult")
      .RW(TdResult, t).RW(TdResult, residual).RW(TdResult, feasible).RW(TdResult, iterations)
      .RW(TdResult, n_active);
  py::class_<PlatformController>(m, "PlatformController")
      .RW(PlatformController, e_int).RW(PlatformController, t_prev).RW(PlatformController, dL).RW(PlatformController, wrench)
      .RW(PlatformController, t_des).RW(PlatformController, t_cap).RW(PlatformController, td)
      .RW(PlatformController, sigma_min).RW(PlatformController, gov);
  py::class_<DroneEskf>(m, "DroneEskf").RW(DroneEskf, p).RW(DroneEskf, v).RW(DroneEskf, R).RW(DroneEskf, bg)
      .RW(DroneEskf, ba);
  py::class_<PlatformObserver>(m, "PlatformObserver")
      .RW(PlatformObserver, p).RW(PlatformObserver, v).RW(PlatformObserver, R).RW(PlatformObserver, w)
      .RW(PlatformObserver, df).RW(PlatformObserver, dtau).RW(PlatformObserver, rejected)
      .def_property_readonly("sigma", [](const PlatformObserver& o) { return VecX(o.P.diagonal().cwiseSqrt()); });

  py::class_<Core>(m, "Core")
      .def(py::init<const Params&>())
      .def("reset", &Core::reset)
      .def("step", &Core::step)
      .RW(Core, P).RW(Core, est).RW(Core, cmd).RW(Core, platform).RW(Core, observer).RW(Core, drone_filter)
      .RW(Core, k).RW(Core, tag_p).RW(Core, d_meas);

  py::class_<Simulator>(m, "Simulator")
      .def(py::init<const Params&, const Vec3&, const Mat83&, const Vec4&>())
      .def("initialise", &Simulator::initialise)
      .def("step", &Simulator::step)
      .def("run", &Simulator::run)
      .def("monodromy", &Simulator::monodromy, py::arg("n_steps"), py::arg("eps") = 1e-6)
      .def("state_dim", &Simulator::state_dim)
      .def("truth", &Simulator::truth)
      .def("copy", [](const Simulator& s) { return Simulator(s); })
      .RW(Simulator, P).RW(Simulator, core).RW(Simulator, refs).RW(Simulator, platform).RW(Simulator, drone)
      .RW(Simulator, rotor_w).RW(Simulator, ugv).RW(Simulator, ugv_vw).RW(Simulator, L).RW(Simulator, Ldot)
      .RW(Simulator, T).RW(Simulator, w_known).RW(Simulator, ext_wrench).RW(Simulator, freeze_ugv)
      .RW(Simulator, traction_use).RW(Simulator, time);

  // perception and planning
  py::class_<Box2>(m, "Box2").def(py::init<>())
      .def(py::init([](const Vec2& lo, const Vec2& hi) { return Box2{lo, hi}; })).RW(Box2, lo).RW(Box2, hi);
  m.def("lidar_scan", &lidar_scan);
  py::class_<OccupancyGrid>(m, "OccupancyGrid")
      .def(py::init<const Vec2&, double, int, int>())
      .def("insert_scan", &OccupancyGrid::insert_scan)
      .def("update_distance", &OccupancyGrid::update_distance)
      .def("distance", &OccupancyGrid::distance)
      .def("map", &OccupancyGrid::map)
      .RW(OccupancyGrid, origin).RW(OccupancyGrid, res).RW(OccupancyGrid, nx).RW(OccupancyGrid, ny);
  py::class_<TeamFootprint>(m, "TeamFootprint").def(py::init<>())
      .RW(TeamFootprint, ugv_offset).RW(TeamFootprint, robot_radius).RW(TeamFootprint, cable_margin)
      .RW(TeamFootprint, body_radius);
  py::class_<TeamPlanner>(m, "TeamPlanner")
      .def(py::init<const OccupancyGrid&, const TeamFootprint&>(), py::keep_alive<1, 2>())
      .def("valid", &TeamPlanner::valid)
      .def("segment_valid", &TeamPlanner::segment_valid)
      .def("plan", [](const TeamPlanner& p, const Vec2& a, const Vec2& b) {
        int n = 0;
        auto path = p.plan(a, b, &n);
        return py::make_tuple(path, n);
      });

  py::class_<Box3>(m, "Box3").def(py::init([](const Vec3& lo, const Vec3& hi) { return Box3{lo, hi}; }))
      .RW(Box3, lo).RW(Box3, hi);
  m.def("depth_scan", &depth_scan);
  py::class_<ElevationGrid>(m, "ElevationGrid")
      .def(py::init<const Vec2&, double, int, int>())
      .def("insert_points", &ElevationGrid::insert_points)
      .def("insert_lidar", &ElevationGrid::insert_lidar)
      .def("update", &ElevationGrid::update)
      .def("height", &ElevationGrid::height, py::arg("p"), py::arg("layer") = 0)
      .def("map", &ElevationGrid::map)
      .def("seen", &ElevationGrid::seen);
  py::class_<TeamShape>(m, "TeamShape").def(py::init<>())
      .RW(TeamShape, ugv_offset).RW(TeamShape, drone_offset).RW(TeamShape, corner).RW(TeamShape, fairlead_h)
      .RW(TeamShape, drone_alt).RW(TeamShape, half_side).RW(TeamShape, tool_drop).RW(TeamShape, robot_radius)
      .RW(TeamShape, cable_margin).RW(TeamShape, body_radius).RW(TeamShape, clearance).RW(TeamShape, step_max)
      .RW(TeamShape, z_min).RW(TeamShape, z_max).RW(TeamShape, dz).RW(TeamShape, climb_cost)
      .RW(TeamShape, scales).RW(TeamShape, feasible).RW(TeamShape, scale_cost);
  py::class_<TeamPlanner3>(m, "TeamPlanner3")
      .def(py::init<const ElevationGrid&, const TeamShape&>(), py::keep_alive<1, 2>())
      .def("valid", &TeamPlanner3::valid)
      .def("segment_valid", &TeamPlanner3::segment_valid)
      .def("valid4", &TeamPlanner3::valid4)
      .def("plan4", [](const TeamPlanner3& p, const Vec4& a, const Vec4& b) {
        int n = 0;
        auto path = p.plan4(a, b, &n);
        return py::make_tuple(path, n);
      })
      .def("plan", [](const TeamPlanner3& p, const Vec3& a, const Vec3& b) {
        int n = 0;
        auto path = p.plan(a, b, &n);
        return py::make_tuple(path, n);
      });

  // model functions
  m.def("structure_matrix", [](const Vec3& p, const Mat3& R, const Mat83& b, const Mat83& a) {
    const CableGeom g = cable_geometry(p, R, b, a);
    return py::make_tuple(g.W, g.u, g.d);
  });
  m.def("forward_kinematics", [](const Vec8& d, const Mat83& b, const Mat83& a, Vec3 p, Mat3 R) {
    const double rms = forward_kinematics(d, b, a, p, R);
    return py::make_tuple(p, R, rms);
  });
  m.def("tension_distribution", &tension_distribution);
  m.def("tension_bounds", [](const Params& P, const Mat83& u) {
    Vec8 lo, hi;
    tension_bounds(P, u, lo, hi);
    return py::make_tuple(lo, hi);
  });
  m.def("drone_tension_cap", &drone_tension_cap);
  m.def("ugv_tension_cap", &ugv_tension_cap);
  m.def("chord_length", &chord_length);
  m.def("solve_box_qp", [](const Mat8& H, const Vec8& g, const Vec8& lo, const Vec8& hi, Vec8 x) {
    const int it = solve_box_qp(H, g, lo, hi, x);
    return py::make_tuple(x, it);
  });
}
