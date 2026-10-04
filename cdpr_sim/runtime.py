"""Per-physics-step coupling between PhysX and the cable / winch / rotor / UGV models.

Bodies are read and written through one NumPy physics-tensor view (CPU PhysX, no device copies):
    index 0     platform
    index 1..4  drones 0..3
    index 5..8  UGV chassis 0..3 (articulation root links)
Cable k (0..3) runs platform top corner k -> drone k; cable 4+k runs bottom corner k -> UGV k.
"""

import numpy as np
import omni.physics.tensors as tensors
import omni.usd
from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager

from .cdpr_control import PlatformController, corner_points, formation, hold_table, lazy_center
from .mathutil import cross, quat_to_rot, rpy_to_rot
from .models import CableWinchModel, DiffDriveCommand, QuadrotorModel

E3 = np.array([0.0, 0.0, 1.0])


def _xyzw_to_wxyz(q):
    return q[:, [3, 0, 1, 2]]


class CdprRuntime:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dt = cfg["sim"]["physics_dt"]
        self.g = cfg["sim"]["gravity"]
        self.b = corner_points(cfg["platform"]["side"])
        self.r_drone = np.array(cfg["drone"]["attach_offset"], float)
        chassis_z = cfg["ugv"]["ground_clearance"] + cfg["ugv"]["chassis_size"][2] / 2
        self.r_ugv = np.array([0.0, 0.0, cfg["ugv"]["attach_height"] - chassis_z])  # mast-top fairlead
        self.r_local = np.vstack([np.tile(self.r_drone, (4, 1)), np.tile(self.r_ugv, (4, 1))])
        self.cables = CableWinchModel(cfg["cable"], cfg["winch"], 8)
        self.quads = QuadrotorModel(cfg["drone"], 4, self.g)
        self.ugv_cmd = DiffDriveCommand(cfg["ugv"], 4)
        self.ctrl = PlatformController(cfg, self.b, self.g)
        self.p0 = np.array(cfg["platform"]["start_position"], float)
        self.form_center = self.p0[:2].copy()
        self.form_vel = np.zeros(2)
        self.form_acc = cfg["layout"].get("formation_acc", 0.0)   # 0 = the centre follows its target without a limit
        self.hold = hold_table(cfg)
        self.hybrid = cfg["winch"]["drone_cable_mode"] == "tension"
        if self.hybrid:  # pose is no longer enforced kinematically by all 8 cables -> close it with a wrench PD
            wn, z = cfg["winch"]["hybrid_pd"]["wn"], cfg["winch"]["hybrid_pd"]["zeta"]
            self.ctrl.kp_lin = self.ctrl.kp_rot = wn**2
            self.ctrl.kd_lin = self.ctrl.kd_rot = 2 * z * wn
            self.ctrl.ki_lin = self.ctrl.ki_rot = cfg["winch"]["hybrid_pd"]["ki_ratio"] * wn**3
        rates = cfg["sim"]["control_rates"]
        self.div = {k: max(1, int(round(1.0 / (rates[k] * self.dt)))) for k in ("drone", "platform", "ugv")}

        # command / mode state (written by ROS callbacks or scripts)
        self.formation_follow = True        # robots track the formation around the platform reference
        self.drone_override = np.zeros(4, bool)
        self.winch_auto = True              # IK + tension distribution drive the winches
        self.ref_target_p = self.p0.copy()  # requested platform pose (rate-limited into ctrl.p_ref/R_ref)
        self.ref_target_R = np.eye(3)
        self.max_lin_speed = 0.2            # reference generator limits
        self.max_ang_speed = np.deg2rad(20.0)
        # live reconfiguration: crossing angles [rad] move towards their targets at cross_rate [rad/s]
        L = cfg["layout"]
        self.cross_top = self.cross_top_target = np.deg2rad(L["drone_cross_angle_deg"])
        self.cross_bot = self.cross_bot_target = np.deg2rad(L["ugv_cross_angle_deg"])
        self.cross_rate = np.deg2rad(4.5)       # max crossing-angle rate [rad/s]
        self.cross_acc = np.deg2rad(3.0)        # max crossing-angle acceleration [rad/s^2] (smooth start/stop)
        self._cross_w = np.zeros(2)
        self.ext_wrench = np.zeros(6)       # extra world-frame wrench on the platform (disturbance experiments)

        self.core = None                    # C++ estimation + control (core_bridge.CoreBridge) when core.enabled
        self.view = None
        self.art = None
        self.step_count = 0
        self.sim_time = 0.0
        self.state = {}
        self._prev_vel = None
        self._sub = None
        self.step_hooks = []                # callables run after every physics-step update (e.g. ROS publishing)

    # ------------------------------------------------------------------ setup
    def initialize(self):
        """Call after the timeline is playing (physics initialised)."""
        stage_id = omni.usd.get_context().get_stage_id()
        self.sim_view = tensors.create_simulation_view("numpy", stage_id=stage_id)
        self.sim_view.set_subspace_roots("/")
        paths = ["/World/Platform/base"] + [f"/World/Drone_{k}" for k in range(4)] + [f"/World/UGV_{k}/chassis" for k in range(4)]
        self.view = self.sim_view.create_rigid_body_view(paths)
        assert self.view.count == 9, f"rigid body view matched {self.view.count} bodies"
        self.art = self.sim_view.create_articulation_view([f"/World/UGV_{k}" for k in range(4)])
        assert self.art.count == 4, f"articulation view matched {self.art.count} UGVs"
        names = self.art.shared_metatype.dof_names
        self.wheel_dofs = [names.index("joint_wheel_left"), names.index("joint_wheel_right")]
        self.all_idx = np.arange(9, dtype=np.int32)
        self.art_idx = np.arange(4, dtype=np.int32)
        self.tool = ToolInterface(self.sim_view, self.cfg)
        g = self.cfg["gripper"]
        m_hang = g["wrist"]["mass"] + g["tool"]["mass"] + 2 * g["finger"]["mass"]
        self.tool.w_on_base[2] = -m_hang * self.g          # nominal until the first measurement
        self.ctrl.w_ext = self.tool.w_on_base.copy()
        self._read_state()

        # start in static equilibrium: cables pre-stretched to the tension distribution for the start pose,
        # drones spun up to carry their own weight + the vertical cable load
        anchors = self._anchor_points()
        self.ctrl.p_ref, self.ctrl.R_ref = self.p0.copy(), np.eye(3)
        t, d = self.ctrl.solve(anchors)
        self.cables.L[:] = self.cables.unstretched_for(d, t)
        self.cables.L_cmd[:] = self.cables.L
        self.cables.T_ff[:] = t
        u = anchors - (self.p0 + self.b)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        self.quads.init_hover(t[:4] * u[:4, 2])
        self._apply_formation()
        self.quads.p_ref = anchors[:4] - self.r_drone
        if self.cfg.get("core", {}).get("enabled", False):
            from .core_bridge import CoreBridge

            self.core = CoreBridge(self)
            self.core.reset(self, t)
        self._sub = SimulationManager.register_callback(self._on_physics_step, SimulationEvent.PHYSICS_PRE_STEP)

    def shutdown(self):
        if self._sub is not None:
            SimulationManager.deregister_callback(self._sub)
            self._sub = None

    # ------------------------------------------------------------------ state
    def _read_state(self):
        tf = self.view.get_transforms().reshape(9, 7).astype(np.float64)
        vel = self.view.get_velocities().reshape(9, 6).astype(np.float64)
        pos, quat = tf[:, :3], _xyzw_to_wxyz(tf[:, 3:])
        R = quat_to_rot(quat)
        self.state = dict(pos=pos, quat=quat, R=R, v=vel[:, :3], w=vel[:, 3:])
        if self._prev_vel is None:
            self._prev_vel = vel[:, :3].copy()

    def _anchor_points(self):
        s = self.state
        return s["pos"][1:] + np.einsum("nij,nj->ni", s["R"][1:], self.r_local)

    # ------------------------------------------------------------------ references
    def _advance_reference(self, dt):
        c = self.ctrl
        dp = self.ref_target_p - c.p_ref
        n = np.linalg.norm(dp)
        step = self.max_lin_speed * dt
        p_new = self.ref_target_p.copy() if n <= step else c.p_ref + dp / n * step
        c.v_ref = (p_new - c.p_ref) / dt
        c.p_ref = p_new
        # rotation: move along the geodesic with bounded rate
        Rerr = c.R_ref.T @ self.ref_target_R
        ang = np.arccos(np.clip((np.trace(Rerr) - 1) / 2, -1, 1))
        if ang > 1e-9:
            frac = min(1.0, self.max_ang_speed * dt / ang)
            axis = np.array([Rerr[2, 1] - Rerr[1, 2], Rerr[0, 2] - Rerr[2, 0], Rerr[1, 0] - Rerr[0, 1]]) / (2 * np.sin(ang))
            c.R_ref = c.R_ref @ _axis_angle(axis, ang * frac)

    def _apply_formation(self, dt=0.0):
        if dt > 0:
            # trapezoidal crossing-angle profile: rate- and acceleration-limited, decelerates onto the target
            cur = np.array([self.cross_top, self.cross_bot])
            err = np.array([self.cross_top_target, self.cross_bot_target]) - cur
            w_stop = np.sign(err) * np.sqrt(2 * self.cross_acc * np.abs(err))
            w_des = np.clip(w_stop, -self.cross_rate, self.cross_rate)
            self._cross_w += np.clip(w_des - self._cross_w, -self.cross_acc * dt, self.cross_acc * dt)
            cur = cur + self._cross_w * dt
            done = np.abs(err) < 1e-4
            cur[done] = np.array([self.cross_top_target, self.cross_bot_target])[done]
            self._cross_w[done] = 0.0
            self.cross_top, self.cross_bot = cur
        # the robots hold position while the platform is inside the hold region of the formation and move the least
        # distance needed once it leaves (hold radius 0 = the formation is always centred on the platform)
        r_hold = float(np.interp(self.ctrl.p_ref[2], *self.hold))
        target = lazy_center(self.form_center, self.ctrl.p_ref, r_hold)
        d = self.ctrl.p_ref[:2] - self.form_center
        n = np.linalg.norm(d)
        if dt > 0 and self.form_acc > 0:
            # The centre is a point with limited acceleration, so all eight robots start and stop together. Without
            # this the formation would start at platform speed from standstill: the drones follow at once, the
            # diff-drive robots fall ~0.2 m behind, and the distorted formation saturates the tensions.
            if r_hold > 1e-6:
                v_t = d / n * max(0.0, d @ self.ctrl.v_ref[:2] / n) if n > r_hold else np.zeros(2)
            else:
                v_t = self.ctrl.v_ref[:2]
            wn = 6.0
            acc = wn * wn * (target - self.form_center) + 2 * wn * (v_t - self.form_vel)
            a_n = np.linalg.norm(acc)
            self.form_vel = self.form_vel + acc * min(1.0, self.form_acc / max(a_n, 1e-9)) * dt
            self.form_center = self.form_center + self.form_vel * dt
        else:
            self.form_center = target
        d_anch, g_anch = formation(self.cfg, self.form_center, self.cross_top, self.cross_bot)
        keep = self.drone_override
        p_new = d_anch - self.r_drone
        if dt > 0:  # formation velocity feed-forward (platform motion + live reconfiguration)
            self.quads.v_ref = np.where(keep[:, None], 0.0, (p_new - self.quads.p_ref) / dt)
        self.quads.p_ref = np.where(keep[:, None], self.quads.p_ref, p_new)
        pose_mode = self.ugv_cmd.mode == "pose"
        if dt > 0:
            self.ugv_cmd.goal_vel[pose_mode] = (g_anch[pose_mode, :2] - self.ugv_cmd.goal[pose_mode, :2]) / dt
        self.ugv_cmd.goal[pose_mode, :2] = g_anch[pose_mode, :2]
        self.ugv_cmd.goal[pose_mode, 2] = np.nan
        if r_hold > 1e-6 and n > 0.5 * r_hold and "R" in self.state:
            # While holding, the ground robots turn on the spot to face along the platform's offset: that is the
            # direction the formation moves in when the platform leaves the hold region, so they can roll off without
            # turning first (the fairlead is on the turning axis; forwards or backwards, whichever is nearer).
            yaw = np.arctan2(self.state["R"][5:9, 1, 0], self.state["R"][5:9, 0, 0])
            th = np.arctan2(d[1], d[0])
            th = np.where(np.abs(np.angle(np.exp(1j * (th - yaw)))) > np.pi / 2, th + np.pi, th)
            self.ugv_cmd.goal[pose_mode, 2] = th[pose_mode]

    # ------------------------------------------------------------------ main step
    def _on_physics_step(self, dt, context=None):
        dt = self.dt
        self._read_state()
        s = self.state
        self.sim_time += dt
        self.step_count += 1

        k = self.step_count
        run_platform = k % self.div["platform"] == 0
        if run_platform:
            dt_p = dt * self.div["platform"]
            self._advance_reference(dt_p)
            if self.formation_follow:
                self._apply_formation(dt_p)

        # ---- cables
        R_p = s["R"][0]
        rb = self.b @ R_p.T
        pa = s["pos"][0] + rb
        va = s["v"][0] + cross(s["w"][0], rb)
        rr = np.einsum("nij,nj->ni", s["R"][1:], self.r_local)
        pb = s["pos"][1:] + rr
        vb = s["v"][1:] + cross(s["w"][1:], rr)
        T, u, _ = self.cables.tension(pa, va, pb, vb)
        Fc = T[:, None] * u  # force on the platform from each cable
        half_w = 0.5 * self.cables.rho * self.cables.L * self.g  # lumped cable weight at each end

        # ---- end-effector: gripper/wrist commands, F/T, wrench of the hanging tool + payload on the platform
        if run_platform:
            self.tool.update(s["R"][0], dt_p)
            self.ctrl.w_ext = self.tool.w_on_base if self.tool.feedforward else np.zeros(6)

        # ---- C++ core: sensors -> estimators -> platform, drone and ground-robot controllers
        if self.core is not None:
            self.core.step(self, T, (s["v"] - self._prev_vel) / dt, run_platform, dt * self.div["platform"])

        # ---- platform-level controller -> winch commands
        elif self.winch_auto and run_platform:
            t_des, d_ref = self.ctrl.solve(pb, s["pos"][0], R_p, s["v"][0], s["w"][0], dt_p)
            self.cables.mode[:] = "length"
            if self.hybrid:
                self.cables.mode[:4] = "tension"
            L_new = self.cables.unstretched_for(d_ref, t_des)
            # velocity feed-forward from the IK lengths, low-passed (20 ms) since anchors are measured
            a = min(1.0, dt_p / 0.02)
            self.cables.Ld_cmd += a * ((L_new - self.cables.L_cmd) / dt_p - self.cables.Ld_cmd)
            self.cables.L_cmd[:] = L_new
            self.cables.T_ff[:] = t_des
        self.cables.step(dt)

        # ---- drones (controller at its own rate, rotor dynamics every step)
        F_on_drone = -Fc[:4] - half_w[:4, None] * E3
        if self.core is None and k % self.div["drone"] == 0:
            self.quads.control(s["pos"][1:5], s["quat"][1:5], s["v"][1:5], s["w"][1:5], F_on_drone,
                               dt * self.div["drone"], R=s["R"][1:5])
        F_rot, tau_rot = self.quads.actuate(s["quat"][1:5], s["v"][1:5], dt, R=s["R"][1:5])

        # ---- UGVs (wheel servo itself is the PhysX joint drive)
        if self.core is None and k % self.div["ugv"] == 0:
            wheel_targets = self.ugv_cmd.step(s["pos"][5:9], s["quat"][5:9], dt * self.div["ugv"], R=s["R"][5:9])
            vt = np.zeros((4, self.art.max_dofs), np.float32)
            vt[:, self.wheel_dofs] = wheel_targets
            self.art.set_dof_velocity_targets(vt, self.art_idx)

        # ---- assemble wrenches (world frame, about each body origin = CoM)
        F = np.zeros((9, 3))
        tau = np.zeros((9, 3))
        F[0] = Fc.sum(0) - half_w.sum() * E3 + self.ext_wrench[:3]
        tau[0] = cross(rb, Fc).sum(0) + self.ext_wrench[3:]
        F[1:5] = F_rot + F_on_drone
        tau[1:5] = tau_rot + cross(rr[:4], F_on_drone)
        F_on_ugv = -Fc[4:] - half_w[4:, None] * E3
        F[5:9] = F_on_ugv
        tau[5:9] = cross(rr[4:], F_on_ugv)
        self.view.apply_forces_and_torques_at_position(F.astype(np.float32), tau.astype(np.float32), None, self.all_idx, True)

        self.last = dict(T=T.copy(), u=u, pa=pa, pb=pb, F_on_drone=F_on_drone, t_des=self.ctrl.t_des.copy(),
                         accel=(s["v"] - self._prev_vel) / dt)
        self._prev_vel = s["v"].copy()
        for hook in self.step_hooks:
            hook()

    # ------------------------------------------------------------------ convenience API
    def set_platform_target(self, p, rpy=None, R=None):
        self.ref_target_p = np.asarray(p, float)
        if R is not None:
            self.ref_target_R = np.asarray(R, float)
        elif rpy is not None:
            self.ref_target_R = rpy_to_rot(rpy)


class ToolInterface:
    """Wrist yaw + parallel gripper on the platform articulation, and the wrist load cell.

    ft_z      : axial load measured by the wrist load cell [N] (positive = tension, i.e. carrying a load)
                = -k q - c q_dot of the stiff sensor spring, i.e. tool + fingers + payload weight + contact forces
                along the tool axis, with a 30 Hz sensor filter.
    w_on_base : world wrench [F; tau about the platform CoM] the hanging assembly exerts on the platform
                (wrist housing weight + measured load along the tool axis, low-passed 20 ms). The tension
                distribution subtracts it, so a picked-up block is compensated as soon as it is measured.
    """

    def __init__(self, sim_view, cfg):
        self.view = sim_view.create_articulation_view(["/World/Platform"])
        assert self.view.count == 1, "platform articulation not found"
        meta = self.view.shared_metatype
        self.dof = {n: meta.dof_names.index(n) for n in ("joint_wrist_yaw", "joint_finger_l", "joint_finger_r", "joint_wrist_ft")}
        g = cfg["gripper"]
        self.half_stroke = g["finger"]["stroke"] / 2
        self.finger_k = g["finger"]["stiffness"]
        self.grip_force = g["grip_force"]
        self.k_ft, self.c_ft = g["ft_sensor"]["stiffness"], g["ft_sensor"]["damping"]
        self.wrist_weight = g["wrist"]["mass"] * cfg["sim"]["gravity"]
        self.yaw_cmd = 0.0
        self.width_cmd = 2 * self.half_stroke                 # inner-face opening command [m]
        self.squeeze = False
        self.feedforward = True
        self.ff_tau = cfg["gripper"]["ft_sensor"].get("feedforward_tau", 0.08)   # load feed-forward low-pass [s]
        self.w_on_base = np.zeros(6)
        self.ft_z = 0.0
        self.ft_raw = 0.0
        self.q = np.zeros(self.view.max_dofs)
        self.idx = np.zeros(1, dtype=np.int32)

    # commands
    def open(self, width=None):
        self.squeeze = False
        self.width_cmd = 2 * self.half_stroke if width is None else float(width)

    def close(self):
        """Squeeze with grip_force: target past contact so the stiffness x error saturates at the force limit."""
        self.squeeze = True

    def set_yaw(self, yaw):
        self.yaw_cmd = float(np.angle(np.exp(1j * yaw)))

    @property
    def width(self):
        return float(self.q[self.dof["joint_finger_l"]] - self.q[self.dof["joint_finger_r"]])

    @property
    def yaw(self):
        return float(self.q[self.dof["joint_wrist_yaw"]])

    def update(self, R_base, dt):
        v = self.view
        self.q = v.get_dof_positions().reshape(-1)
        tgt = np.zeros((1, v.max_dofs), np.float32)
        tgt[0, self.dof["joint_wrist_yaw"]] = self.yaw_cmd
        if self.squeeze:
            # coupled jaws, as on a real parallel gripper: both targets are symmetric about the tool axis, `push`
            # (= force / stiffness) inside the current opening. Each jaw then squeezes with grip_force when the part
            # is centred, and an off-centre part is pushed back by 2 k x offset. (Independent jaws let the block
            # slide in the grip until one jaw reached its end stop: a 14.5 mm placement offset.)
            push = self.grip_force / self.finger_k
            half_open = 0.5 * (self.q[self.dof["joint_finger_l"]] - self.q[self.dof["joint_finger_r"]])
            tgt[0, self.dof["joint_finger_l"]] = max(0.0, half_open - push)
            tgt[0, self.dof["joint_finger_r"]] = min(0.0, -(half_open - push))
        else:
            tgt[0, self.dof["joint_finger_l"]] = self.width_cmd / 2
            tgt[0, self.dof["joint_finger_r"]] = -self.width_cmd / 2
        v.set_dof_position_targets(tgt, self.idx)

        j = self.dof["joint_wrist_ft"]
        self.ft_raw = float(-self.k_ft * self.q[j])          # strain-gauge style: deflection x stiffness
        self.ft_z += min(1.0, dt * 2 * np.pi * 30.0) * (self.ft_raw - self.ft_z)
        f_down = (self.wrist_weight + self.ft_raw) * R_base[:, 2]       # pulled along -z_base by the hanging load
        w_base = np.r_[-f_down, 0.0, 0.0, 0.0]                          # load line passes through the base CoM
        self.w_on_base += min(1.0, dt / self.ff_tau) * (w_base - self.w_on_base)


def _axis_angle(axis, ang):
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
