"""rclpy interface running inside the Isaac process (Isaac's bundled ROS 2 Humble libs).

Published (stamped with simulation time):
  /clock                               rosgraph_msgs/Clock          every physics-rate tick of `clock_div`
  /tf                                  world -> platform, drone_k/base_link, ugv_k/base_link (ground truth),
                                       <robot>/base_link -> <robot>/platform_cam_optical (gimbal, per frame)
  /tf_static                           platform -> tag frames, drone_k/base_link -> vo_cam_optical / imu,
                                       ugv_k/base_link -> lidar
  /cdpr/platform/pose_gt               geometry_msgs/PoseStamped
  /cdpr/platform/odom_gt               nav_msgs/Odometry
  /cdpr/cables                         sensor_msgs/JointState  name=cable_k, position=unstretched length L [m],
                                       velocity=L_dot [m/s], effort=measured tension [N]
  /cdpr/cables/tension_des             std_msgs/Float64MultiArray (tension distribution output) [N]
  /drone_k/odom_gt, /ugv_k/odom_gt     nav_msgs/Odometry
  /drone_k/imu                         sensor_msgs/Imu  (specific force + rate, white noise + bias random walk)
  /drone_k/rotor_speeds                std_msgs/Float64MultiArray [rad/s]
Subscribed:
  /cdpr/platform/cmd_pose              geometry_msgs/PoseStamped   platform target (rate-limited internally)
  /cdpr/formation/enable               std_msgs/Bool               robots follow the formation (default true)
  /drone_k/cmd_pose                    geometry_msgs/PoseStamped   drone position/yaw override
  /ugv_k/cmd_vel                       geometry_msgs/Twist         UGV velocity override (0.5 s watchdog)
  /ugv_k/cmd_goal                      geometry_msgs/Pose2D        UGV go-to-pose override
  /cdpr/winch/cmd                      sensor_msgs/JointState      manual winch: position=L (length mode, with
                                       effort as tension feed-forward) or position=NaN & effort=T (tension mode)
  /cdpr/winch/auto                     std_msgs/Bool               re-enable IK + tension distribution
"""

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Pose2D, PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.qos import DurabilityPolicy, QoSProfile
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool, Float64MultiArray
from tf2_msgs.msg import TFMessage

from .mathutil import quat_to_rot, rot_to_quat

ROBOTS = [f"drone_{k}" for k in range(4)] + [f"ugv_{k}" for k in range(4)]


def _stamp(t):
    s = int(t)
    return Time(sec=s, nanosec=int(round((t - s) * 1e9)))


def _tf(parent, child, p, q, t):
    m = TransformStamped()
    m.header.stamp = _stamp(t)
    m.header.frame_id = parent
    m.child_frame_id = child
    m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = map(float, p)
    m.transform.rotation.w, m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z = map(float, q)
    return m


class ImuModel:
    """Discrete IMU: averages specific force / rate over the sample period, adds white noise + bias random walk.
    Defaults are an ICM-42688-class MEMS unit."""

    def __init__(self, n, rate, physics_dt, rng):
        self.n, self.rng = n, rng
        self.period = 1.0 / rate
        self.gyro_nd, self.acc_nd = 2.8e-3 * np.pi / 180, 70e-6 * 9.81   # density [unit/sqrt(Hz)]
        self.gyro_rw, self.acc_rw = 1e-5, 1e-4                             # bias random walk [unit/s/sqrt(Hz)]
        self.bg = rng.normal(0, 0.003, (n, 3))
        self.ba = rng.normal(0, 0.02, (n, 3))
        self.acc_sum = np.zeros((n, 3))
        self.gyr_sum = np.zeros((n, 3))
        self.count = 0
        self.t_acc = 0.0

    def accumulate(self, R, a_world, w_world, g, dt):
        f = np.einsum("nji,nj->ni", R, a_world + np.array([0, 0, g]))
        w = np.einsum("nji,nj->ni", R, w_world)
        self.acc_sum += f
        self.gyr_sum += w
        self.count += 1
        self.t_acc += dt
        if self.t_acc + 1e-9 < self.period:
            return None
        rate = 1.0 / self.t_acc
        self.bg += self.rng.normal(0, self.gyro_rw * np.sqrt(self.t_acc), self.bg.shape)
        self.ba += self.rng.normal(0, self.acc_rw * np.sqrt(self.t_acc), self.ba.shape)
        acc = self.acc_sum / self.count + self.ba + self.rng.normal(0, self.acc_nd * np.sqrt(rate / 2), self.ba.shape)
        gyr = self.gyr_sum / self.count + self.bg + self.rng.normal(0, self.gyro_nd * np.sqrt(rate / 2), self.bg.shape)
        self.acc_sum[:] = 0
        self.gyr_sum[:] = 0
        self.count = 0
        self.t_acc = 0.0
        return acc, gyr


class RosInterface:
    def __init__(self, rt, cfg, sensor_frames):
        self.rt, self.cfg = rt, cfg
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("cdpr_sim")
        n = self.node
        self.pub_clock = n.create_publisher(Clock, "/clock", 10)
        self.pub_tf = n.create_publisher(TFMessage, "/tf", 100)
        self.pub_tf_static = n.create_publisher(
            TFMessage, "/tf_static", QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.pub_pose = n.create_publisher(PoseStamped, "/cdpr/platform/pose_gt", 10)
        self.pub_podom = n.create_publisher(Odometry, "/cdpr/platform/odom_gt", 10)
        self.pub_cables = n.create_publisher(JointState, "/cdpr/cables", 10)
        self.pub_tdes = n.create_publisher(Float64MultiArray, "/cdpr/cables/tension_des", 10)
        self.pub_odom = [n.create_publisher(Odometry, f"/{r}/odom_gt", 10) for r in ROBOTS]
        self.pub_imu = [n.create_publisher(Imu, f"/drone_{k}/imu", 50) for k in range(4)]
        self.pub_rotor = [n.create_publisher(Float64MultiArray, f"/drone_{k}/rotor_speeds", 10) for k in range(4)]

        n.create_subscription(PoseStamped, "/cdpr/platform/cmd_pose", self._on_platform_cmd, 10)
        n.create_subscription(Bool, "/cdpr/formation/enable", self._on_formation, 10)
        n.create_subscription(JointState, "/cdpr/winch/cmd", self._on_winch, 10)
        n.create_subscription(Bool, "/cdpr/winch/auto", self._on_winch_auto, 10)
        for k in range(4):
            n.create_subscription(PoseStamped, f"/drone_{k}/cmd_pose", lambda m, k=k: self._on_drone_cmd(k, m), 10)
            n.create_subscription(Twist, f"/ugv_{k}/cmd_vel", lambda m, k=k: self._on_ugv_vel(k, m), 10)
            n.create_subscription(Pose2D, f"/ugv_{k}/cmd_goal", lambda m, k=k: self._on_ugv_goal(k, m), 10)
        self.ugv_vel_stamp = np.full(4, -1.0)

        imu_rate = cfg["sensors"]["imu"]["rate_hz"]
        self.imu = ImuModel(4, imu_rate, rt.dt, np.random.default_rng(11))
        # per-sample variance = density^2 * rate / 2
        self.acc_cov = (np.eye(3) * self.imu.acc_nd**2 * imu_rate / 2).ravel().tolist()
        self.gyr_cov = (np.eye(3) * self.imu.gyro_nd**2 * imu_rate / 2).ravel().tolist()
        self.state_div = max(1, int(round(1.0 / (100.0 * rt.dt))))  # odometry / cables / tf at 100 Hz
        self.clock_div = max(1, int(round(1.0 / (250.0 * rt.dt))))  # /clock at 250 Hz
        self.sensor_frames = sensor_frames
        self._publish_static()

    # ---------------------------------------------------------------- static frames
    def _publish_static(self):
        msgs = []
        for parent, child, p, q in self.sensor_frames.static_frames():
            msgs.append(_tf(parent, child, p, q, 0.0))
        self.pub_tf_static.publish(TFMessage(transforms=msgs))

    # ---------------------------------------------------------------- callbacks (commands)
    def _on_platform_cmd(self, m):
        p = [m.pose.position.x, m.pose.position.y, m.pose.position.z]
        o = m.pose.orientation
        self.rt.ref_target_p = np.array(p)
        self.rt.ref_target_R = quat_to_rot(np.array([o.w, o.x, o.y, o.z]))

    def _on_formation(self, m):
        self.rt.formation_follow = bool(m.data)
        if m.data:
            self.rt.drone_override[:] = False
            self.rt.ugv_cmd.mode[:] = "pose"

    def _on_drone_cmd(self, k, m):
        self.rt.drone_override[k] = True
        self.rt.quads.p_ref[k] = [m.pose.position.x, m.pose.position.y, m.pose.position.z]
        self.rt.quads.v_ref[k] = 0.0
        o = m.pose.orientation
        self.rt.quads.yaw_ref[k] = np.arctan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y * o.y + o.z * o.z))

    def _on_ugv_vel(self, k, m):
        self.rt.ugv_cmd.mode[k] = "velocity"
        self.rt.ugv_cmd.cmd_v[k] = m.linear.x
        self.rt.ugv_cmd.cmd_om[k] = m.angular.z
        self.ugv_vel_stamp[k] = self.rt.sim_time

    def _on_ugv_goal(self, k, m):
        self.rt.ugv_cmd.mode[k] = "goal"   # pose controller, but not overwritten by the formation
        self.rt.ugv_cmd.goal[k] = [m.x, m.y, m.theta]

    def _on_winch(self, m):
        self.rt.winch_auto = False
        c = self.rt.cables
        for i, name in enumerate(m.name):
            k = int(name.split("_")[-1])
            pos = m.position[i] if i < len(m.position) else float("nan")
            eff = m.effort[i] if i < len(m.effort) else 0.0
            if np.isnan(pos):
                c.mode[k] = "tension"
                c.T_ff[k] = eff
            else:
                c.mode[k] = "length"
                c.L_cmd[k] = pos
                c.Ld_cmd[k] = 0.0
                c.T_ff[k] = eff

    def _on_winch_auto(self, m):
        self.rt.winch_auto = bool(m.data)

    # ---------------------------------------------------------------- per physics step
    def on_physics_step(self):
        rt = self.rt
        t = rt.sim_time
        s = rt.state
        k = rt.step_count
        # velocity-command watchdog
        stale = (rt.ugv_cmd.mode == "velocity") & (self.ugv_vel_stamp >= 0) & (t - self.ugv_vel_stamp > 0.5)
        rt.ugv_cmd.cmd_v[stale] = 0.0
        rt.ugv_cmd.cmd_om[stale] = 0.0

        out = self.imu.accumulate(s["R"][1:5], rt.last["accel"][1:5], s["w"][1:5], rt.g, rt.dt)
        if out is not None:
            acc, gyr = out
            for i in range(4):
                m = Imu()
                m.header.stamp = _stamp(t)
                m.header.frame_id = f"drone_{i}/imu"
                m.orientation_covariance[0] = -1.0
                m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = map(float, acc[i])
                m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = map(float, gyr[i])
                m.linear_acceleration_covariance = self.acc_cov
                m.angular_velocity_covariance = self.gyr_cov
                self.pub_imu[i].publish(m)

        if k % self.clock_div == 0:
            self.pub_clock.publish(Clock(clock=_stamp(t)))
        if k % self.state_div == 0:
            self._publish_state(t)

    def _publish_state(self, t):
        rt, s = self.rt, self.rt.state
        frames = ["platform"] + [f"{r}/base_link" for r in ROBOTS]
        tfs = [_tf("world", f, s["pos"][i], s["quat"][i], t) for i, f in enumerate(frames)]
        self.pub_tf.publish(TFMessage(transforms=tfs))
        ps = PoseStamped()
        ps.header.stamp = _stamp(t)
        ps.header.frame_id = "world"
        _fill_pose(ps.pose, s["pos"][0], s["quat"][0])
        self.pub_pose.publish(ps)
        self.pub_podom.publish(_odom(t, "platform", s["pos"][0], s["quat"][0], s["v"][0], s["w"][0], s["R"][0]))
        for i, r in enumerate(ROBOTS):
            j = i + 1
            self.pub_odom[i].publish(_odom(t, f"{r}/base_link", s["pos"][j], s["quat"][j], s["v"][j], s["w"][j], s["R"][j]))
        c = rt.cables
        js = JointState()
        js.header.stamp = _stamp(t)
        js.name = [f"cable_{i}" for i in range(8)]
        js.position = c.L.tolist()
        js.velocity = c.L_dot.tolist()
        js.effort = rt.last["T"].tolist()
        self.pub_cables.publish(js)
        self.pub_tdes.publish(Float64MultiArray(data=rt.ctrl.t_des.tolist()))
        for i in range(4):
            self.pub_rotor[i].publish(Float64MultiArray(data=rt.quads.w[i].tolist()))

    # ---------------------------------------------------------------- per render frame
    def on_render_frame(self, t):
        tfs = [_tf(parent, child, p, q, t) for parent, child, p, q in self.sensor_frames.dynamic_frames()]
        if tfs:
            self.pub_tf.publish(TFMessage(transforms=tfs))
        for _ in range(20):  # drain queued commands without blocking the sim
            rclpy.spin_once(self.node, timeout_sec=0.0)

    def shutdown(self):
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _fill_pose(pose, p, q):
    pose.position.x, pose.position.y, pose.position.z = map(float, p)
    pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = map(float, q)


def _odom(t, child, p, q, v, w, R):
    m = Odometry()
    m.header.stamp = _stamp(t)
    m.header.frame_id = "world"
    m.child_frame_id = child
    _fill_pose(m.pose.pose, p, q)
    vb, wb = R.T @ v, R.T @ w  # twist in the child frame (REP-105)
    m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z = map(float, vb)
    m.twist.twist.angular.x, m.twist.twist.angular.y, m.twist.twist.angular.z = map(float, wb)
    return m
