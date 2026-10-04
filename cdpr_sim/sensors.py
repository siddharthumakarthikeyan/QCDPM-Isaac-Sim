"""Cameras and lidars, all parented to the robot bodies so they ride the PhysX poses at render time.

platform_cam (all 8 robots): ideal 2-axis pan/tilt gimbal that keeps the platform centred. The gimbal angle is
    set before each rendered frame (one-frame actuation lag, like a real gimbal) and published on /tf as
    <robot>/base_link -> <robot>/platform_cam_optical, so tag detections can be chained back to the world.
vo_cam (drones): fixed, forward-looking, pitched down.            -> /drone_k/vo_cam/{image_raw,camera_info}
lidar (UGVs): RTX 2D lidar (SICK TiM781).                           -> /ugv_k/scan (LaserScan), /ugv_k/points
Camera images use ROS optical frames (z forward, x right, y down).
"""

import carb
import numpy as np
import omni.graph.core as og
import usdrt.Sdf
from pxr import Gf, UsdGeom

from .mathutil import look_at_quat, rot_to_quat

Q_USD_TO_OPTICAL = np.array([0.0, 1.0, 0.0, 0.0])  # 180 deg about x: USD camera (-z fwd, +y up) -> optical


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _camera(stage, path, pos, quat, res, hfov_deg):
    cam = UsdGeom.Camera.Define(stage, path)
    ha = 20.955
    cam.CreateHorizontalApertureAttr(ha)
    cam.CreateVerticalApertureAttr(ha * res[1] / res[0])
    cam.CreateFocalLengthAttr(ha / 2.0 / np.tan(np.deg2rad(hfov_deg) / 2.0))
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200.0))
    cam.CreateProjectionAttr("perspective")
    cam.ClearXformOpOrder()
    cam.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    op = cam.AddOrientOp()
    op.Set(Gf.Quatf(*map(float, quat)))
    return cam, op


def _ros_camera_graph(graph_path, cam_path, res, namespace, topic, frame_id, skip):
    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("tick", "omni.graph.action.OnPlaybackTick"),
                ("rp", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("rgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("info", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ],
            keys.CONNECT: [
                ("tick.outputs:tick", "rp.inputs:execIn"),
                ("rp.outputs:execOut", "rgb.inputs:execIn"),
                ("rp.outputs:execOut", "info.inputs:execIn"),
                ("rp.outputs:renderProductPath", "rgb.inputs:renderProductPath"),
                ("rp.outputs:renderProductPath", "info.inputs:renderProductPath"),
            ],
            keys.SET_VALUES: [
                ("rp.inputs:cameraPrim", [usdrt.Sdf.Path(cam_path)]),
                ("rp.inputs:width", int(res[0])),
                ("rp.inputs:height", int(res[1])),
                ("rgb.inputs:nodeNamespace", namespace),
                ("rgb.inputs:topicName", topic + "/image_raw"),
                ("rgb.inputs:frameId", frame_id),
                ("rgb.inputs:type", "rgb"),
                ("rgb.inputs:frameSkipCount", int(skip)),
                ("info.inputs:nodeNamespace", namespace),
                ("info.inputs:topicName", topic + "/camera_info"),
                ("info.inputs:frameId", frame_id),
                ("info.inputs:frameSkipCount", int(skip)),
            ],
        },
    )


class SensorSuite:
    def __init__(self, stage, cfg, ros=True):
        self.cfg = cfg
        sc = cfg["sensors"]
        render_hz = 1.0 / cfg["sim"]["render_dt"]
        self.robot_paths = [f"/World/Drone_{k}" for k in range(4)] + [f"/World/UGV_{k}/chassis" for k in range(4)]
        self.robot_names = [f"drone_{k}" for k in range(4)] + [f"ugv_{k}" for k in range(4)]
        pc = sc["platform_camera"]
        self.gimbal_offsets = [np.array(pc["mount_offset_drone"], float)] * 4 + [np.array(pc["mount_offset_ugv"], float)] * 4
        self.gimbal_q = [np.array([1.0, 0, 0, 0])] * 8
        self.gimbal_ops = []
        self.platform_target = np.array(cfg["platform"]["start_position"], float)
        skip_pc = max(0, int(round(render_hz / pc["rate_hz"])) - 1)
        for i, (bp, name) in enumerate(zip(self.robot_paths, self.robot_names)):
            cam_path = f"{bp}/platform_cam"
            _, op = _camera(stage, cam_path, self.gimbal_offsets[i], (1, 0, 0, 0), pc["resolution"], pc["hfov_deg"])
            self.gimbal_ops.append(op)
            if ros:
                _ros_camera_graph(f"/Graphs/{name}_platform_cam", cam_path, pc["resolution"], name, "platform_cam",
                                  f"{name}/platform_cam_optical", skip_pc)

        vo = sc["vo_camera"]
        pitch = np.deg2rad(vo["pitch_down_deg"])
        q_vo = look_at_quat(np.zeros(3), np.array([np.cos(pitch), 0.0, -np.sin(pitch)]))
        self.vo_offset, self.vo_q = np.array(vo["mount_offset"], float), q_vo
        skip_vo = max(0, int(round(render_hz / vo["rate_hz"])) - 1)
        for k in range(4):
            cam_path = f"/World/Drone_{k}/vo_cam"
            _camera(stage, cam_path, self.vo_offset, q_vo, vo["resolution"], vo["hfov_deg"])
            if ros:
                _ros_camera_graph(f"/Graphs/drone_{k}_vo_cam", cam_path, vo["resolution"], f"drone_{k}", "vo_cam",
                                  f"drone_{k}/vo_cam_optical", skip_vo)

        self.lidar_offset = np.array(sc["lidar_2d"]["mount_offset"], float)
        self.lidars = []
        if ros:
            self._create_lidars(sc["lidar_2d"])

    # ------------------------------------------------------------------ lidar
    def _create_lidars(self, lc):
        from isaacsim.sensors.experimental.rtx import Lidar, LidarSensor

        for k in range(4):
            path = f"/World/UGV_{k}/chassis/lidar"
            config = lc["config"]
            try:
                lidar = Lidar.create(path, config=config, tick_rate=float(lc["rate_hz"]), translations=[self.lidar_offset])
            except Exception as e:  # asset server unreachable -> bundled example lidar
                carb.log_warn(f"lidar config {config} unavailable ({e}); using {lc['fallback_config']}")
                config = lc["fallback_config"]
                lidar = Lidar.create(path, config=config, tick_rate=None, translations=[self.lidar_offset])
            prim = lidar.prims[0]
            meta = _laser_scan_meta(prim)
            sensor = LidarSensor(lidar, annotators=[])
            sensor.attach_writer("RtxLidarROS2PublishPointCloud", topicName=f"ugv_{k}/points", frameId=f"ugv_{k}/lidar")
            try:
                sensor.attach_writer("RtxLidarROS2PublishLaserScan", topicName=f"ugv_{k}/scan", frameId=f"ugv_{k}/lidar", **meta)
            except Exception as e:
                carb.log_warn(f"LaserScan writer unavailable for {config}: {e}; PointCloud2 only")
            self.lidars.append((lidar, sensor))

    # ------------------------------------------------------------------ per frame
    def update_gimbals(self, state, target=None):
        """Point every platform camera at `target` (default: true platform centre)."""
        target = state["pos"][0] if target is None else target
        for i in range(8):
            j = i + 1
            R = state["R"][j]
            eye_w = state["pos"][j] + R @ self.gimbal_offsets[i]
            d_body = R.T @ (target - eye_w)
            q = look_at_quat(np.zeros(3), d_body, up=(0.0, 0.0, 1.0))
            self.gimbal_q[i] = q
            self.gimbal_ops[i].Set(Gf.Quatf(*map(float, q)))

    def dynamic_frames(self):
        for i, name in enumerate(self.robot_names):
            yield f"{name}/base_link", f"{name}/platform_cam_optical", self.gimbal_offsets[i], _qmul(self.gimbal_q[i], Q_USD_TO_OPTICAL)

    def static_frames(self):
        tag = self.cfg["platform"]["apriltag"]
        h = self.cfg["platform"]["side"] / 2.0
        # Tag frames: x = printed image right, y = image up, z = out of the tag face (towards the viewer).
        # Top tag reads upright from above with +y_platform up; bottom tag reads upright from below with +y up.
        yield "platform", f"tag36h11_{tag['top_id']}", (0, 0, h + 0.001), (1, 0, 0, 0)
        off = tag["bottom_offset"]
        for tid, (sx, sy) in zip(tag["bottom_ids"], ((1, 1), (-1, 1), (-1, -1), (1, -1))):
            yield "platform", f"tag36h11_{tid}", (sx * off, sy * off, -h - 0.001), rot_to_quat(np.diag([-1.0, 1.0, -1.0]))
        for k in range(4):
            yield f"drone_{k}/base_link", f"drone_{k}/vo_cam_optical", self.vo_offset, _qmul(self.vo_q, Q_USD_TO_OPTICAL)
            yield f"drone_{k}/base_link", f"drone_{k}/imu", (0, 0, 0), (1, 0, 0, 0)
            yield f"ugv_{k}/base_link", f"ugv_{k}/lidar", self.lidar_offset, (1, 0, 0, 0)


def _laser_scan_meta(prim):
    g = lambda n, d=0.0: prim.GetAttribute(n).Get() if prim.GetAttribute(n) else d  # noqa: E731
    rate = float(g("omni:sensor:Core:scanRateBaseHz") or 10.0)
    near, far = float(g("omni:sensor:Core:nearRangeM")), float(g("omni:sensor:Core:farRangeM"))
    if str(g("omni:sensor:Core:scanType", "")).upper() == "SOLID_STATE":
        rays = g("omni:sensor:Core:numRaysPerLine") or [811]
        fov = 270.0  # SICK TiM781 field of view
        return {"horizontalFov": fov, "horizontalResolution": fov / (int(rays[0]) - 1), "depthRange": [near, far],
                "rotationRate": rate, "azimuthRange": [-fov / 2, fov / 2]}
    firing = float(g("omni:sensor:Core:patternFiringRateHz") or 1.0)
    return {"horizontalFov": 360.0, "horizontalResolution": 360.0 * rate / firing, "depthRange": [near, far],
            "rotationRate": rate, "azimuthRange": [-180.0, 180.0]}
