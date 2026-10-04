"""Mobile CDPR simulation in Isaac Sim 6.0 (PhysX).

    ./launch.sh                      # GUI, sensors + ROS 2
    ./launch.sh --headless --demo    # scripted manoeuvre, headless
    ./launch.sh --no-sensors --no-ros --demo --duration 30

Options:
  --config PATH   parameter file (default config/cdpr.yaml)
  --headless      no viewport window
  --no-sensors    skip cameras / lidars (physics + ROS state only)
  --no-ros        skip ROS 2 entirely
  --demo          run the scripted manoeuvre (lift + tilt, then translate the formation)
  --duration S    stop after S seconds of simulated time
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=os.path.join(ROOT, "config", "cdpr.yaml"))
parser.add_argument("--headless", action="store_true")
parser.add_argument("--no-sensors", action="store_true")
parser.add_argument("--no-ros", action="store_true")
parser.add_argument("--demo", action="store_true")
parser.add_argument("--duration", type=float, default=float("inf"))
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": args.headless, "width": 1600, "height": 900})

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import yaml  # noqa: E402
from isaacsim.core.experimental.utils.app import enable_extension  # noqa: E402
from isaacsim.core.rendering_manager import RenderingManager, ViewportManager  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from pxr import Gf, UsdGeom, Vt  # noqa: E402

from cdpr_sim.runtime import CdprRuntime  # noqa: E402
from cdpr_sim.scene import build_scene  # noqa: E402

cfg = yaml.safe_load(open(args.config))
use_ros = cfg["ros"]["enabled"] and not args.no_ros
use_sensors = cfg["sensors"]["enabled"] and not args.no_sensors
if use_ros:
    enable_extension("isaacsim.ros2.bridge")
    app.update()

omni.usd.get_context().new_stage()
app.update()
stage = omni.usd.get_context().get_stage()
build_scene(stage, cfg)

sensors = None
if use_sensors:
    from cdpr_sim.sensors import SensorSuite

    sensors = SensorSuite(stage, cfg, ros=use_ros)
app.update()

SimulationManager.set_physics_sim_device(cfg["sim"]["device"])
SimulationManager.set_physics_dt(cfg["sim"]["physics_dt"])
RenderingManager.set_dt(cfg["sim"]["render_dt"])
if not args.headless:
    ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[6.5, -6.5, 4.5], target=[0.0, 0.0, 1.5])
app.update()

omni.timeline.get_timeline_interface().play()
app.update()
rt = CdprRuntime(cfg)
rt.initialize()

ros = None
if use_ros:
    from cdpr_sim.ros_iface import RosInterface

    class _NoSensors:
        def static_frames(self):
            return []

        def dynamic_frames(self):
            return []

    ros = RosInterface(rt, cfg, sensors if sensors else _NoSensors())
    rt.step_hooks.append(ros.on_physics_step)

cables = UsdGeom.BasisCurves(stage.GetPrimAtPath("/World/Cables"))


def update_cable_visuals():
    last = getattr(rt, "last", None)
    if last is None:
        return
    pts = np.empty((16, 3))
    pts[0::2] = last["pa"]
    pts[1::2] = last["pb"]
    cables.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))


CROSSED = abs(cfg["layout"]["drone_cross_angle_deg"]) > 1e-6
DEMO = [  # (time, platform position offset from start, rpy deg). Straight cables can only hold the platform level.
    (3.0, [0.0, 0.0, 0.3], [10, 0, 15] if CROSSED else [0, 0, 0]),
    (11.0, [1.0, 0.0, 0.3], [0, 0, 0]),
    (19.0, [1.0, 1.0, 0.0], [0, -8, -10] if CROSSED else [0, 0, 0]),
    (27.0, [0.0, 0.0, 0.0], [0, 0, 0]),
]
demo_i = 0
wall0, frames = time.time(), 0
print(f"[cdpr] running: sensors={use_sensors} ros={use_ros} physics_dt={rt.dt} render_dt={cfg['sim']['render_dt']}")
while app.is_running() and rt.sim_time < args.duration:
    if args.demo and demo_i < len(DEMO) and rt.sim_time >= DEMO[demo_i][0]:
        _, dp, rpy = DEMO[demo_i]
        rt.set_platform_target(rt.p0 + np.array(dp), rpy=np.deg2rad(rpy))
        print(f"[cdpr] t={rt.sim_time:.1f}s target {dp} rpy {rpy}")
        demo_i += 1
    if sensors:
        sensors.update_gimbals(rt.state)
    app.update()
    update_cable_visuals()
    if ros:
        ros.on_render_frame(rt.sim_time)
    frames += 1
    if frames % 150 == 0:
        e = rt.state["pos"][0] - rt.ctrl.p_ref
        print(f"[cdpr] t={rt.sim_time:6.1f}s  RTF={rt.sim_time / (time.time() - wall0):.2f}  "
              f"platform err={np.linalg.norm(e) * 1000:.1f} mm  T=[{rt.last['T'].min():.1f}, {rt.last['T'].max():.1f}] N")

rt.shutdown()
if ros:
    ros.shutdown()
omni.timeline.get_timeline_interface().stop()
app.close()
