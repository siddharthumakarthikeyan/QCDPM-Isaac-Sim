"""Platform end-effector: the cube platform as the root of an articulation carrying a wrist-yaw joint and a
parallel-jaw gripper.

    /World/Platform                 ArticulationRoot (floating base)
      base                          cube, 8 cable eyelets, AprilTags (top tag + 4-tag bottom bundle)
      wrist   <- joint_wrist_yaw    revolute about z, position-driven (motorised wrist)
      tool    <- joint_wrist_ft     axial load cell: stiff prismatic spring, force = -k q - c q_dot
      finger_l <- joint_finger_l    prismatic (+y), force-limited position drive
      finger_r <- joint_finger_r    prismatic (-y)
The finger drives saturate at max_force, so commanding "closed" on a block squeezes it with that force, like an
electric gripper's force limit.
"""

import numpy as np
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

from .cdpr_control import corner_points
from .scene import _bind_phys, _box, _cyl, _phys_material, _quad, _rigid, _sphere, _xform

PLATFORM_ROOT = "/World/Platform"
PLATFORM_BASE = "/World/Platform/base"


def tool_geometry(cfg):
    """Vertical layout below the platform centre (all distances positive downwards from the base centre)."""
    s = cfg["platform"]["side"]
    g = cfg["gripper"]
    z_wrist = s / 2 + g["wrist"]["height"] / 2
    z_tool_top = s / 2 + g["wrist"]["height"]
    z_palm = z_tool_top + g["tool"]["tube_length"] + g["tool"]["palm"][2] / 2
    z_finger = z_palm + g["tool"]["palm"][2] / 2 + g["finger"]["size"][2] / 2
    palm_bottom = z_palm + g["tool"]["palm"][2] / 2
    return dict(z_wrist=z_wrist, z_tool_top=z_tool_top, z_palm=z_palm, z_finger=z_finger, palm_drop=palm_bottom,
                tip_drop=z_finger + g["finger"]["size"][2] / 2)


def grasp_height(cfg, block_center_z, clearance=0.01):
    """Platform-centre height for grasping a block: palm `clearance` above the block top, fingers on its upper part."""
    geo = tool_geometry(cfg)
    return block_center_z + cfg["blocks"]["size"][2] / 2 + clearance + geo["palm_drop"]


def build_platform(stage, cfg, mats):
    pc, g = cfg["platform"], cfg["gripper"]
    s, m = pc["side"], pc["mass"]
    p0 = np.array(pc["start_position"], float)
    geo = tool_geometry(cfg)

    root = _xform(stage, PLATFORM_ROOT)
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    pxa = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    pxa.CreateEnabledSelfCollisionsAttr(False)
    pxa.CreateSolverPositionIterationCountAttr(32)
    pxa.CreateSolverVelocityIterationCountAttr(8)
    pxa.CreateSleepThresholdAttr(0.0)

    # ---- base (the cable platform)
    base = _xform(stage, PLATFORM_BASE, p0)
    _rigid(base.GetPrim(), m, (m * s * s / 6,) * 3)
    _box(stage, PLATFORM_BASE + "/body", (s, s, s), mat=mats["graphite"], collide=True)
    tag = pc["apriltag"]
    _quad(stage, PLATFORM_BASE + "/tag_top", tag["size"] * 10 / 8, s / 2 + 0.001, False, mats["tag0"])
    off, q = tag["bottom_offset"], tag["bottom_size"] * 10 / 8
    for i, (sx, sy) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
        quad = _quad(stage, f"{PLATFORM_BASE}/tag_bottom_{i}", q, -s / 2 - 0.001, True, mats[f"tagb{i}"])
        UsdGeom.XformCommonAPI(quad).SetTranslate(Gf.Vec3d(sx * off, sy * off, 0))
    for i, b in enumerate(corner_points(s)):
        _sphere(stage, f"{PLATFORM_BASE}/eyelet_{i}", 0.014, b, mats["accent"])

    # ---- wrist (rotary actuator housing)
    w = g["wrist"]
    wrist_p = p0 - [0, 0, geo["z_wrist"]]
    wr = _xform(stage, PLATFORM_ROOT + "/wrist", wrist_p)
    _rigid(wr.GetPrim(), w["mass"], (w["mass"] * w["radius"] ** 2 / 4 * 1.5,) * 2 + (w["mass"] * w["radius"] ** 2 / 2,))
    _cyl(stage, PLATFORM_ROOT + "/wrist/housing", w["radius"], w["height"], mat=mats["graphite"])
    _box(stage, PLATFORM_ROOT + "/wrist/index_mark", (0.012, 0.02, w["height"] * 0.9), (w["radius"], 0, 0), mat=mats["accent"])
    j = UsdPhysics.RevoluteJoint.Define(stage, PLATFORM_ROOT + "/joint_wrist_yaw")
    _joint_frames(j, PLATFORM_BASE, PLATFORM_ROOT + "/wrist", (0, 0, -s / 2), (0, 0, w["height"] / 2), "Z")
    d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    d.CreateTypeAttr("force")
    d.CreateStiffnessAttr(float(w["stiffness"] * np.pi / 180))   # USD angular drive units are per degree
    d.CreateDampingAttr(float(w["damping"] * np.pi / 180))
    d.CreateMaxForceAttr(float(w["max_torque"]))
    d.CreateTargetPositionAttr(0.0)
    PhysxSchema.PhysxJointAPI.Apply(j.GetPrim()).CreateMaxJointVelocityAttr(float(np.rad2deg(w["max_speed"])))

    # ---- tool: force/torque sensor + extension tube + palm, fixed to the wrist
    t = g["tool"]
    tube_c = p0 - [0, 0, geo["z_tool_top"] + t["tube_length"] / 2]
    tool = _xform(stage, PLATFORM_ROOT + "/tool", tube_c)
    L = t["tube_length"]
    _rigid(tool.GetPrim(), t["mass"], (t["mass"] * L * L / 12,) * 2 + (t["mass"] * t["tube_radius"] ** 2,))
    _cyl(stage, PLATFORM_ROOT + "/tool/ft_sensor", 0.04, 0.03, pos=(0, 0, L / 2 - 0.015), mat=mats["accent"])
    _cyl(stage, PLATFORM_ROOT + "/tool/tube", t["tube_radius"], L - 0.03, pos=(0, 0, -0.015), mat=mats["alu"])
    palm_z = -(geo["z_palm"] - geo["z_tool_top"] - L / 2)
    _box(stage, PLATFORM_ROOT + "/tool/palm", t["palm"], (0, 0, palm_z), mat=mats["graphite"], collide=True)
    # axial load cell: a stiff spring along the tool axis; its deflection x stiffness is the measured force
    j = UsdPhysics.PrismaticJoint.Define(stage, PLATFORM_ROOT + "/joint_wrist_ft")
    _joint_frames(j, PLATFORM_ROOT + "/wrist", PLATFORM_ROOT + "/tool", (0, 0, -w["height"] / 2), (0, 0, L / 2), "Z")
    j.CreateLowerLimitAttr(-0.003)
    j.CreateUpperLimitAttr(0.003)
    d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
    d.CreateTypeAttr("force")
    d.CreateStiffnessAttr(float(g["ft_sensor"]["stiffness"]))
    d.CreateDampingAttr(float(g["ft_sensor"]["damping"]))
    d.CreateMaxForceAttr(1.0e5)
    d.CreateTargetPositionAttr(0.0)

    # ---- fingers
    f = g["finger"]
    pm_pad = _phys_material(stage, "/World/PhysicsMaterials/gripper_pad", **g["pad_friction"])
    half = f["stroke"] / 2
    for side, name in ((1, "l"), (-1, "r")):
        fp = p0 - [0, 0, geo["z_finger"]] + [0, side * (half + f["size"][1] / 2), 0]
        fl = _xform(stage, f"{PLATFORM_ROOT}/finger_{name}", fp)
        m_f = f["mass"]
        sx, sy, sz = f["size"]
        _rigid(fl.GetPrim(), m_f, (m_f * (sy * sy + sz * sz) / 12, m_f * (sx * sx + sz * sz) / 12, m_f * (sx * sx + sy * sy) / 12))
        body = _box(stage, f"{PLATFORM_ROOT}/finger_{name}/body", f["size"], mat=mats["alu"], collide=True)
        _bind_phys(body.GetPrim(), pm_pad)
        _box(stage, f"{PLATFORM_ROOT}/finger_{name}/pad", (sx * 0.9, 0.004, sz * 0.7), (0, -side * (sy / 2 + 0.002), -sz * 0.1), mat=mats["rubber"])
        j = UsdPhysics.PrismaticJoint.Define(stage, f"{PLATFORM_ROOT}/joint_finger_{name}")
        # joint position = signed inner-face offset from the tool axis (left: [0, +half], right: [-half, 0])
        _joint_frames(j, PLATFORM_ROOT + "/tool", f"{PLATFORM_ROOT}/finger_{name}",
                      (0, 0, palm_z - t["palm"][2] / 2 - sz / 2), (0, -side * sy / 2, 0), "Y")
        j.CreateLowerLimitAttr(0.0 if side > 0 else -half)
        j.CreateUpperLimitAttr(half if side > 0 else 0.0)
        d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
        d.CreateTypeAttr("force")
        d.CreateStiffnessAttr(float(f["stiffness"]))
        d.CreateDampingAttr(float(f["damping"]))
        d.CreateMaxForceAttr(float(f["max_force"]))
        d.CreateTargetPositionAttr(float(side * half))
    return geo


def _joint_frames(j, body0, body1, pos0, pos1, axis):
    j.CreateBody0Rel().SetTargets([body0])
    j.CreateBody1Rel().SetTargets([body1])
    j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, pos0)))
    j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    j.CreateLocalPos1Attr(Gf.Vec3f(*map(float, pos1)))
    j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    if axis:
        j.CreateAxisAttr(axis)
