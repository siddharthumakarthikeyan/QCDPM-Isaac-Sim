"""Shot list and edit of the thesis trailer (pure Python: imported by tools/render_shots.py inside Isaac and by
tools/compose_trailer.py outside it).

A shot is rendered from one recording with a camera that orbits an anchor:
    eye    = anchor(t) + (r cos az, r sin az, z)        r, az [deg], z, focal given as (start, end)
    target = look_anchor(t) + look
Anchors: platform, grip, drone_k, ugv_k (follow the recording, smoothed), build, staging, mid, truck (fixed).
`t` is the simulation time span that is played during `dur` seconds of film.
"""

import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FPS = 30
RES = (1920, 804)                 # 2.39:1 picture, letterboxed to 1920x1080 in the edit
SUN = dict(sun_elevation_deg=24.0, sun_azimuth_deg=205.0, turbidity=3.6, sun_intensity=3000.0, sky_intensity=620.0)   # low, warm sun

TITLE = "Quadrotor-Based Mobile Cable-Driven Parallel Manipulator"
AUTHOR = "Dr. Siddharth Umakarthikeyan"
SUPERVISOR = "Dr. Badri Narayanan R"
UNIVERSITY = "SASTRA Deemed to be University, India"
CLOSING = ""    # optional last line of the end card (left empty: credits only)


def _events(name):
    p = os.path.join(ROOT, "recordings", f"{name}.npz")
    if not os.path.exists(p):
        return None, None
    rec = np.load(p)
    ev = json.loads(str(rec["events"]))
    return (lambda b, s: next(t for t, bb, ss in ev if bb == b and ss == s)), float(rec["t"][-1])


def shot(id, rec, t, dur, anchor, r, az, z, look=(0, 0, 0), focal=30.0, look_anchor=None, smooth=15, fstop=0.0):
    two = lambda v: tuple(v) if isinstance(v, (tuple, list)) else (v, v)   # noqa: E731
    return dict(id=id, rec=rec, t=two(t), dur=dur, anchor=anchor, look_anchor=look_anchor or anchor, r=two(r), az=two(az),
                z=two(z), look=tuple(look), focal=two(focal), smooth=smooth, fstop=fstop)


def shots():
    out = []
    ev, tend = _events("tower")
    if ev:
        B = 4                                             # the block followed through pick / carry / place
        out += [
            # ---- opening and the system at rest over the footing
            shot("open", "tower", (0.3, 3.3), 6.6, "mid", (30, 19), (-128, -119), (0.9, 3.2), (1.2, 0, 1.9), (40, 35)),
            shot("system", "tower", (1.0, 5.0), 4.0, "build", 9.5, (-108, -78), (2.9, 2.4), (0, 0, 2.0), 26),
            shot("drone", "tower", (2.0, 5.0), 2.6, "drone_1", (0.95, 0.85), (-160, -128), (-0.10, 0.02), (0, 0, -0.03), 30, fstop=900.0),
            shot("ugv", "tower", (2.0, 5.0), 2.6, "ugv_3", (1.25, 1.15), (-146, -116), (0.36, 0.28), (0, 0, 0.08), 34, fstop=900.0),
            shot("platform", "tower", (2.0, 5.0), 2.8, "platform", 2.0, (-120, -90), (0.15, -0.05), (0, 0, -0.2), 40, fstop=900.0),
            # ---- model and learning
            shot("lookup", "tower", (2.0, 5.0), 3.5, "platform", (1.25, 1.0), (-70, -40), -1.12, (0, 0, 1.2), 17),
            shot("ws", "tower", (1.0, 5.0), 4.5, "build", (8.2, 7.4), (-118, -86), (2.6, 2.2), (0, 0, 1.55), 30),
            # ---- manipulation, close: one block from the stock to the tower
            shot("pick", "tower", (ev(B, "GRASP") - 2.2, ev(B, "GRASP") + 0.9), 2.6, "grip", 1.35, (-64, -58), (0.24, 0.3),
                 (0, 0, -0.05), 32, fstop=900.0),
            shot("grasp_macro", "tower", (ev(B, "GRASP") - 0.6, ev(B, "LIFT") + 2.4), 4.0, "grip", (0.62, 0.56), (-28, -16), (0.10, 0.06),
                 (0, 0, -0.06), 45, fstop=500.0),
            shot("carry", "tower", (ev(B, "TRANSIT") + 1.0, ev(B, "PLACE") - 3.0), 3.0, "platform", 4.6, (-125, -114),
                 (0.7, 0.55), (0, 0, -0.45), 28, smooth=30),
            shot("seat_macro", "tower", (ev(B, "RELEASE") - 2.6, ev(B, "RETRACT") + 2.2), 4.4, "grip", (0.95, 0.62), (-172, -128),
                 (0.10, -0.08), (0, 0, -0.14), 40, fstop=600.0),
            # ---- construction
            shot("tower_lapse", "tower", (ev(0, "RETRACT"), tend - 4.0), 3.6, "build", 5.6, (-112, -62), (2.4, 2.0),
                 (0.7, 0, 0.75), 24),
            shot("tower_hero", "tower", tend, 1.6, "build", 2.3, (-100, -82), (0.5, 0.65), (0, 0, 0.62), 17),
        ]
    if ev:                                                # poster stills (16:9, rendered with --res): block in hand over the tower
        tp = ev(4, "PLACE") - 1.0
        out += [
            shot("poster_a", "tower", tp, 0.1, "build", 8.6, -124, 0.75, (1.5, 0.2, 2.25), 21),
            shot("poster_b", "tower", tp, 0.1, "build", 6.4, -150, 0.55, (0.9, 0.0, 2.0), 19),
            shot("poster_c", "tower", tp, 0.1, "build", 9.5, -100, 1.6, (1.9, 0.0, 1.9), 24),
            shot("poster", "tower", tp, 0.1, "ugv_3", 1.55, -131, 0.34, (-0.55, 0, 0.55), 16, look_anchor="platform"),
            shot("poster_e", "tower", tp, 0.1, "platform", 2.5, -112, -1.0, (0.1, 0, 0.75), 16),
            shot("poster_f", "tower", tp, 0.1, "platform", 3.3, -62, -1.15, (0.2, 0, 0.55), 18),
        ]
    ev, tend = _events("pyramid")
    if ev:
        out += [
            shot("pyr_lapse", "pyramid", (ev(0, "RETRACT"), tend - 4.0), 3.2, "build", 5.8, (-120, -75), (2.6, 2.1),
                 (0.7, 0, 0.5), 24),
            shot("pyr_hero", "pyramid", tend, 1.6, "build", 3.4, (-106, -90), (0.6, 0.8), (0, 0, 0.3), 22),
        ]
    ev, tend = _events("compound")
    if ev:
        tm = ev(24, "TO_PICK")
        out += [
            shot("wall_lapse", "compound", (ev(0, "RETRACT"), tm), 2.6, "build", 6.2, (-104, -70), (3.3, 2.9),
                 (0.8, 0, 0.6), 24),
            shot("wall_top", "compound", (tm, tend - 4.0), 2.4, "build", (0.7, 0.7), (-90, -75), (9.5, 9.0), (0, 0, 0.3), 22),
            shot("wall_hero", "compound", tend, 8.4, "build", 3.9, (-158, -88), (0.55, 1.2), (0, 0, 0.26), 21),
            shot("end", "compound", tend, 6.5, "mid", (11, 21), (-112, -122), (2.0, 5.6), (1.0, 0, 1.6), 35),
        ]
    ev, tend = _events("track")
    if ev:
        t0 = ev(0, "TRACK")
        out += [
            shot("track_wide", "track", (t0 + 6.0, t0 + 21.0), 5.0, (0, 0, 0), (7.6, 7.0), (-116, -92), (2.5, 2.2), (0, 0, 1.25), 30),
            shot("track_close", "track", (t0 + 27.0, t0 + 35.0), 4.0, "platform", (2.6, 2.3), (-70, -50), (0.35, 0.2), (0, 0, -0.25), 30,
                 smooth=20),
        ]
    ev, tend = _events("plan")
    if ev:
        t0 = ev(0, "TRACK")
        out += [
            shot("plan_top", "plan", (0.5, 2.9), 3.6, (-1.9, 0.1, 0), (5.2, 4.6), (-180, -176), (16.5, 15.5), (0, 0, 0.6), 27),
            shot("plan_go1", "plan", (t0 + 9.5, t0 + 23.5), 3.5, (-1.2, -1.3, 0), (9.0, 8.2), (-152, -128), (3.6, 3.0), (-1.6, 0.6, 0.9), 26),
            shot("plan_go2", "plan", (t0 + 30.0, t0 + 38.0), 3.5, (-0.6, 2.4, 0), (5.4, 4.8), (38, 62), (1.5, 1.2), (-0.5, 0.3, 0.75), 24),
        ]
    # ---- applications (portfolio page; not in the trailer edit)
    ev, tend = _events("paint")
    if ev:
        t0, t1 = ev(0, "TRACK"), ev(0, "DONE")
        out += [
            shot("paint_wide", "paint", (t0 + 2.0, t0 + 12.0), 5.0, (0.0, -1.5, 0), (11.2, 10.2), (24, 46), (2.6, 2.2), (-0.8, -1.0, 1.5), 22),   # inside the doorway
            shot("paint_close", "paint", (t0 + 13.5, t0 + 21.5), 4.0, "platform", (1.7, 1.45), (48, 82), (-0.72, -0.8), (0, 0, -0.98), 34,
                 fstop=700.0, smooth=20),
            shot("paint_top", "paint", (t0 + 1.0, t1), 4.5, (0.1, 0.85, 0), (0.6, 0.6), (-90, -78), (8.6, 7.6), (0, 0, 0.9), 27),
            shot("paint_hero", "paint", tend, 3.5, (0.0, 0.6, 0.9), (7.4, 6.4), (18, 46), (2.3, 1.9), (0, -0.4, 0.2), 26),
        ]
    ev, tend = _events("print")
    if ev:
        t0, t1 = ev(0, "TRACK"), ev(0, "DONE")
        out += [
            shot("print_lapse", "print", (t0 + 2.0, t1), 6.0, "build", (5.0, 4.4), (-120, -55), (2.1, 1.6), (0.4, 0, 0.5), 25),
            shot("print_close", "print", (t0 + 60.0, t0 + 68.0), 4.0, "platform", (1.25, 1.1), (-70, -35), (-0.52, -0.56), (0, 0, -0.70), 38,
                 fstop=600.0, smooth=20),
            shot("print_hero", "print", tend, 3.5, "build", (2.7, 2.4), (-130, -95), (0.55, 0.85), (0, 0, 0.25), 24),
        ]
    return out


# ------------------------------------------------------------------------------------------------ the edit
# clip(shot, a, b): seconds a..b of the rendered shot. Overlays (all optional):
#   chapter : small section label, top left          caption : lower-third title + line
#   labels  : callouts pinned to scene points        hud     : live cable tensions / load cell
#   fact    : one physical fact, top left            speed   : time-lapse tag
#   tags    : cable tension tags only                viz     : "workspace" | "track" | "plan" | "path" scene graphics
#   card    : ("title" | "advantage" | "credits", ...) full-frame typography
def edit(have):
    """`have`: set of rendered shot ids. Returns the list of clips in order (90 s when everything is rendered)."""
    E = []

    def clip(s, a=0.0, b=None, **kw):
        if s in have:
            E.append(dict(shot=s, a=a, b=b, **kw))

    clip("open", 0.0, 6.0, card=("title",))
    ch = "THE SYSTEM"
    clip("system", chapter=ch, caption=("Eight robots. Eight cables. One platform.",
                                        "Four quadrotors above, four ground robots below, each with its own winch"),
         labels=[("drone_1", "Quadrotor + winch"), ("ugv_0", "Ground robot + winch"), ("platform", "6-DOF platform")])
    clip("drone", 0.0, 2.2, chapter=ch, caption=("Aerial anchors", "Quadrotors carry the upper cables and reel them on board"))
    clip("ugv", 0.0, 2.2, chapter=ch, caption=("Ground anchors", "Differential-drive robots hold the lower cables"))
    clip("platform", chapter=ch, caption=("The end-effector", "Motorised wrist, a load cell and a tool that can be changed"))
    ch = "MODEL AND LEARNING"
    clip("lookup", chapter=ch, tags=True, fact=("W(p) t = w", "Eight cable tensions balance six components of wrench, within limits"),
         caption=("One model for the whole team", "Drones, ground robots, cables and platform in a single set of equations"))
    clip("ws", 0.0, 4.0, chapter=ch, viz="workspace", caption=("A learned workspace", "A Random Forest predicts where the load can be held, without solving for tensions"))
    ch = "CONTROL"
    clip("track_wide", chapter=ch, viz="track", speed=True, caption=("Tracking a defined trajectory", "A 3D figure-eight, 1.8 m wide, flown by the whole team"))
    clip("track_close", 0.0, 3.0, chapter=ch, viz="track", tags=True, speed=True, caption=("Millimetre accuracy", "Commanded and measured end-effector position, live"))
    ch = "PLANNING"
    clip("plan_top", chapter=ch, viz="plan", caption=("A path for the whole team", "RRT* samples it, pruning shortens it, B\u00e9zier arcs smooth the bends"))
    clip("plan_go1", chapter=ch, viz="path", speed=True, caption=("Around the cabin", "Platform, ground robots and all eight cables stay clear"))
    clip("plan_go2", chapter=ch, viz="path", speed=True, caption=("Over the pallet", "The ground robots straddle it while the payload is lifted across"))
    ch = "APPLICATION: CONSTRUCTION"
    clip("grasp_macro", chapter=ch, hud=True, fact=("Held by friction alone", "80 N squeeze on rubber pads. No glue, no attachment joint."),
         caption=("Grasp and lift", "The load cell picks up the block: 2.2 kg, about 21.6 N"))
    clip("seat_macro", chapter=ch, hud=True, fact=("Contact on the real bearing faces", "Hollow blocks, dry-stacked. Collision uses the true shell geometry."),
         caption=("Place", "The load cell unloads as the block seats, then the jaws open"))
    clip("tower_lapse", chapter=ch, caption=("Twisted tower", "6 blocks, each course turned 22.5\u00b0"), speed=True)
    clip("wall_lapse", chapter=ch, caption=("Enclosure", "42 blocks, three courses, interlocked corners"), speed=True)
    clip("wall_top", chapter=ch, caption=("Enclosure", "42 blocks, three courses, interlocked corners"), speed=True)
    ch = "APPLICATION: AIRCRAFT PAINTING"
    clip("paint_wide", 0.8, 4.1, chapter=ch, speed=True, caption=("No scaffolding", "The team works around the aircraft where it stands"))
    # the first 2 s of paint_close have a cable right in front of the lens
    clip("paint_close", 2.0, 4.0, chapter=ch, speed=True, caption=("A spray gun in place of the gripper", "Lanes planned so the cables clear the wing"))
    clip("paint_top", 0.5, 4.5, chapter=ch, speed=True, caption=("Lane by lane", "Robots hold position while the winches do the work"))
    ch = "APPLICATION: 3D PRINTING"
    clip("print_lapse", 1.5, 6.0, chapter=ch, speed=True, caption=("A printer as large as the site", "One continuous bead, fed by hose from a tanker"))
    clip("print_close", 0.5, 3.0, chapter=ch, caption=("Layer on layer", "The nozzle follows a twisted, four-lobed spiral"))
    clip("print_hero", 0.5, 2.5, chapter=ch, caption=("Fourteen layers", "No gantry, no rails, no frame"))
    ch = "WHY IT MATTERS"
    adv = [("No fixed frame", "Nothing to erect on site. The robots are the structure."),
           ("Pulls down as well as up", "An all-aerial cable team can only pull upward. Ground anchors add the rest.")]
    for i, a in enumerate(adv):
        clip("wall_hero", 2.8 * i, 2.8 * (i + 1), chapter=ch, card=("advantage", i + 1, *a))
    clip("end", 0.0, 6.0, card=("credits",))
    return E
