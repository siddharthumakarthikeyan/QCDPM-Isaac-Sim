"""Block-assembly task: pick each block from the staging stock and place it at its pattern pose.

Per block (all moves are vertical-horizontal-vertical at a transit height that clears the stock and the structure):
    TO_PICK     rise to transit height, travel above the source block, descend to the approach height
    ALIGN       wait until the platform has settled over the block; wrist yaw = block yaw
    DESCEND     slow guarded descent to the grasp height (stops on load-cell contact)
    GRASP       squeeze; verify the fingers stopped at the block width
    LIFT        rise to transit height; verify the load cell gained ~ the block weight
    TRANSIT     travel above the target; wrist turns to the target yaw on the way
    PLACE       slow descent until the block lands (load cell loses >= 50 % of the block weight) or target - 5 mm
    RELEASE     open the gripper
    RETRACT     rise to transit height
The planner itself is a tick() state machine driven from the render loop (10 Hz is plenty); it only sets platform
and gripper references, the 1 kHz physics/control in runtime.py does the rest.
"""

import numpy as np

from .tool import grasp_height, tool_geometry

G = 9.81


class AssemblyTask:
    def __init__(self, rt, cfg, stock_poses, target_poses, block_paths):
        self.rt, self.cfg = rt, cfg
        self.src, self.dst = stock_poses, target_poses
        self.paths = block_paths
        t = cfg["task"]
        self.H = cfg["blocks"]["size"][2]
        self.W = cfg["blocks"]["size"][1]
        self.m_block = cfg["blocks"]["mass"]
        self.geo = tool_geometry(cfg)
        self.clear = t["transit_clearance"]
        self.approach = t["approach_height"]
        self.v_desc = t["descent_speed"]
        self.v_transit = t["transit_speed"]
        self.contact = t["contact_force"]
        self.i = 0
        self.state = "TO_PICK"
        self.sub = 0
        self.t_state = 0.0
        self.placed = []
        self.failed = []
        self.events = []          # (time, block index, state) for the video edit
        self.ft_ref = 0.0
        self.done = False
        self.stock_top = max(p[2] for p in stock_poses) + self.H / 2

    # ------------------------------------------------------------------ geometry
    def _structure_top(self):
        tops = [self.dst[k][2] + self.H / 2 for k in self.placed] or [0.0]
        return max(tops)

    def _transit_z(self):
        """Platform height so that a carried block's bottom clears everything by `clear`."""
        top = max(self.stock_top, self._structure_top())
        return top + self.clear + self.H + 0.01 + self.geo["palm_drop"]

    def _place_z(self, k):
        return grasp_height(self.cfg, self.dst[k][2])

    # ------------------------------------------------------------------ helpers
    def _goto(self, xyz, speed):
        self.rt.max_lin_speed = speed
        self.rt.set_platform_target(np.asarray(xyz, float))

    def _at(self, tol=0.004, vtol=0.02):
        s = self.rt.state
        return (np.linalg.norm(s["pos"][0] - self.rt.ref_target_p) < tol and np.linalg.norm(s["v"][0]) < vtol
                and np.linalg.norm(self.rt.ctrl.p_ref - self.rt.ref_target_p) < 1e-6)

    def _enter(self, state):
        self.state, self.sub, self.t_state = state, 0, self.rt.sim_time
        self.events.append((self.rt.sim_time, self.i, state))

    def _elapsed(self):
        return self.rt.sim_time - self.t_state

    # ------------------------------------------------------------------ main loop
    def tick(self):
        if self.done:
            return
        rt, tool = self.rt, self.rt.tool
        s = rt.state
        k = self.i
        sx, sy, sz, syaw = self.src[k]
        tx, ty, tz, tyaw = self.dst[k]
        zt = self._transit_z()
        p = s["pos"][0]

        if self.state == "TO_PICK":
            legs = [(p[0], p[1], max(zt, p[2])), (sx, sy, zt), (sx, sy, grasp_height(self.cfg, sz) + self.approach)]
            if self.sub == 0:
                tool.open()
                self._goto(legs[0], self.v_transit)
                self.sub = 1
            elif self.sub <= 3 and self._at(tol=0.01, vtol=0.05):
                if self.sub < 3:
                    self._goto(legs[self.sub], self.v_transit)
                    tool.set_yaw(syaw)
                    self.sub += 1
                else:
                    self._enter("ALIGN")
        elif self.state == "ALIGN":
            if self._at(tol=0.003, vtol=0.01) and abs(np.angle(np.exp(1j * (tool.yaw - syaw)))) < 0.01:
                self.ft_ref = tool.ft_z
                self._goto((sx, sy, grasp_height(self.cfg, sz)), self.v_desc)
                self._enter("DESCEND")
        elif self.state == "DESCEND":
            if abs(tool.ft_z - self.ft_ref) > self.contact:       # unexpected contact: stop where we are
                self._goto(p + [0, 0, 0.005], self.v_desc)
                self._enter("GRASP")
            elif self._at(tol=0.003, vtol=0.01):
                self._enter("GRASP")
        elif self.state == "GRASP":
            if self.sub == 0:
                tool.close()
                self.sub = 1
            elif self._elapsed() > 1.5:
                if abs(tool.width - self.W) < 0.006:
                    # ft_ref stays the pre-grasp reading: squeezing already lifts part of the block weight
                    self._goto((sx, sy, zt), self.v_desc * 2)
                    self._enter("LIFT")
                else:                                             # missed: open, back off, retry once from above
                    self._fail(f"grasp width {tool.width * 1000:.0f} mm")
        elif self.state == "LIFT":
            if self._at(tol=0.01, vtol=0.05):
                gained = tool.ft_z - self.ft_ref
                if gained < 0.6 * self.m_block * G:
                    self._fail(f"load gain {gained:.1f} N")
                else:
                    self._goto((tx, ty, zt), self.v_transit)
                    tool.set_yaw(tyaw)
                    self._enter("TRANSIT")
        elif self.state == "TRANSIT":
            if self._at(tol=0.003, vtol=0.01) and abs(np.angle(np.exp(1j * (tool.yaw - tyaw)))) < 0.01:
                self.ft_ref = tool.ft_z
                self._goto((tx, ty, self._place_z(k) - 0.005), self.v_desc)
                self._enter("PLACE")
        elif self.state == "PLACE":
            landed = self.ft_ref - tool.ft_z > 0.5 * self.m_block * G
            if landed or self._at(tol=0.002, vtol=0.01):
                self._goto(p, self.v_desc)                       # hold here while releasing
                self._enter("RELEASE")
        elif self.state == "RELEASE":
            if self.sub == 0:
                tool.open()
                self.sub = 1
            elif self._elapsed() > 1.2:
                self._goto((tx, ty, zt + 0.0), self.v_desc * 2)
                self._enter("RETRACT")
        elif self.state == "RETRACT":
            if self._at(tol=0.01, vtol=0.05):
                self.placed.append(k)
                self._next()

    def _fail(self, why):
        self.failed.append((self.i, why))
        self.rt.tool.open()
        print(f"[task] block {self.i}: {why} -> skipped", flush=True)
        self._next()

    def _next(self):
        self.i += 1
        if self.i >= len(self.dst):
            self.done = True
            self.events.append((self.rt.sim_time, self.i, "DONE"))
            print(f"[task] done: {len(self.placed)} placed, {len(self.failed)} failed, t={self.rt.sim_time:.1f}s", flush=True)
        else:
            print(f"[task] block {self.i}/{len(self.dst)} t={self.rt.sim_time:.1f}s", flush=True)
            self._enter("TO_PICK")
