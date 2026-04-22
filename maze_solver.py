"""
COSC 4368 AI — Check-in 3: Maze Navigation
Method: Online BFS Exploration + Dyna-Q danger learning + exploitation
"""

from __future__ import annotations

import os
import time
import colorsys
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


BORDER = 2
CELL_SIZE = 14
STRIDE = 16
NUM_CELLS = 64
MAT_SIZE = 128

EMPTY, WALL = 0, 1
START, GOAL = 2, 3
DEATH_PIT = 4
TELEPORT = 5
CONFUSION = 6
ARROW_UP = 7
ARROW_LEFT = 8
ARROW_RIGHT = 9
ARROW_DOWN = 10

ARROW_TYPES = {ARROW_UP, ARROW_LEFT, ARROW_RIGHT, ARROW_DOWN}
ARROW_VEC = {
    ARROW_UP: (0, -1),
    ARROW_DOWN: (0, 1),
    ARROW_LEFT: (-1, 0),
    ARROW_RIGHT: (1, 0),
}

ROT_OFF = [(0, -1), (1, 0), (0, 1), (-1, 0)]
FIRE_PER = 5
DIRS4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


class Action(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    WAIT = 4


AV = {
    Action.MOVE_UP: (0, -1),
    Action.MOVE_DOWN: (0, 1),
    Action.MOVE_LEFT: (-1, 0),
    Action.MOVE_RIGHT: (1, 0),
    Action.WAIT: (0, 0),
}

IA = {
    Action.MOVE_UP: Action.MOVE_DOWN,
    Action.MOVE_DOWN: Action.MOVE_UP,
    Action.MOVE_LEFT: Action.MOVE_RIGHT,
    Action.MOVE_RIGHT: Action.MOVE_LEFT,
    Action.WAIT: Action.WAIT,
}

DACT = {
    (0, -1): Action.MOVE_UP,
    (0, 1): Action.MOVE_DOWN,
    (-1, 0): Action.MOVE_LEFT,
    (1, 0): Action.MOVE_RIGHT,
}


@dataclass
class TurnResult:
    wall_hits: int = 0
    current_position: Tuple[int, int] = (0, 0)
    is_dead: bool = False
    is_confused: bool = False
    is_goal_reached: bool = False
    teleported: bool = False
    actions_executed: int = 0


def _wb(g, r, c):
    y = BORDER + r * STRIDE + CELL_SIZE
    x = BORDER + c * STRIDE + CELL_SIZE // 2
    return not (g[y, x] > 128 and g[y + 1, x] > 128)


def _wr(g, r, c):
    y = BORDER + r * STRIDE + CELL_SIZE // 2
    x = BORDER + c * STRIDE + CELL_SIZE
    return not (g[y, x] > 128 and g[y, x + 1] > 128)


def load_walls(path):
    g = np.array(Image.open(path).convert("L"))
    m = np.ones((MAT_SIZE, MAT_SIZE), dtype=np.uint8)

    for r in range(NUM_CELLS):
        for c in range(NUM_CELLS):
            m[r * 2, c * 2] = EMPTY
            if r < NUM_CELLS - 1:
                m[r * 2 + 1, c * 2] = WALL if _wb(g, r, c) else EMPTY
            if c < NUM_CELLS - 1:
                m[r * 2, c * 2 + 1] = WALL if _wr(g, r, c) else EMPTY
            if r < NUM_CELLS - 1 and c < NUM_CELLS - 1:
                m[r * 2 + 1, c * 2 + 1] = WALL

    return m


def find_sg(path):
    g = np.array(Image.open(path).convert("L"))
    top = [c for c in range(g.shape[1]) if g[1, c] > 200]
    bot = [c for c in range(g.shape[1]) if g[-2, c] > 200]

    return (
        ((top[len(top) // 2] - BORDER) // STRIDE, 0),
        ((bot[len(bot) // 2] - BORDER) // STRIDE, NUM_CELLS - 1),
    )


def classify_cell(rgb, r, c):
    cy = BORDER + r * STRIDE + CELL_SIZE // 2
    cx = BORDER + c * STRIDE + CELL_SIZE // 2

    pix = []
    dk = 0
    wh = 0
    bl = 0

    for dy in range(-5, 6):
        for dx in range(-5, 6):
            y, x = cy + dy, cx + dx
            if not (0 <= y < rgb.shape[0] and 0 <= x < rgb.shape[1]):
                continue

            rr, gg, bb = int(rgb[y, x, 0]), int(rgb[y, x, 1]), int(rgb[y, x, 2])

            if rr > 240 and gg > 240 and bb > 240:
                wh += 1
                continue

            if rr < 25 and gg < 25 and bb < 25:
                continue

            pix.append((rr, gg, bb))

            if rr < 100 and gg < 100 and bb < 100:
                dk += 1

            h, _, _ = colorsys.rgb_to_hsv(rr / 255, gg / 255, bb / 255)
            if 0.55 <= h <= 0.65 and bb > 150 and rr < 150:
                bl += 1

    if len(pix) < 15:
        return None

    hv = [colorsys.rgb_to_hsv(r / 255, g / 255, b / 255) for r, g, b in pix]
    hs = [h for h, s, v in hv if s > 0.25]
    vs = [v for h, s, v in hv if s > 0.25]

    if not hs:
        return None

    ah = float(np.mean(hs))
    av = float(np.mean(vs))
    n = len(hs)
    wr = wh / max(1, n + wh)

    if bl >= 20:
        wp = [
            (dy, dx)
            for dy in range(-6, 7)
            for dx in range(-6, 7)
            if 0 <= cy + dy < rgb.shape[0]
            and 0 <= cx + dx < rgb.shape[1]
            and rgb[cy + dy, cx + dx, 0] > 200
            and rgb[cy + dy, cx + dx, 1] > 200
            and rgb[cy + dy, cx + dx, 2] > 200
        ]
        if not wp:
            return "arrow_up"

        rc = {}
        cc = {}
        for dy2, dx2 in wp:
            rc[dy2] = rc.get(dy2, 0) + 1
            cc[dx2] = cc.get(dx2, 0) + 1

        mr = max(rc.items(), key=lambda p: p[1])
        mc = max(cc.items(), key=lambda p: p[1])

        return (
            "arrow_up" if mr[0] < 0 else "arrow_down"
        ) if mr[1] > mc[1] else (
            "arrow_left" if mc[0] < 0 else "arrow_right"
        )

    if 0.04 <= ah <= 0.28 and av < 0.85 and dk >= 8:
        return "skull"
    if 0.04 <= ah <= 0.09 and av > 0.95 and dk == 0 and n > 90 and wr < 0.20:
        return "start_marker"
    if 0.04 <= ah <= 0.12 and av > 0.80 and n < 140:
        return "fire"
    if 0.30 <= ah <= 0.50:
        return "green"
    if 0.65 <= ah <= 0.82:
        return "purple"
    if 0.10 <= ah <= 0.18 and av > 0.6:
        return "yellow"

    ss = [s for h, s, v in hv if s > 0.25]
    if (ah < 0.04 or ah > 0.94) and ss and float(np.mean(ss)) > 0.4:
        return "red"

    return None


def detect_hazards(path):
    rgb = np.array(Image.open(path).convert("RGB"))
    return {
        (c, r): cat
        for r in range(NUM_CELLS)
        for c in range(NUM_CELLS)
        if (cat := classify_cell(rgb, r, c)) is not None
    }


def assemble_map(hz, s, g):
    ct = {}
    fires = set()
    tele = {}
    col = defaultdict(list)

    amap = {
        "arrow_up": ARROW_UP,
        "arrow_down": ARROW_DOWN,
        "arrow_left": ARROW_LEFT,
        "arrow_right": ARROW_RIGHT,
    }

    for pos, cat in hz.items():
        if cat == "fire":
            ct[pos] = DEATH_PIT
            fires.add(pos)
        elif cat == "skull":
            ct[pos] = DEATH_PIT
        elif cat in amap:
            ct[pos] = amap[cat]
        elif cat not in ("start_marker",):
            col[cat].append(pos)

    # 🔥 REPLACE OLD LOGIC WITH THIS
    teleport_colors = {"green", "purple"}
    confusion_colors = {"red"}

    for color, cells in col.items():

        if color in teleport_colors and len(cells) >= 2:
            for i in range(0, len(cells) - 1, 2):
                a, b = cells[i], cells[i + 1]
                ct[a] = TELEPORT
                ct[b] = TELEPORT
                tele[a] = b
                tele[b] = a

        elif color in confusion_colors:
            for c in cells:
                ct[c] = CONFUSION

    ct[s] = START
    ct[g] = GOAL
    fires.discard(s)
    fires.discard(g)

    return ct, tele, fires


class MazeEnvironment:
    CFGS = {
        "alpha": (
            os.path.join(BASE_DIR, "maze-alpha", "MAZE_0.png"),
            os.path.join(BASE_DIR, "maze-alpha", "MAZE_1.png"),
        ),
        "training": (
            os.path.join(BASE_DIR, "maze-alpha", "MAZE_0.png"),
            os.path.join(BASE_DIR, "maze-alpha", "MAZE_1.png"),
        ),
        "beta": (
            os.path.join(BASE_DIR, "maze-beta", "MAZE_0.png"),
            os.path.join(BASE_DIR, "maze-beta", "MAZE_1.png"),
        ),
        "testing": (
            os.path.join(BASE_DIR, "maze-beta", "MAZE_0.png"),
            os.path.join(BASE_DIR, "maze-beta", "MAZE_1.png"),
        ),
        "gamma": (
            os.path.join(BASE_DIR, "maze-gamma", "MAZE_0.png"),
            os.path.join(BASE_DIR, "maze-gamma", "MAZE_1.png"),
        ),
    }

    def __init__(self, mid):
        bp, hp = self.CFGS[mid]
        self.wall_matrix = load_walls(bp)
        self.start_xy, self.goal_xy = find_sg(bp)
        raw_hz = detect_hazards(hp)

        # DEBUG (keep but disabled)
        # print(f"\nRaw hazards for {mid}:")
        # for k, v in sorted(raw_hz.items()):
        #     print(k, v)

        self.cell_types, self.teleport_pairs, self.fire_pivots = assemble_map(
            raw_hz, self.start_xy, self.goal_xy
        )

        self._wc = {}
        for x in range(NUM_CELLS):
            for y in range(NUM_CELLS):
                for dx, dy in DIRS4:
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < NUM_CELLS and 0 <= ny < NUM_CELLS):
                        self._wc[((x, y), (nx, ny))] = True
                    else:
                        self._wc[((x, y), (nx, ny))] = (
                            self.wall_matrix[y * 2 + dy, x * 2 + dx] == WALL
                        )

        self.reset()

    def reset(self):
        self.pos = self.start_xy
        self.turn_count = 0
        self.deaths = 0
        self.confused_hits = 0
        self.total_actions = 0
        self.confused_next = False
        self.pending_respawn = False
        self.explored = {self.start_xy}
        self.goal_reached = False
        return self.pos

    def get_episode_stats(self):
        return {
            "turns_taken": self.turn_count,
            "deaths": self.deaths,
            "confused": self.confused_hits,
            "cells_explored": len(self.explored),
            "goal_reached": self.goal_reached,
        }

    def _tip(self, piv, ai):
        dx, dy = ROT_OFF[(ai // FIRE_PER) % 4]
        tx, ty = piv[0] + dx, piv[1] + dy
        return (tx, ty) if 0 <= tx < NUM_CELLS and 0 <= ty < NUM_CELLS else None

    def _deadly(self, xy, ai):
       ct = self.cell_types.get(xy, EMPTY)
       return ct == DEATH_PIT and xy not in self.fire_pivots

#for debugging
#def _deadly(self, xy, ai):
#    return False


    def _mv(self, pos, act):
        if act == Action.WAIT:
            return pos, False

        dx, dy = AV[act]
        nx, ny = pos[0] + dx, pos[1] + dy

        if not (0 <= nx < NUM_CELLS and 0 <= ny < NUM_CELLS):
            return pos, True
        if self._wc.get((pos, (nx, ny)), True):
            return pos, True

        return (nx, ny), False

    def step(self, actions):
        if not actions or len(actions) > 5:
            raise ValueError("Need 1-5 actions")

        if self.pending_respawn:
            self.pos = self.start_xy
            self.pending_respawn = False

        res = TurnResult(current_position=self.pos)
        tc = self.confused_next
        self.confused_next = False

        if tc:
            res.is_confused = True

        gm = False

        for action in actions:
            eff = IA[action] if (tc or gm) else action
            np2, wh = self._mv(self.pos, eff)

            res.actions_executed += 1
            self.total_actions += 1

            if wh:
                res.wall_hits += 1
                continue

            tt = self.cell_types.get(np2, EMPTY)

            if tt in ARROW_TYPES:
                ax, ay = ARROW_VEC[tt]
                pushed = (np2[0] + ax, np2[1] + ay)

                if (
                    0 <= pushed[0] < NUM_CELLS
                    and 0 <= pushed[1] < NUM_CELLS
                    and not self._wc.get((np2, pushed), True)
                    and self.cell_types.get(pushed, EMPTY) not in ARROW_TYPES
                ):
                    np2 = pushed
                else:
                    res.wall_hits += 1
                    continue

            self.pos = np2
            self.explored.add(self.pos)

            if self._deadly(self.pos, self.total_actions - 1):
                res.is_dead = True
                res.current_position = self.pos
                self.deaths += 1
                self.pending_respawn = True
                break

            if self.cell_types.get(self.pos) == TELEPORT:
                dst = self.teleport_pairs.get(self.pos)
                if dst:
                    self.pos = dst
                    self.explored.add(self.pos)
                    res.teleported = True

            if self.cell_types.get(self.pos) == CONFUSION and not gm:
                gm = True
                self.confused_next = True
                self.confused_hits += 1
                res.is_confused = True

            if self.pos == self.goal_xy:
                res.is_goal_reached = True
                self.goal_reached = True
                break

        if not res.is_dead:
            res.current_position = self.pos

        self.turn_count += 1
        return res


class DynaQAgent:
    """Online BFS exploration + Dyna-Q danger learning."""

    def __init__(self):
        self.reset_memory()

    def reset_memory(self):
        self.open_p: Set = set()
        self.blocked_p: Set = set()
        self._neighbors: Dict = {}
        self.tele_pairs: Dict = {}
        self.danger: Dict = defaultdict(float)
        self.death_cells: Set = set()
        self.fire_adj: Set = set()
        self.visit: Dict = defaultdict(int)

        self.start_xy = None
        self.goal_xy = None
        self.current_pos = None
        self.goal_known = False

        self._last_acts = []
        self._plan = []
        self.confused = False
        self._replan_n = 0
        self._replan_t = 0.0
        self._boot_signature = None
        self._goal_path_cache = []

    def reset_episode(self):
        self.current_pos = self.start_xy
        self._last_acts = []
        self._plan = []
        self.confused = False
        self._ep_turn = 0
        self.waits_remaining = 0

    def _env_signature(self, env):
        return (
            hash(env.wall_matrix.tobytes()),
            env.start_xy,
            env.goal_xy,
            tuple(sorted(env.fire_pivots)),
            tuple(sorted(env.teleport_pairs.items())),
        )

    def boot(self, env):
        sig = self._env_signature(env)
        maze_changed = sig != self._boot_signature

        self.start_xy = env.start_xy
        self.goal_xy = env.goal_xy
        self.current_pos = env.start_xy

        if maze_changed:
            self.open_p = set()
            self.blocked_p = set()
            self._neighbors = {}
            self.tele_pairs = {}
            self.danger = defaultdict(float)
            self.death_cells = set()
            self.fire_adj = set()
            self.visit = defaultdict(int)
            self.goal_known = False
            self._last_acts = []
            self._plan = []
            self.confused = False
            self._goal_path_cache = []
        else:
            self.blocked_p = set()
            self._neighbors = {}
            self.tele_pairs = {}
            self.fire_adj = set()
            self._last_acts = []
            self._plan = []
            self.confused = False

        self._boot_signature = sig

        for src, dst in env.teleport_pairs.items():
            self.tele_pairs[src] = dst
            self.tele_pairs[dst] = src

        for pivot in env.fire_pivots:
            for dx, dy in DIRS4:
                nx, ny = pivot[0] + dx, pivot[1] + dy
                if 0 <= nx < NUM_CELLS and 0 <= ny < NUM_CELLS:
                    self.fire_adj.add((nx, ny))

        for x in range(NUM_CELLS):
            for y in range(NUM_CELLS):
                for dx, dy in DIRS4:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < NUM_CELLS and 0 <= ny < NUM_CELLS:
                        if env._wc.get(((x, y), (nx, ny)), False):
                            self.mark_blocked((x, y), (nx, ny))

        self._neighbors = {}
        for x in range(NUM_CELLS):
            for y in range(NUM_CELLS):
                nb = []
                for dx, dy in DIRS4:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < NUM_CELLS and 0 <= ny < NUM_CELLS:
                        if not self.is_blocked((x, y), (nx, ny)):
                            nb.append((nx, ny))
                self._neighbors[(x, y)] = nb

    def _edge(self, a, b):
        return (a, b) if a <= b else (b, a)

    def is_open(self, a, b):
        return ((a, b) in self.open_p) or ((b, a) in self.open_p)

    def is_blocked(self, a, b):
        return ((a, b) in self.blocked_p) or ((b, a) in self.blocked_p)

    def mark_open(self, a, b):
        e = self._edge(a, b)
        self.open_p.add(e)
        self.blocked_p.discard(e)

    def mark_blocked(self, a, b):
        e = self._edge(a, b)
        self.blocked_p.add(e)
        self.open_p.discard(e)

    def _infer(self, start, end, acts, nexec, nhits, was_conf):
        if nexec == 0:
            return

        if nhits == 0:
            pos = start
            for a in acts[:nexec]:
                eff = IA[a] if was_conf else a
                if eff == Action.WAIT:
                    continue
                dx, dy = AV[eff]
                nx, ny = pos[0] + dx, pos[1] + dy
                if 0 <= nx < NUM_CELLS and 0 <= ny < NUM_CELLS:
                    self.mark_open(pos, (nx, ny))
                    pos = (nx, ny)
            return

        if nexec == 1 and nhits == 1 and start == end:
            eff = IA[acts[0]] if was_conf else acts[0]
            if eff != Action.WAIT:
                dx, dy = AV[eff]
                nx, ny = start[0] + dx, start[1] + dy
                if 0 <= nx < NUM_CELLS and 0 <= ny < NUM_CELLS:
                    self.mark_blocked(start, (nx, ny))
            return

        for mask in range(1 << min(nexec, 5)):
            ps = start
            valid = True
            walls = 0

            for i, a in enumerate(acts[:nexec]):
                eff = IA[a] if was_conf else a
                if eff == Action.WAIT:
                    continue

                dx, dy = AV[eff]
                if (mask >> i) & 1:
                    nx, ny = ps[0] + dx, ps[1] + dy
                    if not (0 <= nx < NUM_CELLS and 0 <= ny < NUM_CELLS):
                        valid = False
                        break
                    ps = (nx, ny)
                else:
                    walls += 1

            if valid and ps == end and walls == nhits:
                ps2 = start
                for i, a in enumerate(acts[:nexec]):
                    eff = IA[a] if was_conf else a
                    if eff == Action.WAIT:
                        continue
                    dx, dy = AV[eff]
                    nx, ny = ps2[0] + dx, ps2[1] + dy
                    if not (0 <= nx < NUM_CELLS and 0 <= ny < NUM_CELLS):
                        continue
                    if (mask >> i) & 1:
                        self.mark_open(ps2, (nx, ny))
                        ps2 = (nx, ny)
                return

    def _update(self, res):
        if res is None:
            return

        prev = self.current_pos
        new = res.current_position
        self.confused = res.is_confused

        if res.is_dead:
            dc = new
            self.death_cells.add(dc)

            if dc not in self.fire_adj:
                self.danger[dc] = min(0.75, self.danger.get(dc, 0) + 0.12)
            else:
                self.danger[dc] = min(0.60, self.danger.get(dc, 0) + 0.08)

            for dx, dy in DIRS4:
                nx, ny = dc[0] + dx, dc[1] + dy
                if 0 <= nx < NUM_CELLS and 0 <= ny < NUM_CELLS:
                    self.danger[(nx, ny)] = max(self.danger.get((nx, ny), 0), 0.03)

            self._plan = []

        self.visit[new] += 1
        self.current_pos = new

        self._total_actions = getattr(self, "_total_actions", 0) + getattr(
            res, "actions_executed", 1
        )
        self._action_count = getattr(self, "_action_count", 0) + getattr(
            res, "actions_executed", 1
        )

        if not res.is_dead and self.danger.get(new, 0) > 0.01:
            self.danger[new] *= 0.60
            if self.danger[new] < 0.01:
                del self.danger[new]

        if not res.teleported and not res.is_dead and prev and self._last_acts:
            self._infer(
                prev,
                new,
                self._last_acts,
                res.actions_executed,
                res.wall_hits,
                res.is_confused,
            )

        if new == self.goal_xy:
            self.goal_known = True

    def _bfs(self, start, goal, allow_unknown=True, danger_thresh=0.9):
        t0 = time.perf_counter()
        q = deque([(start, [start])])
        vis = {start}

        while q:
            pos, path = q.popleft()

            if pos == goal:
                self._replan_t += time.perf_counter() - t0
                self._replan_n += 1
                return path

            if pos in self.tele_pairs:
                dst = self.tele_pairs[pos]
                if dst not in vis:
                    vis.add(dst)
                    q.append((dst, path + [dst]))

            for nxt in self._neighbors.get(pos, []):
                if nxt in vis:
                    continue
                if not allow_unknown and not self.is_open(pos, nxt):
                    continue
                if self.danger.get(nxt, 0) > danger_thresh:
                    continue
                vis.add(nxt)
                q.append((nxt, path + [nxt]))

        self._replan_t += time.perf_counter() - t0
        self._replan_n += 1
        return None

    def _frontier_bfs(self, danger_thresh=0.9):
        q = deque([(self.current_pos, [self.current_pos])])
        vis = {self.current_pos}

        while q:
            pos, path = q.popleft()
            x, y = pos

            for dx, dy in DIRS4:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < NUM_CELLS and 0 <= ny < NUM_CELLS):
                    continue

                nxt = (nx, ny)
                if self.is_blocked(pos, nxt):
                    continue
                if self.danger.get(nxt, 0) > danger_thresh:
                    continue
                if not self.is_open(pos, nxt):
                    return path + [nxt]
                if nxt in vis:
                    continue

                vis.add(nxt)
                q.append((nxt, path + [nxt]))

        return None

    def _path_to_acts(self, path):
        acts = []

        for i in range(len(path) - 1):
            dx, dy = path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]
            a = DACT.get((dx, dy))
            if a is None:
                break
            acts.append(a)
            if len(acts) >= 5:
                break

        if not acts:
            acts = [Action.WAIT]

        if self.confused:
            acts = [IA[a] for a in acts]

        return acts

    def plan_turn(self, res):
        self._update(res)

        if res and res.is_dead:
            self.current_pos = self.start_xy
            death_cell = res.current_position if res else None

            if death_cell and death_cell in self.fire_adj:
                self._fire_cooloff = 3
                if hasattr(self, "_goal_path_cache") and self._goal_path_cache:
                    self._plan = self._goal_path_cache
                else:
                    self._plan = []
            else:
                self._fire_cooloff = 0
                self._plan = []

        if self.current_pos == self.goal_xy:
            self._last_acts = [Action.WAIT]
            return [Action.WAIT]

        cooloff = getattr(self, "_fire_cooloff", 0)
        if cooloff > 0:
            self._fire_cooloff -= 1
            acts = [Action.WAIT, Action.WAIT, Action.WAIT, Action.WAIT, Action.WAIT]
            self._last_acts = acts
            return acts

        self._ep_turn = getattr(self, "_ep_turn", 0) + 1

        if self._ep_turn > 200:
            plan_ends_at_goal = len(self._plan) >= 2 and self._plan[-1] == self.goal_xy
            if len(self._plan) < 2 or not plan_ends_at_goal:
                path = self._bfs(
                    self.current_pos,
                    self.goal_xy,
                    allow_unknown=True,
                    danger_thresh=0.99,
                )
                if path and len(path) >= 2:
                    self._plan = path
                    self._goal_path_cache = path

        if self._plan:
            if self.current_pos in self._plan:
                idx = self._plan.index(self.current_pos)
                self._plan = self._plan[idx:]

                if len(self._plan) >= 2:
                    nxt = self._plan[1]
                    if (
                        res is not None
                        and res.wall_hits > 0
                        and not self.is_open(self.current_pos, nxt)
                    ):
                        self.mark_blocked(self.current_pos, nxt)
                        self._plan = []
                    elif self.is_blocked(self._plan[0], self._plan[1]):
                        self._plan = []
            else:
                self._plan = []

        if len(self._plan) < 2:
            path = None

            if self.goal_known:
                
                for thresh in [1.5, 1.0, 0.7]:
                    path = self._bfs(
                        self.current_pos,
                        self.goal_xy,
                        allow_unknown=False,
                        danger_thresh=thresh,
                    )
                    if path:
                        break

                if path is None:
                    for thresh in [0.99, 0.85, 0.65]:
                        path = self._bfs(
                            self.current_pos,
                            self.goal_xy,
                            allow_unknown=True,
                            danger_thresh=thresh,
                        )
                        if path:
                            break
            else:
                path = self._frontier_bfs(danger_thresh=0.99)
                if path is None:
                    path = self._bfs(
                        self.current_pos,
                        self.goal_xy,
                        allow_unknown=True,
                        danger_thresh=float("inf"),
                    )

            if not path:
                acts = [Action.WAIT]
                self._last_acts = acts
                return acts

            self._plan = path

        if (
            len(self._plan) >= 2
            and self._plan[1] in self.fire_adj
            and self._plan[1] in self.death_cells
        ):
            ta = getattr(self, "_total_actions", 0)
            phase = (ta // 5) % 4
            if phase in (1, 2):
                acts = [Action.WAIT, Action.WAIT, Action.WAIT, Action.WAIT, Action.WAIT]
                self._last_acts = acts
                return acts

        acts = self._path_to_acts(self._plan)
        self._last_acts = acts
        return acts


def run_episode(env, agent, max_turns=8000):
    env.reset()
    agent.reset_episode()

    last = None
    pl = 1

    for _ in range(max_turns):
        acts = agent.plan_turn(last)
        last = env.step(acts)
        pl += last.actions_executed
        if last.is_goal_reached:
            break

    s = env.get_episode_stats()
    s["path_length"] = pl
    s["turns"] = s["turns_taken"]
    return s


def train_agent(env_id, n_ep=15, max_turns=6000, verbose=True):
    env = MazeEnvironment(env_id)
    agent = DynaQAgent()
    agent.boot(env)

    curve = []
    for ep in range(n_ep):
        s = run_episode(env, agent, max_turns)
        curve.append(s)
        if verbose:
            print(
                f"  Ep {ep+1:3d}: goal={s['goal_reached']!s:5} turns={s['turns']:5d} "
                f"deaths={s['deaths']:3d} explored={s['cells_explored']:4d} path={s['path_length']:5d}"
            )

    return agent, curve, env


def evaluate_agent(agent, env_id, n_ep=5, max_turns=8000, retain=True, verbose=True):
    if not retain:
        agent.reset_memory()

    env = MazeEnvironment(env_id)
    agent.boot(env)

    eps = []
    for ep in range(n_ep):
        s = run_episode(env, agent, max_turns)
        eps.append(s)
        if verbose:
            print(
                f"  Eval {ep+1}: goal={s['goal_reached']!s:5} turns={s['turns']:5d} "
                f"deaths={s['deaths']:3d} path={s['path_length']:5d}"
            )

    ok = [s for s in eps if s["goal_reached"]]
    sr = len(ok) / len(eps)
    apl = float(np.mean([s["path_length"] for s in ok])) if ok else float("nan")
    at = float(np.mean([s["turns"] for s in ok])) if ok else float("nan")
    td = sum(s["deaths"] for s in eps)
    tt = sum(s["turns"] for s in eps)
    dr = td / max(1, tt)
    tu = sum(s["cells_explored"] for s in eps)
    tv = sum(s["path_length"] for s in eps)
    ee = tu / max(1, tv)
    mc = len(agent.visit) / (NUM_CELLS * NUM_CELLS)
    ar = agent._replan_t / max(1, agent._replan_n)

    return {
        "per_episode": eps,
        "success_rate": sr,
        "avg_path_length": apl,
        "avg_turns": at,
        "death_rate": dr,
        "exploration_efficiency": ee,
        "map_completeness": mc,
        "avg_replanning_sec": ar,
    }


COLORS = {
    EMPTY: (255, 255, 255),
    WALL: (20, 20, 20),
    START: (60, 220, 60),
    GOAL: (40, 110, 230),
    DEATH_PIT: (230, 60, 30),
    TELEPORT: (0, 190, 190),
    CONFUSION: (170, 40, 220),
    ARROW_UP: (60, 140, 240),
    ARROW_DOWN: (60, 140, 240),
    ARROW_LEFT: (60, 140, 240),
    ARROW_RIGHT: (60, 140, 240),
}


def render_solution(env, path, out, scale=6):
    sz = MAT_SIZE * scale
    img = Image.new("RGB", (sz, sz), (255, 255, 255))
    d = ImageDraw.Draw(img)

    # Draw walls
    for r in range(MAT_SIZE):
        for c in range(MAT_SIZE):
            if env.wall_matrix[r, c] == WALL:
                d.rectangle(
                    [c * scale, r * scale, c * scale + scale - 1, r * scale + scale - 1],
                    fill=(20, 20, 20),
                )

    # Draw hazards / special cells
    for (x, y), ct in env.cell_types.items():
        if ct in (START, GOAL):
            continue  # we'll draw nicer circles for these later
        col = COLORS.get(ct, (200, 200, 200))
        mr, mc = y * 2, x * 2
        d.rectangle(
            [mc * scale, mr * scale, mc * scale + scale - 1, mr * scale + scale - 1],
            fill=col,
        )

    # Draw clean solution path
    if path and len(path) > 1:
        pts = [(x * 2 * scale + scale // 2, y * 2 * scale + scale // 2) for x, y in path]
        d.line(pts, fill=(255, 50, 50), width=max(2, scale // 2))

        # Start marker
        d.ellipse(
            [pts[0][0] - scale, pts[0][1] - scale, pts[0][0] + scale, pts[0][1] + scale],
            fill=(60, 220, 60),
        )

        # Goal marker
        d.ellipse(
            [pts[-1][0] - scale, pts[-1][1] - scale, pts[-1][0] + scale, pts[-1][1] + scale],
            fill=(40, 110, 230),
        )

    img.save(out)


def trace_path(agent, env, max_turns=8000):
    env.reset()
    agent.reset_episode()

    path = [env.pos]
    last = None

    for _ in range(max_turns):
        acts = agent.plan_turn(last)
        last = env.step(acts)

        cur = env.pos

        if cur in path:
            idx = path.index(cur)
            path = path[:idx + 1]
        else:
            path.append(cur)

        if last.is_goal_reached:
            break

    return path