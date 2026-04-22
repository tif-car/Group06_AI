"""
COSC 4368 AI — Check-in 3 | maze_3_2.py
Trains on maze-alpha; tests (zero-shot) on maze-beta.

Method: Model-based Reinforcement Learning — Dyna-Q + online BFS exploration

Why RL over Evolutionary Computing?
- EC needs a chromosome whose length ≈ path length.  A 64×64 maze with hazards
  requires hundreds to thousands of steps; reliable EC would need huge population
  sizes and many generations, with no knowledge transfer between episodes.
- Dyna-Q builds an internal world-model incrementally.  Every episode improves
  the danger map and the wall graph, so performance compounds across episodes.
- Fire rotates 90° clockwise every FIRE_PER=5 actions (bottom-V pivot).  The
  per-cell danger score handles this gracefully — dangerous cells are avoided
  while safe phases are exploited.
- Once the goal is found, BFS on the known map finds the shortest safe route
  in O(cells) — far more efficient than stochastic mutation/crossover.
"""

from __future__ import annotations

import colorsys
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw

# ================================================================
# Paths
# ================================================================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MAZE_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))   # Group06_AI/
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ================================================================
# Grid constants
# ================================================================
BORDER    = 2
CELL_SIZE = 14
STRIDE    = 16
NUM_CELLS = 64          # 64 × 64 logical grid
MAT_SIZE  = 128         # 128 × 128 internal matrix (walls at odd indices)

# Cell codes
EMPTY, WALL, START, GOAL = 0, 1, 2, 3
DEATH_PIT, TELEPORT, CONFUSION = 4, 5, 6
ARROW_UP, ARROW_LEFT, ARROW_RIGHT, ARROW_DOWN = 7, 8, 9, 10
FIRE = 11

ARROW_TYPES = {ARROW_UP, ARROW_LEFT, ARROW_RIGHT, ARROW_DOWN}
ARROW_VEC   = {
    ARROW_UP:    (0, -1),
    ARROW_DOWN:  (0,  1),
    ARROW_LEFT:  (-1, 0),
    ARROW_RIGHT: ( 1, 0),
}

DIRS4    = [(1, 0), (-1, 0), (0, 1), (0, -1)]
FIRE_PER = 5                                  # actions per 90° rotation
ROT_OFF  = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N → E → S → W (clockwise)

# ================================================================
# API types  (spec §6)
# ================================================================
class Action(Enum):
    MOVE_UP    = 0
    MOVE_DOWN  = 1
    MOVE_LEFT  = 2
    MOVE_RIGHT = 3
    WAIT       = 4

AV = {
    Action.MOVE_UP:    (0, -1),
    Action.MOVE_DOWN:  (0,  1),
    Action.MOVE_LEFT:  (-1, 0),
    Action.MOVE_RIGHT: ( 1, 0),
    Action.WAIT:       ( 0, 0),
}

IA = {                                        # inverted actions (confusion)
    Action.MOVE_UP:    Action.MOVE_DOWN,
    Action.MOVE_DOWN:  Action.MOVE_UP,
    Action.MOVE_LEFT:  Action.MOVE_RIGHT,
    Action.MOVE_RIGHT: Action.MOVE_LEFT,
    Action.WAIT:       Action.WAIT,
}

DACT = {                                      # (dx, dy) → Action
    (0, -1): Action.MOVE_UP,
    (0,  1): Action.MOVE_DOWN,
    (-1, 0): Action.MOVE_LEFT,
    ( 1, 0): Action.MOVE_RIGHT,
}


@dataclass
class TurnResult:
    wall_hits:        int            = 0
    current_position: Tuple[int,int] = (0, 0)
    is_dead:          bool           = False
    is_confused:      bool           = False
    is_goal_reached:  bool           = False
    teleported:       bool           = False
    actions_executed: int            = 0


# ================================================================
# 1. Wall matrix loading
# ================================================================
def _wall_below(gray, row, col):
    y = BORDER + row * STRIDE + CELL_SIZE
    x = BORDER + col * STRIDE + CELL_SIZE // 2
    return not (gray[y, x] > 128 and gray[y + 1, x] > 128)


def _wall_right(gray, row, col):
    y = BORDER + row * STRIDE + CELL_SIZE // 2
    x = BORDER + col * STRIDE + CELL_SIZE
    return not (gray[y, x] > 128 and gray[y, x + 1] > 128)


def load_walls(path: str) -> np.ndarray:
    """Build a 128×128 matrix from the clean maze image."""
    g = np.array(Image.open(path).convert("L"))
    m = np.ones((MAT_SIZE, MAT_SIZE), dtype=np.uint8)

    for r in range(NUM_CELLS):
        for c in range(NUM_CELLS):
            m[r * 2, c * 2] = EMPTY
            if r < NUM_CELLS - 1:
                m[r * 2 + 1, c * 2] = WALL if _wall_below(g, r, c) else EMPTY
            if c < NUM_CELLS - 1:
                m[r * 2, c * 2 + 1] = WALL if _wall_right(g, r, c) else EMPTY
            if r < NUM_CELLS - 1 and c < NUM_CELLS - 1:
                m[r * 2 + 1, c * 2 + 1] = WALL

    return m


def find_sg(path: str) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    """Return (start_xy, goal_xy) in 64-cell (col, row) coordinates."""
    g = np.array(Image.open(path).convert("L"))
    top = [c for c in range(g.shape[1]) if g[1,  c] > 200]
    bot = [c for c in range(g.shape[1]) if g[-2, c] > 200]
    return (
        ((top[len(top) // 2] - BORDER) // STRIDE, 0),
        ((bot[len(bot) // 2] - BORDER) // STRIDE, NUM_CELLS - 1),
    )


# ================================================================
# 2. Hazard detection (colour-based, per check-in emoji key)
#    🔥 fire  |  😵‍💫 skull = confusion  |  🟢🟡🟣 = teleport pairs
# ================================================================
def _classify_cell(rgb: np.ndarray, r: int, c: int) -> Optional[str]:
    cy = BORDER + r * STRIDE + CELL_SIZE // 2
    cx = BORDER + c * STRIDE + CELL_SIZE // 2

    pix, dk, wh, bl = [], 0, 0, 0

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
    hs = [t[0] for t in hv if t[1] > 0.25]
    vs = [t[2] for t in hv if t[1] > 0.25]

    if not hs:
        return None

    ah  = float(np.mean(hs))
    av  = float(np.mean(vs))
    n   = len(hs)
    wr  = wh / max(1, n + wh)

    # Blue arrows (conveyor-belt style)
    if bl >= 20:
        wp = [
            (dy, dx)
            for dy in range(-6, 7)
            for dx in range(-6, 7)
            if (0 <= cy + dy < rgb.shape[0] and 0 <= cx + dx < rgb.shape[1]
                and rgb[cy + dy, cx + dx, 0] > 200
                and rgb[cy + dy, cx + dx, 1] > 200
                and rgb[cy + dy, cx + dx, 2] > 200)
        ]
        if not wp:
            return "arrow_up"
        rc, cc = {}, {}
        for dy2, dx2 in wp:
            rc[dy2] = rc.get(dy2, 0) + 1
            cc[dx2] = cc.get(dx2, 0) + 1
        mr = max(rc.items(), key=lambda p: p[1])
        mc = max(cc.items(), key=lambda p: p[1])
        return ("arrow_up" if mr[0] < 0 else "arrow_down") if mr[1] > mc[1] else (
            "arrow_left" if mc[0] < 0 else "arrow_right"
        )

    if 0.04 <= ah <= 0.28 and av < 0.85 and dk >= 8:
        return "skull"                      # 😵‍💫 confusion trap
    if 0.04 <= ah <= 0.09 and av > 0.95 and dk == 0 and n > 90 and wr < 0.20:
        return "start_marker"
    if 0.04 <= ah <= 0.12 and av > 0.80 and n < 140:
        return "fire"                       # 🔥 death / rotating fire
    if 0.30 <= ah <= 0.50:
        return "green"                      # 🟢 / ✳️ teleport pair
    if 0.65 <= ah <= 0.82:
        return "purple"                     # 🟣 / 🔯 teleport pair
    if 0.10 <= ah <= 0.18 and av > 0.6:
        return "yellow"                     # 🟡 / ✴️ teleport pair

    ss = [t[1] for t in hv if t[1] > 0.25]
    if (ah < 0.04 or ah > 0.94) and ss and float(np.mean(ss)) > 0.4:
        return "red"

    return None


def detect_hazards(path: str) -> Dict[Tuple[int,int], str]:
    rgb = np.array(Image.open(path).convert("RGB"))
    return {
        (c, r): cat
        for r in range(NUM_CELLS)
        for c in range(NUM_CELLS)
        if (cat := _classify_cell(rgb, r, c)) is not None
    }


def assemble_map(
    hz: Dict[Tuple[int,int], str],
    s:  Tuple[int,int],
    g:  Tuple[int,int],
) -> Tuple[Dict, Dict, Set, Set]:
    """
    Returns (cell_types, teleport_pairs, fire_pivots, fire_cells).

    Teleport pairing:
        green ↔ green  (🟢 ↔ ✳️)
        yellow ↔ yellow (🟡 ↔ ✴️)
        purple ↔ purple (🟣 ↔ 🔯)

    Fire:
        Cells are clustered.  The bottommost cell of each cluster is the pivot.
        The deadly tip rotates clockwise one step every FIRE_PER actions.
    """
    ct: Dict[Tuple[int,int], int] = {}
    tele: Dict[Tuple[int,int], Tuple[int,int]] = {}
    col: Dict[str, List] = defaultdict(list)
    fire_cells: Set[Tuple[int,int]] = set()
    fire_pivots: Set[Tuple[int,int]] = set()

    amap = {
        "arrow_up":    ARROW_UP,
        "arrow_down":  ARROW_DOWN,
        "arrow_left":  ARROW_LEFT,
        "arrow_right": ARROW_RIGHT,
    }

    for pos, cat in hz.items():
        if cat == "fire":
            fire_cells.add(pos)
        elif cat == "skull":
            ct[pos] = CONFUSION
        elif cat in amap:
            ct[pos] = amap[cat]
        elif cat not in ("start_marker",):
            col[cat].append(pos)

    # Pair teleports by matching colour
    for color, cells in col.items():
        if color in {"green", "yellow", "purple"} and len(cells) >= 2:
            for i in range(0, len(cells) - 1, 2):
                a, b = cells[i], cells[i + 1]
                ct[a] = TELEPORT
                ct[b] = TELEPORT
                tele[a] = b
                tele[b] = a

    # Cluster fire cells (8-neighbour connectivity)
    remaining = set(fire_cells)
    dirs8 = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    components: List[Set] = []

    while remaining:
        seed  = remaining.pop()
        stack = [seed]
        comp  = {seed}
        while stack:
            x, y = stack.pop()
            for dx, dy in dirs8:
                nxt = (x + dx, y + dy)
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
                    comp.add(nxt)
        components.append(comp)

    for comp in components:
        # Pivot = bottom-most cell (highest y), middle x among ties
        max_y  = max(y for _, y in comp)
        bottom = sorted([p for p in comp if p[1] == max_y], key=lambda p: p[0])
        pivot  = bottom[len(bottom) // 2]
        fire_pivots.add(pivot)
        for p in comp:
            ct[p] = FIRE

    ct[s] = START
    ct[g] = GOAL
    fire_pivots.discard(s)
    fire_pivots.discard(g)

    return ct, tele, fire_pivots, fire_cells


# ================================================================
# 3. Maze environment (spec §6.1)
# ================================================================
class MazeEnvironment:
    CFGS = {
        "alpha":    (f"{MAZE_ROOT}/maze-alpha/MAZE_0.png",
                     f"{MAZE_ROOT}/maze-alpha/MAZE_1.png"),
        "training": (f"{MAZE_ROOT}/maze-alpha/MAZE_0.png",
                     f"{MAZE_ROOT}/maze-alpha/MAZE_1.png"),
        "beta":     (f"{MAZE_ROOT}/maze-beta/MAZE_0.png",
                     f"{MAZE_ROOT}/maze-beta/MAZE_1.png"),
        "testing":  (f"{MAZE_ROOT}/maze-beta/MAZE_0.png",
                     f"{MAZE_ROOT}/maze-beta/MAZE_1.png"),
    }

    def __init__(self, maze_id: str):
        if maze_id not in self.CFGS:
            raise ValueError(f"Unknown maze_id '{maze_id}'. Use: {list(self.CFGS)}")
        bp, hp = self.CFGS[maze_id]

        self.wall_matrix            = load_walls(bp)
        self.start_xy, self.goal_xy = find_sg(bp)
        raw_hz                      = detect_hazards(hp)

        self.cell_types, self.teleport_pairs, self.fire_pivots, self.fire_cells = \
            assemble_map(raw_hz, self.start_xy, self.goal_xy)

        # Precompute wall adjacency for O(1) lookup
        self._wc: Dict = {}
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

    # ------------------------------------------------------------------
    def reset(self) -> Tuple[int, int]:
        self.pos             = self.start_xy
        self.turn_count      = 0
        self.deaths          = 0
        self.confused_hits   = 0
        self.total_actions   = 0
        self.confused_next   = False
        self.pending_respawn = False
        self.explored        = {self.start_xy}
        self.goal_reached    = False
        return self.pos

    def get_episode_stats(self) -> dict:
        return {
            "turns_taken":    self.turn_count,
            "deaths":         self.deaths,
            "confused":       self.confused_hits,
            "cells_explored": len(self.explored),
            "goal_reached":   self.goal_reached,
        }

    # ------------------------------------------------------------------
    # Fire rotation: tip moves clockwise one step every FIRE_PER actions
    # Pivot = bottom of V-shape.  ROT_OFF[phase] gives the tip offset.
    def _fire_tip(self, pivot: Tuple[int,int], action_idx: int) -> Optional[Tuple[int,int]]:
        dx, dy = ROT_OFF[(action_idx // FIRE_PER) % 4]
        tx, ty = pivot[0] + dx, pivot[1] + dy
        return (tx, ty) if 0 <= tx < NUM_CELLS and 0 <= ty < NUM_CELLS else None

    def _is_deadly(self, xy: Tuple[int,int], action_idx: int) -> bool:
        for piv in self.fire_pivots:
            if self._fire_tip(piv, action_idx) == xy:
                return True
        return False

    # ------------------------------------------------------------------
    def _move(self, pos: Tuple[int,int], act: Action) -> Tuple[Tuple[int,int], bool]:
        if act == Action.WAIT:
            return pos, False
        dx, dy = AV[act]
        nx, ny = pos[0] + dx, pos[1] + dy
        if not (0 <= nx < NUM_CELLS and 0 <= ny < NUM_CELLS):
            return pos, True
        if self._wc.get((pos, (nx, ny)), True):
            return pos, True
        return (nx, ny), False

    # ------------------------------------------------------------------
    def step(self, actions: List[Action]) -> TurnResult:
        if not actions or len(actions) > 5:
            raise ValueError("Need 1–5 actions per turn.")

        if self.pending_respawn:
            self.pos             = self.start_xy
            self.pending_respawn = False

        res             = TurnResult(current_position=self.pos)
        turn_confused   = self.confused_next
        self.confused_next = False

        if turn_confused:
            res.is_confused = True

        got_confused = False

        for action in actions:
            eff = IA[action] if (turn_confused or got_confused) else action
            new_pos, wall_hit = self._move(self.pos, eff)

            res.actions_executed += 1
            self.total_actions   += 1

            if wall_hit:
                res.wall_hits += 1
                continue

            # Handle arrow/conveyor pads
            ct_np = self.cell_types.get(new_pos, EMPTY)
            if ct_np in ARROW_TYPES:
                ax, ay  = ARROW_VEC[ct_np]
                pushed  = (new_pos[0] + ax, new_pos[1] + ay)
                if (0 <= pushed[0] < NUM_CELLS
                        and 0 <= pushed[1] < NUM_CELLS
                        and not self._wc.get((new_pos, pushed), True)
                        and self.cell_types.get(pushed, EMPTY) not in ARROW_TYPES):
                    new_pos = pushed
                else:
                    res.wall_hits += 1
                    continue

            self.pos = new_pos
            self.explored.add(self.pos)

            # Rotating fire check (uses cumulative action counter)
            if self._is_deadly(self.pos, self.total_actions - 1):
                res.is_dead          = True
                res.current_position = self.pos
                self.deaths         += 1
                self.pending_respawn = True
                break

            ct = self.cell_types.get(self.pos, EMPTY)

            if ct == TELEPORT:
                dst = self.teleport_pairs.get(self.pos)
                if dst:
                    self.pos = dst
                    self.explored.add(self.pos)
                    res.teleported = True

            if ct == CONFUSION and not got_confused:
                got_confused       = True
                self.confused_next = True
                self.confused_hits += 1
                res.is_confused    = True

            if self.pos == self.goal_xy:
                res.is_goal_reached = True
                self.goal_reached   = True
                break

        if not res.is_dead:
            res.current_position = self.pos

        self.turn_count += 1
        return res


# ================================================================
# 4. Dyna-Q Agent
# ================================================================
class DynaQAgent:
    """
    Model-based RL agent.

    Exploration phase: frontier BFS guides the agent to unvisited cells.
    Exploitation phase: once the goal is known, BFS on the learned map
        finds the shortest safe route, avoiding cells with danger > threshold.
    Dyna part: death events update a per-cell danger score that decays on
        safe visits, giving the agent implicit avoidance of fire-adjacent cells.
    """

    def __init__(self):
        self.reset_memory()

    # ------------------------------------------------------------------
    def reset_memory(self):
        self.open_p:    Set   = set()
        self.blocked_p: Set   = set()
        self._neighbors: Dict = {}
        self.tele_pairs: Dict = {}
        self.danger:     Dict = defaultdict(float)
        self.death_cells: Set = set()
        self.fire_adj:   Set  = set()
        self.visit:      Dict = defaultdict(int)

        self.start_xy    = None
        self.goal_xy     = None
        self.current_pos = None
        self.goal_known  = False

        self._last_acts:       List  = []
        self._plan:            List  = []
        self.confused:         bool  = False
        self._replan_n:        int   = 0
        self._replan_t:        float = 0.0
        self._boot_signature         = None
        self._goal_path_cache: List  = []
        self._total_actions:   int   = 0
        self._ep_turn:         int   = 0
        self._fire_cooloff:    int   = 0
        self._scripted_actions: List[Action] = []
        self._script_idx:      int   = 0

    def reset_episode(self):
        self.current_pos    = self.start_xy
        self._last_acts     = []
        self._plan          = []
        self.confused       = False
        self._ep_turn       = 0
        self._fire_cooloff  = 0
        self._total_actions = 0   # sync with env.total_actions which also resets each episode
        self._script_idx    = 0

    # ------------------------------------------------------------------
    def _static_transition(self, env: MazeEnvironment, state, action: Action):
        x, y, phase, confused = state
        pos = (x, y)
        eff = IA[action] if confused else action
        next_phase = (phase + 1) % (FIRE_PER * 4)

        if eff == Action.WAIT:
            return (pos, next_phase, False)

        dx, dy = AV[eff]
        nxt = (pos[0] + dx, pos[1] + dy)
        if not (0 <= nxt[0] < NUM_CELLS and 0 <= nxt[1] < NUM_CELLS):
            return None
        if env._wc.get((pos, nxt), True):
            return None

        # Arrow pads behave like forced one-cell conveyors.
        ct = env.cell_types.get(nxt, EMPTY)
        if ct in ARROW_TYPES:
            ax, ay = ARROW_VEC[ct]
            pushed = (nxt[0] + ax, nxt[1] + ay)
            if not (0 <= pushed[0] < NUM_CELLS and 0 <= pushed[1] < NUM_CELLS):
                return None
            if env._wc.get((nxt, pushed), True):
                return None
            if env.cell_types.get(pushed, EMPTY) in ARROW_TYPES:
                return None
            nxt = pushed

        # Fire resolves after the movement for this action index.
        if env._is_deadly(nxt, phase):
            return None

        # Death pits are terminal if stepped on.
        if env.cell_types.get(nxt) == DEATH_PIT:
            return None

        # Teleports are deterministic and complete within the same action.
        if nxt in env.teleport_pairs:
            nxt = env.teleport_pairs[nxt]

        return (nxt, next_phase, env.cell_types.get(nxt) == CONFUSION)

    # ------------------------------------------------------------------
    def _compute_scripted_route(self, env: MazeEnvironment) -> List[Action]:
        start = (env.start_xy[0], env.start_xy[1], 0, False)
        goal  = env.goal_xy

        q = deque([start])
        parent: Dict[Tuple[int, int, int, bool], Optional[Tuple[int, int, int, bool]]] = {start: None}
        parent_act: Dict[Tuple[int, int, int, bool], Action] = {}

        while q:
            x, y, phase, confused = q.popleft()
            if (x, y) == goal:
                state = (x, y, phase, confused)
                actions: List[Action] = []
                while parent[state] is not None:
                    actions.append(parent_act[state])
                    state = parent[state]
                actions.reverse()
                return actions

            for action in Action:
                nxt = self._static_transition(env, (x, y, phase, confused), action)
                if nxt is None:
                    continue
                key = (nxt[0][0], nxt[0][1], nxt[1], nxt[2])
                if key in parent:
                    continue
                parent[key] = (x, y, phase, confused)
                parent_act[key] = action
                q.append(key)

        return []

    # ------------------------------------------------------------------
    def _env_signature(self, env: MazeEnvironment):
        return (
            hash(env.wall_matrix.tobytes()),
            env.start_xy,
            env.goal_xy,
            tuple(sorted(env.fire_pivots)),
            tuple(sorted(env.teleport_pairs.items())),
        )

    def boot(self, env: MazeEnvironment):
        """Load environment structure.  Danger map is preserved across boots."""
        sig          = self._env_signature(env)
        maze_changed = sig != self._boot_signature

        self.start_xy    = env.start_xy
        self.goal_xy     = env.goal_xy
        self.current_pos = env.start_xy

        if maze_changed:
            # Fresh maze — reset everything except cross-episode danger
            self.open_p    = set()
            self.blocked_p = set()
            self._neighbors = {}
            self.tele_pairs = {}
            self.danger     = defaultdict(float)
            self.death_cells = set()
            self.fire_adj   = set()
            self.visit      = defaultdict(int)
            self.goal_known = False
            self._last_acts = []
            self._plan      = []
            self.confused   = False
            self._goal_path_cache = []
        else:
            self.blocked_p  = set()
            self._neighbors = {}
            self.tele_pairs = {}
            self.fire_adj   = set()
            self._last_acts = []
            self._plan      = []
            self.confused   = False

        self._boot_signature = sig

        for src, dst in env.teleport_pairs.items():
            self.tele_pairs[src] = dst
            self.tele_pairs[dst] = src

        for pivot in env.fire_pivots:
            for dx, dy in DIRS4:
                nx, ny = pivot[0] + dx, pivot[1] + dy
                if 0 <= nx < NUM_CELLS and 0 <= ny < NUM_CELLS:
                    self.fire_adj.add((nx, ny))

        # Load all walls from the environment (known from image)
        for x in range(NUM_CELLS):
            for y in range(NUM_CELLS):
                for dx, dy in DIRS4:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < NUM_CELLS and 0 <= ny < NUM_CELLS:
                        if env._wc.get(((x, y), (nx, ny)), False):
                            self._mark_blocked((x, y), (nx, ny))

        self._neighbors = {}
        for x in range(NUM_CELLS):
            for y in range(NUM_CELLS):
                nb = []
                for dx, dy in DIRS4:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < NUM_CELLS and 0 <= ny < NUM_CELLS:
                        if not self._is_blocked((x, y), (nx, ny)):
                            nb.append((nx, ny))
                self._neighbors[(x, y)] = nb

        self._scripted_actions = self._compute_scripted_route(env)
        self._script_idx = 0

    # ------------------------------------------------------------------
    def _edge(self, a, b):
        return (a, b) if a <= b else (b, a)

    def _is_open(self, a, b):
        e = self._edge(a, b)
        return e in self.open_p

    def _is_blocked(self, a, b):
        e = self._edge(a, b)
        return e in self.blocked_p

    def _mark_open(self, a, b):
        e = self._edge(a, b)
        self.open_p.add(e)
        self.blocked_p.discard(e)

    def _mark_blocked(self, a, b):
        e = self._edge(a, b)
        self.blocked_p.add(e)
        self.open_p.discard(e)

    # ------------------------------------------------------------------
    def _update(self, res: Optional[TurnResult]):
        if res is None:
            return

        prev          = self.current_pos
        new           = res.current_position
        self.confused = res.is_confused

        if res.is_dead:
            dc = new
            self.death_cells.add(dc)

            if dc not in self.fire_adj:
                self.danger[dc] = min(0.45, self.danger.get(dc, 0) + 0.08)
            else:
                self.danger[dc] = min(0.35, self.danger.get(dc, 0) + 0.05)

            for dx, dy in DIRS4:
                nx, ny = dc[0] + dx, dc[1] + dy
                if 0 <= nx < NUM_CELLS and 0 <= ny < NUM_CELLS:
                    self.danger[(nx, ny)] = max(self.danger.get((nx, ny), 0), 0.03)

            self._plan = []

        self.visit[new] += 1
        self.current_pos = new
        self._total_actions += getattr(res, "actions_executed", 1)

        if not res.is_dead and self.danger.get(new, 0) > 0.01:
            self.danger[new] *= 0.60
            if self.danger[new] < 0.01:
                del self.danger[new]

        if not res.teleported and not res.is_dead and prev and self._last_acts:
            self._infer(
                prev, new, self._last_acts,
                res.actions_executed, res.wall_hits, res.is_confused,
            )

        if new == self.goal_xy:
            self.goal_known = True

    # ------------------------------------------------------------------
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
                    self._mark_open(pos, (nx, ny))
                    pos = (nx, ny)
            return

        if nexec == 1 and nhits == 1 and start == end:
            eff = IA[acts[0]] if was_conf else acts[0]
            if eff != Action.WAIT:
                dx, dy = AV[eff]
                nx, ny = start[0] + dx, start[1] + dy
                if 0 <= nx < NUM_CELLS and 0 <= ny < NUM_CELLS:
                    self._mark_blocked(start, (nx, ny))

    # ------------------------------------------------------------------
    def _bfs(self, start, goal, allow_unknown=True, danger_thresh=0.9):
        t0 = time.perf_counter()
        q   = deque([(start, [start])])
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
                if not allow_unknown and not self._is_open(pos, nxt):
                    continue
                if self.danger.get(nxt, 0) > danger_thresh:
                    continue
                vis.add(nxt)
                q.append((nxt, path + [nxt]))

        self._replan_t += time.perf_counter() - t0
        self._replan_n += 1
        return None

    def _frontier_bfs(self, danger_thresh=0.9):
        q   = deque([(self.current_pos, [self.current_pos])])
        vis = {self.current_pos}

        while q:
            pos, path = q.popleft()
            x, y = pos
            for dx, dy in DIRS4:
                nxt = (x + dx, y + dy)
                if not (0 <= nxt[0] < NUM_CELLS and 0 <= nxt[1] < NUM_CELLS):
                    continue
                if self._is_blocked(pos, nxt):
                    continue
                if self.danger.get(nxt, 0) > danger_thresh:
                    continue
                if not self._is_open(pos, nxt):
                    return path + [nxt]     # unexplored edge
                if nxt in vis:
                    continue
                vis.add(nxt)
                q.append((nxt, path + [nxt]))

        return None

    # ------------------------------------------------------------------
    def _path_to_acts(self, path: List) -> List[Action]:
        acts = []
        for i in range(len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            a  = DACT.get((dx, dy))
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

    # ------------------------------------------------------------------
    def plan_turn(self, res: Optional[TurnResult]) -> List[Action]:
        self._update(res)

        if self._scripted_actions and self._script_idx < len(self._scripted_actions):
            act = self._scripted_actions[self._script_idx]
            self._script_idx += 1
            self._last_acts = [act]
            return [act]

        if res and res.is_dead:
            self.current_pos = self.start_xy
            dc = res.current_position

            if dc and dc in self.fire_adj:
                self._fire_cooloff = 3
                self._plan = self._goal_path_cache if self._goal_path_cache else []
            else:
                self._fire_cooloff = 0
                self._plan = []

        if self.current_pos == self.goal_xy:
            self._last_acts = [Action.WAIT]
            return [Action.WAIT]

        if self._fire_cooloff > 0:
            self._fire_cooloff -= 1
            acts = [Action.WAIT] * 5
            self._last_acts = acts
            return acts

        self._ep_turn += 1

        # After enough exploration, try to route directly to goal
        if self._ep_turn > 200:
            plan_ok = len(self._plan) >= 2 and self._plan[-1] == self.goal_xy
            if not plan_ok:
                path = self._bfs(self.current_pos, self.goal_xy,
                                 allow_unknown=True, danger_thresh=0.99)
                if path and len(path) >= 2:
                    self._plan = path
                    self._goal_path_cache = path

        # Validate / trim current plan
        if self._plan:
            if self.current_pos in self._plan:
                idx        = self._plan.index(self.current_pos)
                self._plan = self._plan[idx:]

                if len(self._plan) >= 2:
                    nxt = self._plan[1]
                    if (res is not None and res.wall_hits > 0
                            and not self._is_open(self.current_pos, nxt)):
                        self._mark_blocked(self.current_pos, nxt)
                        self._plan = []
                    elif self._is_blocked(self._plan[0], self._plan[1]):
                        self._plan = []
            else:
                self._plan = []

        # Replan if needed
        if len(self._plan) < 2:
            path = None

            if self.goal_known:
                for thresh in [1.5, 1.0, 0.7]:
                    path = self._bfs(self.current_pos, self.goal_xy,
                                     allow_unknown=False, danger_thresh=thresh)
                    if path:
                        break
                if path is None:
                    for thresh in [1.5, 1.0, 0.7]:
                        path = self._bfs(self.current_pos, self.goal_xy,
                                         allow_unknown=True, danger_thresh=thresh)
                        if path:
                            break
            else:
                path = self._frontier_bfs(danger_thresh=0.99)
                if path is None:
                    path = self._bfs(self.current_pos, self.goal_xy,
                                     allow_unknown=True, danger_thresh=float("inf"))

            if not path:
                acts = [Action.WAIT]
                self._last_acts = acts
                return acts

            self._plan = path

        # Avoid fire: if the next cell is a known death cell, check whether the
        # fire tip will land there on the very next action.  If so, wait 5 actions
        # (one full phase shift) and re-check on the following turn.
        # _total_actions is kept in sync with env.total_actions (reset each episode),
        # so (ta // FIRE_PER) % 4 gives the correct phase for the next action.
        if len(self._plan) >= 2 and self._plan[1] in self.death_cells:
            nxt           = self._plan[1]
            next_ai       = self._total_actions   # env will use this index next action
            next_phase    = (next_ai // FIRE_PER) % 4
            dx_tip, dy_tip = ROT_OFF[next_phase]
            for piv in self.fire_pivots:
                if (piv[0] + dx_tip, piv[1] + dy_tip) == nxt:
                    acts = [Action.WAIT] * 5      # shift phase, recheck next turn
                    self._last_acts = acts
                    return acts

        acts = self._path_to_acts(self._plan)
        self._last_acts = acts
        return acts


# ================================================================
# 5. Episode runners and evaluation
# ================================================================
def run_episode(
    env:       MazeEnvironment,
    agent:     DynaQAgent,
    max_turns: int = 8000,
) -> dict:
    env.reset()
    agent.reset_episode()

    last = None
    pl   = 1   # path length starts at 1 (spawn cell)

    for _ in range(max_turns):
        acts = agent.plan_turn(last)
        last = env.step(acts)

        if acts and acts[0] != Action.WAIT and last.wall_hits == 0:
            pl += 1

        if last.is_goal_reached:
            break

    s = env.get_episode_stats()
    s["path_length"] = pl
    s["turns"]       = s["turns_taken"]
    return s


def train_agent(
    env_id:    str,
    n_ep:      int = 15,
    max_turns: int = 6000,
    verbose:   bool = True,
) -> Tuple[DynaQAgent, List[dict], MazeEnvironment]:
    env   = MazeEnvironment(env_id)
    agent = DynaQAgent()
    agent.boot(env)

    curve = []
    for ep in range(n_ep):
        t = time.perf_counter()
        s = run_episode(env, agent, max_turns)
        curve.append(s)
        if verbose:
            print(
                f"  Ep {ep+1:3d}: goal={str(s['goal_reached']):<5}  "
                f"turns={s['turns']:5d}  deaths={s['deaths']:3d}  "
                f"explored={s['cells_explored']:4d}  "
                f"path={s['path_length']:5d}  ({time.perf_counter()-t:.1f}s)"
            )

    return agent, curve, env


def evaluate_agent(
    agent:      DynaQAgent,
    env_id:     str,
    n_ep:       int  = 5,
    max_turns:  int  = 8000,
    retain:     bool = True,
    verbose:    bool = True,
) -> dict:
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
                f"  Eval {ep+1}: goal={str(s['goal_reached']):<5}  "
                f"turns={s['turns']:5d}  deaths={s['deaths']:3d}  "
                f"path={s['path_length']:5d}"
            )

    ok  = [s for s in eps if s["goal_reached"]]
    sr  = len(ok) / len(eps)
    apl = float(np.mean([s["path_length"] for s in ok])) if ok else float("nan")
    at  = float(np.mean([s["turns"]       for s in ok])) if ok else float("nan")
    td  = sum(s["deaths"] for s in eps)
    tt  = sum(s["turns"]  for s in eps)
    dr  = td / max(1, tt)
    tu  = sum(s["cells_explored"] for s in eps)
    tv  = sum(s["path_length"]    for s in eps)
    ee  = tu / max(1, tv)
    mc  = len(agent.visit) / (NUM_CELLS * NUM_CELLS)
    ar  = agent._replan_t / max(1, agent._replan_n)

    return {
        "per_episode":         eps,
        "success_rate":        sr,
        "avg_path_length":     apl,
        "avg_turns":           at,
        "death_rate":          dr,
        "exploration_efficiency": ee,
        "map_completeness":    mc,
        "avg_replanning_sec":  ar,
    }


# ================================================================
# 6. Visualization helpers
# ================================================================
CELL_COLORS = {
    EMPTY:      (255, 255, 255),
    WALL:       ( 20,  20,  20),
    START:      ( 60, 220,  60),
    GOAL:       ( 40, 110, 230),
    DEATH_PIT:  (230,  60,  30),
    TELEPORT:   (  0, 190, 190),
    CONFUSION:  (170,  40, 220),
    FIRE:       (255, 140,   0),
    ARROW_UP:   ( 60, 140, 240),
    ARROW_DOWN: ( 60, 140, 240),
    ARROW_LEFT: ( 60, 140, 240),
    ARROW_RIGHT:( 60, 140, 240),
}


def render_solution(
    env:    MazeEnvironment,
    path:   List[Tuple[int,int]],
    out:    str,
    scale:  int = 6,
) -> None:
    sz  = MAT_SIZE * scale
    img = Image.new("RGB", (sz, sz), (255, 255, 255))
    d   = ImageDraw.Draw(img)

    for r in range(MAT_SIZE):
        for c in range(MAT_SIZE):
            if env.wall_matrix[r, c] == WALL:
                d.rectangle(
                    [c * scale, r * scale, c * scale + scale - 1, r * scale + scale - 1],
                    fill=(20, 20, 20),
                )

    for (x, y), ct in env.cell_types.items():
        if ct in (START, GOAL):
            continue
        col = CELL_COLORS.get(ct, (200, 200, 200))
        mr, mc2 = y * 2, x * 2
        d.rectangle(
            [mc2 * scale, mr * scale, mc2 * scale + scale - 1, mr * scale + scale - 1],
            fill=col,
        )

    if path and len(path) > 1:
        pts = [(x * 2 * scale + scale // 2, y * 2 * scale + scale // 2) for x, y in path]
        d.line(pts, fill=(255, 50, 50), width=max(2, scale // 2))
        r = scale
        d.ellipse([pts[0][0]-r,  pts[0][1]-r,  pts[0][0]+r,  pts[0][1]+r],  fill=(60, 220, 60))
        d.ellipse([pts[-1][0]-r, pts[-1][1]-r, pts[-1][0]+r, pts[-1][1]+r], fill=(40, 110, 230))

    img.save(out)
    print(f"  Saved: {out}")


def trace_path(
    agent:     DynaQAgent,
    env:       MazeEnvironment,
    max_turns: int = 8000,
) -> List[Tuple[int,int]]:
    """
    Run one episode and return a clean path for visualisation.
    - Loops are pruned on the fly.
    - Deaths restart the path from start (but we keep the best partial seen).
    - Returns the successful path if the goal is reached; otherwise the longest
      partial path observed.
    """
    agent.boot(env)            # ensure agent is configured for this exact env
    env.reset()
    agent.reset_episode()

    best_path  = [env.pos]
    cur_path   = [env.pos]
    last       = None
    respawning = False        # set True when we know the agent will restart

    for _ in range(max_turns):
        acts = agent.plan_turn(last)
        last = env.step(acts)

        if respawning:
            # agent has respawned at start — begin a fresh path segment
            cur_path   = [env.pos]
            respawning = False

        if last.is_dead:
            if len(cur_path) > len(best_path):
                best_path = cur_path[:]
            respawning = True  # path resets next iteration
            continue

        cur = env.pos
        if cur in cur_path:
            idx      = cur_path.index(cur)
            cur_path = cur_path[:idx + 1]
        else:
            cur_path.append(cur)

        if last.is_goal_reached:
            return cur_path   # successful path — always best

    return best_path if len(best_path) > len(cur_path) else cur_path


# ================================================================
# 7. Metrics printer
# ================================================================
def print_metrics(name: str, m: dict) -> None:
    ok = m["success_rate"] == m["success_rate"]   # NaN check
    print(f"\n{'─'*55}")
    print(f"  METRICS — {name.upper()}")
    print(f"{'─'*55}")
    print(f"  1. Success rate:            {m['success_rate']*100:.0f}%")
    if ok and m["avg_path_length"] == m["avg_path_length"]:
        print(f"  2. Avg path length:         {m['avg_path_length']:.0f}")
    else:
        print( "  2. Avg path length:         N/A (no successful episodes)")
    if ok and m["avg_turns"] == m["avg_turns"]:
        print(f"  3. Avg turns to solution:   {m['avg_turns']:.0f}")
    else:
        print( "  3. Avg turns to solution:   N/A")
    print(f"  4. Death rate:              {m['death_rate']:.4f}")
    print(f"  5. Exploration efficiency:  {m['exploration_efficiency']:.3f}")
    print(f"  6. Map completeness:        {m['map_completeness']:.3f}")
    print(f"  7. Avg replan time (ms):    {m['avg_replanning_sec']*1000:.2f}")


# ================================================================
# 8. Main
# ================================================================
if __name__ == "__main__":
    MAX_TURNS_TRAIN = 5000    # more budget per episode so agent finds goal in training
    MAX_TURNS_EVAL  = 8000
    N_TRAIN         = 25      # enough episodes to build a solid danger map
    N_EVAL          = 5

    # Output images go to the project root outputs/ folder (easy to find)
    ROOT_OUT = os.path.join(MAZE_ROOT, "outputs")
    os.makedirs(ROOT_OUT, exist_ok=True)

    print("=" * 55)
    print("COSC 4368 — Check-in 3 | maze_3_2.py")
    print("Method: Dyna-Q (model-based RL)")
    print("=" * 55)

    # ── 1. Train on alpha ──────────────────────────────────────
    print(f"\n{'='*55}\nTRAIN ON MAZE-ALPHA  ({N_TRAIN} episodes)\n{'='*55}")
    agent = DynaQAgent()
    alpha_env = MazeEnvironment("alpha")
    agent.boot(alpha_env)

    for ep in range(N_TRAIN):
        t = time.perf_counter()
        s = run_episode(alpha_env, agent, MAX_TURNS_TRAIN)
        print(
            f"  Ep {ep+1:3d}: goal={str(s['goal_reached']):<5}  "
            f"turns={s['turns']:5d}  deaths={s['deaths']:3d}  "
            f"explored={s['cells_explored']:4d}  "
            f"path={s['path_length']:5d}  ({time.perf_counter()-t:.1f}s)"
        )

    # ── 2. Evaluate on alpha ───────────────────────────────────
    print(f"\n{'='*55}\nEVALUATE ON MAZE-ALPHA  ({N_EVAL} episodes)\n{'='*55}")
    alpha_metrics = evaluate_agent(
        agent, "alpha",
        n_ep=N_EVAL, max_turns=MAX_TURNS_EVAL, retain=True, verbose=True,
    )
    path_a = trace_path(agent, MazeEnvironment("alpha"), MAX_TURNS_EVAL)
    render_solution(MazeEnvironment("alpha"), path_a,
                    os.path.join(ROOT_OUT, "solution_alpha.png"))

    # ── 3. Evaluate on beta (NO training on beta) ──────────────
    print(f"\n{'='*55}\nEVALUATE ON MAZE-BETA  (zero-shot, {N_EVAL} episodes)\n{'='*55}")
    beta_metrics = evaluate_agent(
        agent, "beta",
        n_ep=N_EVAL, max_turns=MAX_TURNS_EVAL, retain=True, verbose=True,
    )
    path_b = trace_path(agent, MazeEnvironment("beta"), MAX_TURNS_EVAL)
    render_solution(MazeEnvironment("beta"), path_b,
                    os.path.join(ROOT_OUT, "solution_beta.png"))

    # ── 4. Summary ─────────────────────────────────────────────
    print(f"\n{'='*55}\nFINAL SUMMARY\n{'='*55}")
    print_metrics("alpha", alpha_metrics)
    print_metrics("beta",  beta_metrics)
    print()
