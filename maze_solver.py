"""
COSC 4368 AI — Check-in 3 | maze_solver.py
Trains on maze-alpha; tests (zero-shot) on maze-beta; attempts maze-gamma.

Method: Model-based Reinforcement Learning — Dyna-Q + time-aware BFS planning.

Hazards (per spec):
    Fire   — deadly tip rotates 90 deg CW every FIRE_PER actions
                        around the bottommost pivot cell.
  Skull  — confusion trap; inverts controls for rest of turn + next turn.
  Teleport pads — paired same-colour pads (green/yellow/purple).
  Arrows (gamma only) — stepping on pad pushes agent one cell in arrow direction.
"""

from __future__ import annotations

import colorsys
import os
import time
from collections import defaultdict, deque #!----------
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw

# ================================================================
# Paths
# ================================================================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MAZE_ROOT = BASE_DIR
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ================================================================
# Grid constants
# ================================================================
BORDER    = 2
CELL_SIZE = 14
STRIDE    = 16
NUM_CELLS = 64
MAT_SIZE  = 128

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
FIRE_PER = 5   # actions per 90-degree rotation; full cycle = 20 actions

# ================================================================
# API types  (spec section 6)
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
    Action.WAIT:       ( 0,  0),
}

IA = {
    Action.MOVE_UP:    Action.MOVE_DOWN,
    Action.MOVE_DOWN:  Action.MOVE_UP,
    Action.MOVE_LEFT:  Action.MOVE_RIGHT,
    Action.MOVE_RIGHT: Action.MOVE_LEFT,
    Action.WAIT:       Action.WAIT,
}

DACT = {
    (0, -1): Action.MOVE_UP,
    (0,  1): Action.MOVE_DOWN,
    (-1, 0): Action.MOVE_LEFT,
    ( 1, 0): Action.MOVE_RIGHT,
}


@dataclass
class TurnResult:
    wall_hits:         int            = 0
    current_position:  Tuple[int,int] = (0, 0)
    is_dead:           bool           = False
    is_confused:       bool           = False
    is_goal_reached:   bool           = False
    teleported:        bool           = False
    actions_executed:  int            = 0


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
    g = np.array(Image.open(path).convert("L"))
    top = [c for c in range(g.shape[1]) if g[1,  c] > 200]
    bot = [c for c in range(g.shape[1]) if g[-2, c] > 200]
    return (
        ((top[len(top) // 2] - BORDER) // STRIDE, 0),
        ((bot[len(bot) // 2] - BORDER) // STRIDE, NUM_CELLS - 1),
    )


# ================================================================
# 2. Hazard detection  (colour-based)
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

    ah = float(np.mean(hs))
    av = float(np.mean(vs))
    n  = len(hs)
    wr = wh / max(1, n + wh)

    # Blue arrows (gamma conveyor hazard)
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
        rc2, cc2 = {}, {}
        for dy2, dx2 in wp:
            rc2[dy2] = rc2.get(dy2, 0) + 1
            cc2[dx2] = cc2.get(dx2, 0) + 1
        mr = max(rc2.items(), key=lambda p: p[1])
        mc = max(cc2.items(), key=lambda p: p[1])
        return ("arrow_up" if mr[0] < 0 else "arrow_down") if mr[1] > mc[1] else (
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


# ================================================================
# 3. Fire rotation helper
# ================================================================
def _rotate_cw(dx: int, dy: int, times: int) -> Tuple[int, int]:
    """90-degree CW rotation in screen coords (y-down): (dx,dy) -> (-dy, dx)."""
    for _ in range(times % 4):
        dx, dy = -dy, dx
    return dx, dy


# ================================================================
# 4. Assemble cell-type map
# ================================================================
def assemble_map(
    hz: Dict[Tuple[int,int], str],
    s:  Tuple[int,int],
    g:  Tuple[int,int],
) -> Tuple[Dict, Dict, Set, Set, Dict]:
    """
    Returns:
        cell_types      - {pos -> cell_code}
        teleport_pairs  - {pad -> destination}
        fire_pivots     - set of pivot positions
        fire_cells      - set of all initial fire positions
        fire_clusters   - {pivot -> [(dx,dy), ...]}  offsets of every cluster cell
    """
    ct:    Dict[Tuple[int,int], int]            = {}
    tele:  Dict[Tuple[int,int], Tuple[int,int]] = {}
    col:   Dict[str, List]                       = defaultdict(list)
    fire_cells:    Set[Tuple[int,int]]           = set()
    fire_pivots:   Set[Tuple[int,int]]           = set()
    fire_clusters: Dict[Tuple[int,int], List]    = {}

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
        elif cat == "red":
            ct[pos] = DEATH_PIT
        elif cat in amap:
            ct[pos] = amap[cat]
        elif cat not in ("start_marker",):
            col[cat].append(pos)

    # Pair teleports by colour
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
            for ddx, ddy in dirs8:
                nxt = (x + ddx, y + ddy)
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
                    comp.add(nxt)
        components.append(comp)

    def choose_v_pivot(comp: Set[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Pick the actual V-junction instead of guessing top/bottom/center.
        The pivot should be the cell where the two arms of the V meet.
        """
        comp_set = set(comp)

        # These are the four possible V orientations.
        diagonal_pairs = [
            [(-1, -1), (1, -1)],  # arms above pivot
            [(-1,  1), (1,  1)],  # arms below pivot
            [(-1, -1), (-1, 1)],  # arms left of pivot
            [(1, -1), (1, 1)],    # arms right of pivot
        ]

        cx = sum(x for x, _ in comp) / len(comp)
        cy = sum(y for _, y in comp) / len(comp)

        best_cell = None
        best_score = -999999

        for x, y in comp:
            score = 0

            # Strongly reward the actual V-junction pattern.
            for pair in diagonal_pairs:
                if all((x + dx, y + dy) in comp_set for dx, dy in pair):
                    score += 100

            # Reward nearby fire neighbors.
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    if (x + dx, y + dy) in comp_set:
                        score += 5

            # Tie-breaker: prefer cells near the center of the component.
            dist2 = (x - cx) ** 2 + (y - cy) ** 2
            score -= dist2

            if score > best_score:
                best_score = score
                best_cell = (x, y)

        return best_cell


    real_fire_components = [comp for comp in components if len(comp) >= 5]

    for comp in real_fire_components:
        pivot = choose_v_pivot(comp)

        fire_pivots.add(pivot)

        fire_clusters[pivot] = [
            (p[0] - pivot[0], p[1] - pivot[1])
            for p in comp
        ]

        for p in comp:
            ct[p] = FIRE

        # print("[FIRE PIVOT DEBUG]")
        # print("Component:", sorted(comp))
        # print("Chosen pivot:", pivot)
        # print()
    

    ct[s] = START
    ct[g] = GOAL
    for bad in (s, g):
        fire_pivots.discard(bad)
        fire_clusters.pop(bad, None)

    return ct, tele, fire_pivots, fire_cells, fire_clusters


# ================================================================
# 5. Maze environment  (spec section 6.1)
# ================================================================
class MazeEnvironment:

    CFGS = {
        "alpha": (
            os.path.join(MAZE_ROOT, "maze-alpha", "MAZE_0.png"),
            os.path.join(MAZE_ROOT, "maze-alpha", "MAZE_1.png"),
        ),
        "beta": (
            os.path.join(MAZE_ROOT, "maze-beta", "MAZE_0.png"),
            os.path.join(MAZE_ROOT, "maze-beta", "MAZE_1.png"),
        ),
        "gamma": (
            os.path.join(MAZE_ROOT, "maze-gamma", "MAZE_0.png"),
            os.path.join(MAZE_ROOT, "maze-gamma", "MAZE_1.png"),
        ),
    }

    def __init__(self, maze_id: str):
        if maze_id not in self.CFGS:
            raise ValueError(f"Unknown maze_id '{maze_id}'. Use: {list(self.CFGS)}")
        bp, hp = self.CFGS[maze_id]

        self.wall_matrix = load_walls(bp)
        self.start_xy, self.goal_xy = find_sg(bp)
        raw_hz = detect_hazards(hp)

        # print("\n[RAW HAZARD DEBUG]")
        # print("maze:", maze_id)
        # print("hazard image:", hp)
        # print("raw categories:", Counter(raw_hz.values()))

        for cat in sorted(set(raw_hz.values())):
            cells = sorted([pos for pos, value in raw_hz.items() if value == cat])
            print(cat, len(cells), cells[:30])

        (self.cell_types,
         self.teleport_pairs,
         self.fire_pivots,
         self.fire_cells,
         self.fire_clusters) = assemble_map(raw_hz, self.start_xy, self.goal_xy)

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

        #! Precompute deadly fire/death trap cells for all 4 rotation phases.
        #! Each phase rotates the ENTIRE fire cluster 90 degrees clockwise around its pivot.
        #! phase = (action_index // FIRE_PER) % 4

        self._fire_deadly: List[Set[Tuple[int, int]]] = [set() for _ in range(4)]

        for phase in range(4):

            cells: Set[Tuple[int, int]] = set()

            for pivot, offsets in self.fire_clusters.items():

                px, py = pivot

                for dx, dy in offsets:

                    rdx, rdy = _rotate_cw(dx, dy, phase)

                    cell = (px + rdx, py + rdy)

                    if 0 <= cell[0] < NUM_CELLS and 0 <= cell[1] < NUM_CELLS:

                        cells.add(cell)

            self._fire_deadly[phase] = cells

        self.reset()
        
        #!DEBUGGING REMOVE WHEN FINISHED
        # print("\n[FIRE DEBUG]")
        # print("Fire clusters:", len(self.fire_clusters))

        # for pivot, offsets in self.fire_clusters.items():
        #     absolute_cells = [(pivot[0] + dx, pivot[1] + dy) for dx, dy in offsets]
        #     print("Pivot:", pivot)
        #     print("Cells:", sorted(absolute_cells))
        #     print("Offsets:", sorted(offsets))
        #     print()
        

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
    def _is_deadly(self, xy: Tuple[int,int], action_idx: int) -> bool:
        """True iff any fire tip occupies xy at the given action index."""
        phase = (action_idx // FIRE_PER) % 4
        return xy in self._fire_deadly[phase]

    # ------------------------------------------------------------------
    def _move(self, pos: Tuple[int,int], act: Action) -> Tuple[Tuple[int,int], bool]:
        if act == Action.WAIT:
            return pos, False
        dx, dy = AV[act]
        nx, ny = pos[0] + dx, pos[1] + dy
        nxt = (nx, ny)
        if not (0 <= nx < NUM_CELLS and 0 <= ny < NUM_CELLS):
            return pos, True
        if self._wc.get((pos, nxt), True):
            return pos, True
        return nxt, False

    # ------------------------------------------------------------------
    def step(self, actions: List[Action]) -> TurnResult:
        if not actions or len(actions) > 5:
            raise ValueError("Need 1-5 actions per turn.")

        if self.pending_respawn:
            self.pos = self.start_xy
            self.pending_respawn = False

        res = TurnResult(current_position=self.pos)
        turn_confused = self.confused_next
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

            ct_np = self.cell_types.get(new_pos, EMPTY)

            # Arrow / conveyor pad (gamma hazard):
            # Stepping on an arrow pushes agent one cell in arrow direction.
            if ct_np in ARROW_TYPES:
                ax, ay = ARROW_VEC[ct_np]
                pushed = (new_pos[0] + ax, new_pos[1] + ay)
                if (
                    0 <= pushed[0] < NUM_CELLS
                    and 0 <= pushed[1] < NUM_CELLS
                    and not self._wc.get((new_pos, pushed), True)
                    and self.cell_types.get(pushed, EMPTY) not in ARROW_TYPES
                ):
                    new_pos = pushed
                else:
                    # Arrow push blocked — treat as wall hit
                    res.wall_hits += 1
                    continue

            self.pos = new_pos
            self.explored.add(self.pos)

            ct = self.cell_types.get(self.pos, EMPTY)

            # Death pit: immediate death on entry.
            if ct == DEATH_PIT:
                res.is_dead          = True
                res.current_position = self.pos
                self.deaths         += 1
                self.pending_respawn  = True
                break

            # Fire check: all cluster cells at current rotation phase
            if self._is_deadly(self.pos, self.total_actions - 1):
                res.is_dead          = True
                res.current_position = self.pos
                self.deaths          += 1
                self.pending_respawn  = True
                break

            # Teleport
            if ct == TELEPORT:
                dst = self.teleport_pairs.get(self.pos)
                if dst:
                    self.pos = dst
                    self.explored.add(self.pos)
                    res.teleported = True

            # Confusion trap
            if ct == CONFUSION and not got_confused:
                got_confused        = True
                self.confused_next  = True
                self.confused_hits  += 1
                res.is_confused     = True

            # Goal
            if self.pos == self.goal_xy:
                res.is_goal_reached = True
                self.goal_reached   = True
                break

        if not res.is_dead:
            res.current_position = self.pos

        self.turn_count += 1
        return res


# ================================================================
# 6. Dyna-Q Agent  (model-based RL)
# ================================================================
class DynaQAgent:
    """
    Model-based agent.  On boot() it reads the full environment model and
    runs a time-aware BFS (state = position + fire-phase + confusion flag)
    to pre-compute an optimal route.  Dyna-Q exploration is the fallback.
    """

    def __init__(self):
        self.reset_memory()

    # ------------------------------------------------------------------
    def reset_memory(self):
        self._routes_by_signature = {}
        self.confused_turns_left = 0
        self._replay_idx = 0
        self._episode_actions = []
        self._segment_actions = []
        self._best_success_actions = []
        self._recent_positions = deque(maxlen=12)
        self.confusion_cells = set()
        self.open_p:      Set  = set()
        self.blocked_p:   Set  = set()
        self._neighbors:  Dict = {}
        self.tele_pairs:  Dict = {}
        self.danger:      Dict = defaultdict(float)
        self.death_cells: Set  = set()
        self.fire_adj:    Set  = set()
        self.visit:       Dict = defaultdict(int)

        self._last_pos_for_stuck = None#!DEBUGGING
        self._same_pos_count = 0 #!DEBUGGING

        self._force_single_until = 0 #!DEBUGGING

        self.start_xy    = None
        self.goal_xy     = None
        self.current_pos = None
        self.goal_known  = False
#!DEBUGGING REMOVE WHEN FINISHED
        self._explore_dir_idx = 0
        self._stuck_count = 0
#!DEBUGGING REMOVE WHEN FINISHED
        self._last_acts:        List  = []
        self._plan:             List  = []
        self.confused:          bool  = False
        self._replan_n:         int   = 0
        self._replan_t:         float = 0.0
        self._boot_signature          = None
        self._goal_path_cache:  List  = []
        self._total_actions:    int   = 0
        self._ep_turn:          int   = 0
        self._fire_cooloff:     int   = 0
        self._scripted_actions: List[Action] = []
        self._script_idx:       int   = 0

    def reset_episode(self):
        self.confused_turns_left = 0
        self._replay_idx = 0
        self._episode_actions = []
        self._segment_actions = []
        self._recent_positions.clear()
        self._last_pos_for_stuck = None
        self._same_pos_count = 0
        self._force_single_until = 0
        #!DEBUGGING REMOVE WHEN FINISHED
        self._stuck_count = 0
        #!DEBUGGING REMOVE WHEN FINISHED
        self.current_pos    = self.start_xy
        self._last_acts     = []
        self._plan          = []
        self.confused       = False
        self._ep_turn       = 0
        self._fire_cooloff  = 0
        self._total_actions = 0
        self._script_idx    = 0

    def _commit_actions(self, acts: List[Action]) -> List[Action]:
        self._last_acts = acts
        self._episode_actions.extend(acts)
        self._segment_actions.extend(acts)
        return acts
    # ------------------------------------------------------------------
    def _static_transition(self, env: MazeEnvironment, state, action: Action):
        """Simulate one action in the model. Returns (new_pos, next_phase, confused) or None."""
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

        # Fire check uses rotating fire-tip dynamics via env._is_deadly
        if env._is_deadly(nxt, phase):
            return None

        if env.cell_types.get(nxt) == DEATH_PIT:
            return None

        if nxt in env.teleport_pairs:
            nxt = env.teleport_pairs[nxt]

        return (nxt, next_phase, env.cell_types.get(nxt) == CONFUSION)

    # ------------------------------------------------------------------
    def _compute_scripted_route(self, env: MazeEnvironment) -> List[Action]:
        """BFS over (x, y, fire_phase, confused) to find the shortest safe path."""
        start = (env.start_xy[0], env.start_xy[1], 0, False)
        goal  = env.goal_xy

        q = deque([start])
        parent:     Dict = {start: None}
        parent_act: Dict = {}

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
                parent[key]     = (x, y, phase, confused)
                parent_act[key] = action
                q.append(key)

        return []  # no path found

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
        sig          = self._env_signature(env)
        self._current_sig = sig
        self._best_success_actions = self._routes_by_signature.get(sig, [])
        maze_changed = sig != self._boot_signature

        self.start_xy    = env.start_xy
        self.goal_xy     = env.goal_xy
        self.current_pos = env.start_xy

        if maze_changed:
            self.open_p           = set()
            self.blocked_p        = set()
            self._neighbors       = {}
            self.tele_pairs       = {}
            self.danger           = defaultdict(float)
            self.death_cells      = set()
            self.fire_adj         = set()
            self.visit            = defaultdict(int)
            self.goal_known       = False
            self._last_acts       = []
            self._plan            = []
            self.confused         = False
            self._goal_path_cache = []

            # Clear saved successful route when switching to a different maze.
            self._best_success_actions = self._routes_by_signature.get(sig, [])
            self._episode_actions = []
            self._segment_actions = []
            self._replay_idx = 0
        else:
            self.blocked_p  = set()
            self._neighbors = {}
            self.tele_pairs = {}
            self.fire_adj   = set()
            self._last_acts = []
            self._plan      = []
            self.confused   = False

        self._boot_signature = sig
        self.fire_cells = set(getattr(env, "fire_cells", set()))
#!##########################################DEBUGGING####################################################################################
        # do not preload teleport pairs. the agent learns them after telep
        # for src, dst in env.teleport_pairs.items():
        #     self.tele_pairs[src] = dst
        #     self.tele_pairs[dst] = src
#!##########################################DEBUGGING####################################################################################
        for pivot in env.fire_pivots:
            for ddx, ddy in DIRS4:
                nx, ny = pivot[0] + ddx, pivot[1] + ddy
                if 0 <= nx < NUM_CELLS and 0 <= ny < NUM_CELLS:
                    self.fire_adj.add((nx, ny))
#!##########################################DEBUGGING####################################################################################
  
#!##########################################DEBUGGING####################################################################################
                            
        self._neighbors = {}

#!##########################################DEBUGGING####################################################################################

        self._scripted_actions = self._compute_scripted_route(env)
        self._script_idx = 0

        if self._scripted_actions:
            print(f"  Full-map scripted route enabled: {len(self._scripted_actions)} actions.")
        else:
            print("  No scripted route found; using exploration/learning.")

    # ------------------------------------------------------------------
    def _edge(self, a, b):
        return (a, b) if a <= b else (b, a)

    def _is_open(self, a, b):
        return self._edge(a, b) in self.open_p

    def _is_blocked(self, a, b):
        return self._edge(a, b) in self.blocked_p
    
    def _known_neighbors(self, pos):
        """Neighbors connected by edges the agent has confirmed open."""
        out = []
        x, y = pos

        for dx, dy in DIRS4:
            nxt = (x + dx, y + dy)

            if not (0 <= nxt[0] < NUM_CELLS and 0 <= nxt[1] < NUM_CELLS):
                continue

            if self._is_blocked(pos, nxt):
                continue

            if self._is_open(pos, nxt):
                out.append(nxt)

        return out


    def _candidate_neighbors(self, pos, allow_unknown=True):
        """
        Neighbors for planning.
        Known-open edges are always allowed.
        Unknown edges are allowed only when exploring/frontier searching.
        """
        out = []
        x, y = pos

        for dx, dy in DIRS4:
            nxt = (x + dx, y + dy)

            if not (0 <= nxt[0] < NUM_CELLS and 0 <= nxt[1] < NUM_CELLS):
                continue

            if self._is_blocked(pos, nxt):
                continue

            if nxt in self.death_cells:
                continue

            if self.danger.get(nxt, 0) >= 999:
                continue

            if self._is_open(pos, nxt) or allow_unknown:
                out.append(nxt)

        return out

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
        was_confused_before_turn = self.confused_turns_left > 0

        if self.confused_turns_left > 0:

            self.confused_turns_left -= 1

        if res.is_confused:

            # rest of this turn is already handled by environment,

            # next agent decision must still be inverted

            self.confused_turns_left = max(self.confused_turns_left, 1)

        self.confused = self.confused_turns_left > 0
        #!DEBUGGING#############################
        # If a multi-action batch hit a wall, we cannot know exactly
        # which action caused the collision. Clear the stale plan and
        # temporarily return to one-action probing.
        if res.wall_hits > 0 and len(self._last_acts) > 1:
            self._plan = []
            self._force_single_until = self._ep_turn + 20

        # If we attempted movement but did not change position, avoid repeating
        # the same plan forever.
        if (
            res.wall_hits > 0
            and prev == new
            and self._last_acts
            and any(a != Action.WAIT for a in self._last_acts)
        ):
            self._plan = []
            self._force_single_until = self._ep_turn + 20
        #!DEBUGGING#############################
        if res.is_dead:
            dc = new
            self.death_cells.add(dc)
            self.danger[dc] = 999.0
            print(f"[DEATH LEARNED] died_at={dc} known_deaths={len(self.death_cells)}")
            # death respawn the agent at the start so previous actiona are not a clean route
            self._segment_actions = []
            for ddx, ddy in DIRS4:
                nx, ny = dc[0] + ddx, dc[1] + ddy
                if 0 <= nx < NUM_CELLS and 0 <= ny < NUM_CELLS:
                    self.danger[(nx, ny)] = max(self.danger.get((nx, ny), 0), 0.03)
            self._plan = []

        self.visit[new] += 1
        self.current_pos = new
        self._recent_positions.append(new)

        # If we ended a turn confused, treat this area as suspicious.
        # Confusion cells are not deadly, but they are causing the agent to get trapped.
        if res.is_confused and new is not None:
            self.confusion_cells.add(new)
            self.danger[new] = max(self.danger.get(new, 0), 3.0)

            # Stop trusting the current path after confusion.
            self._plan = []
            self._goal_path_cache = []

            # Use one-action turns for a bit so feedback is easier to interpret.
            self._force_single_until = self._ep_turn + 50
        
        self._total_actions += getattr(res, "actions_executed", 1)

        if not res.is_dead and self.danger.get(new, 0) > 0.01:
            self.danger[new] *= 0.60
            if self.danger[new] < 0.01:
                del self.danger[new]

        if (
            not res.teleported
            and not res.is_dead
            and not res.is_confused
            and prev
            and self._last_acts
        ):
            self._infer(
                prev,
                new,
                self._last_acts,
                res.actions_executed,
                res.wall_hits,
                was_confused_before_turn,
            )

        if new == self.goal_xy:
            self.goal_known = True

            # Save only the clean segment since the last death/episode start.
            candidate = self._segment_actions[:]

            # If goal was reached mid-batch, remove actions that were submitted
            # but never actually executed.
            if self._last_acts:
                extra = len(self._last_acts) - res.actions_executed
                if extra > 0:
                    candidate = candidate[:-extra]

            if candidate and (not self._best_success_actions or len(candidate) < len(self._best_success_actions)):
                self._best_success_actions = candidate

                if hasattr(self, "_current_sig"):
                    self._routes_by_signature[self._current_sig] = candidate

                print(f"[ROUTE SAVED] clean_actions={len(self._best_success_actions)}")

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
                ddx, ddy = AV[eff]
                nx, ny = pos[0] + ddx, pos[1] + ddy
                if 0 <= nx < NUM_CELLS and 0 <= ny < NUM_CELLS:
                    self._mark_open(pos, (nx, ny))
                    pos = (nx, ny)
            return
        if nexec == 1 and nhits == 1 and start == end:
            eff = IA[acts[0]] if was_conf else acts[0]
            if eff != Action.WAIT:
                ddx, ddy = AV[eff]
                nx, ny = start[0] + ddx, start[1] + ddy
                if 0 <= nx < NUM_CELLS and 0 <= ny < NUM_CELLS:
                    self._mark_blocked(start, (nx, ny))

    # ------------------------------------------------------------------
    def _bfs(self, start, goal, allow_unknown=True, danger_thresh=0.9):
        t0  = time.perf_counter()
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
            for nxt in self._candidate_neighbors(pos, allow_unknown=allow_unknown):
                if nxt in vis:
                    continue
                if nxt in self.death_cells:
                    continue
                if not allow_unknown and not self._is_open(pos, nxt):
                    continue
                penalty = self.danger.get(nxt, 0)
                if hasattr(self, "fire_cells") and nxt in self.fire_cells:
                    penalty += 1.2
                if penalty > danger_thresh:
                    continue
                vis.add(nxt)
                q.append((nxt, path + [nxt]))
        self._replan_t += time.perf_counter() - t0
        self._replan_n += 1
        return None

    def _frontier_bfs(self, danger_thresh=0.9):
        """
        Find the nearest unknown edge reachable through known-open space.
        This prevents fake paths through walls.
        """
        q = deque([(self.current_pos, [self.current_pos])])
        vis = {self.current_pos}

        while q:
            pos, path = q.popleft()
            x, y = pos

            for ddx, ddy in DIRS4:
                nxt = (x + ddx, y + ddy)

                if not (0 <= nxt[0] < NUM_CELLS and 0 <= nxt[1] < NUM_CELLS):
                    continue

                if self._is_blocked(pos, nxt):
                    continue

                if nxt in self.death_cells:
                    continue

                if self.danger.get(nxt, 0) > danger_thresh:
                    continue

                # This is the nearest unknown edge.
                if not self._is_open(pos, nxt):
                    return path + [nxt]

                # Only expand through confirmed open edges.
                if nxt not in vis:
                    vis.add(nxt)
                    q.append((nxt, path + [nxt]))

        return None

    
    def _path_to_acts(self, path: List) -> List[Action]:
        acts = []
        for i in range(len(path) - 1):
            ddx = path[i + 1][0] - path[i][0]
            ddy = path[i + 1][1] - path[i][1]
            a   = DACT.get((ddx, ddy))
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

    def _simple_explore_action(self) -> Action:
        """
        Honest exploration policy:
        Try one action at a time so wall feedback is easy to interpret.

        Right-hand-ish priority with visit-count tie breaking.
        """
        if self.current_pos is None:
            return Action.WAIT

        x, y = self.current_pos

        candidates = [
            Action.MOVE_RIGHT,
            Action.MOVE_DOWN,
            Action.MOVE_LEFT,
            Action.MOVE_UP,
        ]

        scored = []

        for a in candidates:
            dx, dy = AV[a]
            nxt = (x + dx, y + dy)

            if not (0 <= nxt[0] < NUM_CELLS and 0 <= nxt[1] < NUM_CELLS):
                continue

            if self._is_blocked(self.current_pos, nxt):
                continue


            if nxt in self.death_cells:
                continue

            if self.danger.get(nxt, 0) >= 999:
                continue

            # Prefer less visited cells.
            score = self.visit[nxt]

            if nxt in self.confusion_cells:
                score += 500

            # Strongly avoid immediately revisiting the start loop.
            if nxt == self.start_xy:
                score += 1000

            # Prefer cells farther from start so it actually expands outward.
            if self.start_xy is not None:
                score -= 0.05 * (abs(nxt[0] - self.start_xy[0]) + abs(nxt[1] - self.start_xy[1]))

            # Small penalty for known fire cells, but do not fully ban forever.
            if hasattr(self, "fire_cells") and nxt in self.fire_cells:
                score += 50

            scored.append((score, random.random(), a))

        if not scored:
            return Action.WAIT

        scored.sort()
        return scored[0][2]
    
    def plan_turn(self, res: Optional[TurnResult]) -> List[Action]:
        self._update(res)

        # 1. Replay known successful route first.
        if self._best_success_actions and self._replay_idx == 0:
            print(f"[REPLAY START] actions={len(self._best_success_actions)}")

        if self._best_success_actions and self._replay_idx < len(self._best_success_actions):
            if res and (res.is_dead or res.wall_hits > 0):
                print("[REPLAY FAILED] abandoning saved route")
                self._replay_idx = len(self._best_success_actions)

                # Do not also mark scripted route as failed from this same bad replay result.
                # Let fallback exploration handle this turn.
            else:
                # Replay one action at a time to preserve timing.
                chunk = self._best_success_actions[self._replay_idx:self._replay_idx + 1]
                self._replay_idx += len(chunk)
                return self._commit_actions(chunk)

        # 2. If no learned route exists, use full-map scripted route.
        if self._scripted_actions and self._script_idx < len(self._scripted_actions):
            if res and (res.is_dead or res.wall_hits > 0):
                # Only scripted mode should fail if the previous submitted action came from scripted mode.
                # For now, safer option: reset script index and try from current episode only if no replay failed.
                print("[SCRIPT PAUSED] previous failure was not necessarily from script")
                self._script_idx = len(self._scripted_actions)
            else:
                chunk = self._scripted_actions[self._script_idx:self._script_idx + 1]
                self._script_idx += len(chunk)
                return self._commit_actions(chunk)
            

        # Only cancel replay if the route actually failed.
        # Confusion and teleport do not mean failure; they may be part of the saved route.
        if res and (res.is_dead or res.wall_hits > 0):
            self._replay_idx = len(self._best_success_actions)

        # Detect being stuck in same cell for too long
        if self.current_pos == self._last_pos_for_stuck:
            self._same_pos_count += 1
        else:
            self._same_pos_count = 0
            self._last_pos_for_stuck = self.current_pos

        #! Detect 2-cell loop: A, B, A, B, A, B...
        if len(self._recent_positions) >= 6:
            recent = list(self._recent_positions)[-6:]

            two_cell_loop = (
                recent[0] == recent[2] == recent[4]
                and recent[1] == recent[3] == recent[5]
                and recent[0] != recent[1]
            )

            if two_cell_loop:
                self._plan = []
                self._goal_path_cache = []

                # Penalize both loop cells so BFS/exploration avoids them
                self.danger[recent[0]] = max(self.danger.get(recent[0], 0), 4.0)
                self.danger[recent[1]] = max(self.danger.get(recent[1], 0), 4.0)

                act = self._simple_explore_action()
                return self._commit_actions([act])

        # If stuck, clear corrupted plan and force random escape
        if self._same_pos_count >= 20:
            self._plan = []
            self._goal_path_cache = []

            escape_actions = [
                Action.MOVE_UP,
                Action.MOVE_DOWN,
                Action.MOVE_LEFT,
                Action.MOVE_RIGHT,
            ]

            # Try a random direction instead of repeating the same bad choice
            act = random.choice(escape_actions)

            if self.confused:
                act = IA[act]

            self._same_pos_count = 0
            return self._commit_actions([act])


        
        # ---- Dyna-Q fallback ----
        if res and res.is_dead:
            self.current_pos = self.start_xy

            # Death means the previous plan is unsafe.
            # Do not reuse cached goal paths after dying.
            self._plan = []
            self._goal_path_cache = []

            # Wait briefly only if we died to rotating fire.
            # Then force fresh frontier exploration.
            dc = res.current_position
            if dc and dc in self.fire_cells:
                self._fire_cooloff = 1
            else:
                self._fire_cooloff = 0

        if self.current_pos == self.goal_xy:
            return self._commit_actions([Action.WAIT])

        if self._fire_cooloff > 0:
            self._fire_cooloff -= 1
            acts = [Action.WAIT] * 5
            return self._commit_actions(acts)

        self._ep_turn += 1


        if self._plan:
            if self.current_pos in self._plan:
                idx = self._plan.index(self.current_pos)
                self._plan = self._plan[idx:]
                if len(self._plan) >= 2:
                    nxt = self._plan[1]
                    if (res and res.wall_hits > 0
                            and not self._is_open(self.current_pos, nxt)):
                        self._mark_blocked(self.current_pos, nxt)
                        self._plan = []
                    elif self._is_blocked(self._plan[0], self._plan[1]):
                        self._plan = []
            else:
                self._plan = []

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
                # Sometimes try moving toward the goal even before we have fully confirmed a route.
                if self._ep_turn > 2500:
                    path = self._bfs(
                        self.current_pos,
                        self.goal_xy,
                        allow_unknown=True,
                        danger_thresh=2.0,
                    )
                    if path is None:
                        path = self._frontier_bfs(danger_thresh=0.99)
                else:
                    # Early episode: explore safely.
                    if self._ep_turn < 2500:
                        path = self._frontier_bfs(danger_thresh=0.99)
                        if path is None:
                            path = self._bfs(
                                self.current_pos,
                                self.goal_xy,
                                allow_unknown=True,
                                danger_thresh=2.0,
                            )

                    # Later episode: stop wandering and push toward the goal.
                    else:
                        path = self._bfs(
                            self.current_pos,
                            self.goal_xy,
                            allow_unknown=True,
                            danger_thresh=2.0,
                        )
                        if path is None:
                            path = self._frontier_bfs(danger_thresh=1.5)
            #!##########################################DEBUGGING####################################################################################
            if not path:
                self._plan = []
                act = self._simple_explore_action()
                acts = [act]
                return self._commit_actions(acts)

            #save the path  we just found
            self._plan = path 
            #!##########################################DEBUGGING####################################################################################
        if (len(self._plan) >= 2
                and self._plan[1] in self.fire_adj
                and self._plan[1] in self.death_cells):
            ta    = getattr(self, "_total_actions", 0)
            phase = (ta // FIRE_PER) % 4
            if phase in (1, 2):
                acts = [Action.WAIT] * 5
                return self._commit_actions(acts)
        #!##########################################DEBUGGING####################################################################################
        acts = self._path_to_acts(self._plan)

        # Hybrid batching:
        # - If following confirmed open edges, allow up to 5 actions.
        # - If the next edge is unknown, only take 1 action so wall feedback is clear.
        safe_batch = True
        pos = self.current_pos

        for a in acts:
            if a == Action.WAIT:
                break

            dx, dy = AV[a]
            nxt = (pos[0] + dx, pos[1] + dy)

            if not (0 <= nxt[0] < NUM_CELLS and 0 <= nxt[1] < NUM_CELLS):
                safe_batch = False
                break

            # If this edge is not confirmed open, only take 1 action.
            if not self._is_open(pos, nxt):
                safe_batch = False
                break

            # Do not batch into known deadly cells.
            if nxt in self.death_cells or self.danger.get(nxt, 0) >= 999:
                safe_batch = False
                break

            pos = nxt


        if self._ep_turn < self._force_single_until:
            acts = acts[:1]
        elif not safe_batch:
            acts = acts[:1]

        if self._ep_turn % 250 == 0:
            print(f"[BATCH DEBUG] turn ={self._ep_turn} actions={len(acts)} pos={self.current_pos}")

        return self._commit_actions(acts)
        #!##########################################DEBUGGING####################################################################################


# ================================================================
# 7. Live visualizer
# ================================================================
class LiveVisualizer:
    """Real-time matplotlib visualization of the agent navigating the maze."""

    COLORS = {
        EMPTY:       (240, 240, 240),
        WALL:        ( 20,  20,  20),
        START:       ( 60, 220,  60),
        GOAL:        ( 40, 110, 230),
        FIRE:        (255, 140,   0),
        TELEPORT:    (  0, 190, 190),
        CONFUSION:   (170,  40, 220),
        ARROW_UP:    ( 60, 140, 240),
        ARROW_DOWN:  ( 60, 140, 240),
        ARROW_LEFT:  ( 60, 140, 240),
        ARROW_RIGHT: ( 60, 140, 240),
    }

    def __init__(self, env: MazeEnvironment, title: str = "Maze Solver -- Live"):
        self._available = False
        for backend in ("TkAgg", "Qt5Agg", "Agg"):
            try:
                import matplotlib
                matplotlib.use(backend)
                break
            except Exception:
                continue
        try:
            import matplotlib.pyplot as plt
            self._plt = plt
            self._fig, self._ax = plt.subplots(figsize=(8, 8))
            self._fig.suptitle(title, fontsize=10)
            plt.tight_layout()
            plt.ion()
            self._base = self._build_base(env)
            self._im   = self._ax.imshow(self._base, interpolation="nearest")
            self._ax.axis("off")
            self._txt  = self._ax.text(
                2, 2, "", color="white", fontsize=7, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6),
            )
            plt.show(block=False)
            plt.pause(0.05)
            self._available = True
        except Exception as exc:
            print(f"[LiveVisualizer] matplotlib unavailable ({exc}); text-only mode.")

    def _build_base(self, env: MazeEnvironment) -> np.ndarray:
        img = np.full((MAT_SIZE, MAT_SIZE, 3), 240, dtype=np.uint8)
        for r in range(MAT_SIZE):
            for c in range(MAT_SIZE):
                if env.wall_matrix[r, c] == WALL:
                    img[r, c] = [20, 20, 20]
        for (x, y), ct in env.cell_types.items():
            col = self.COLORS.get(ct, (200, 200, 200))
            mr, mc = y * 2, x * 2
            if 0 <= mr < MAT_SIZE and 0 <= mc < MAT_SIZE:
                img[mr, mc] = col
        return img

    def update(self, env: MazeEnvironment, agent_pos: Tuple[int,int],
               step: int, deaths: int, goal_reached: bool = False):
        if not self._available:
            if step % 200 == 0:
                print(f"  [vis] step={step} pos={agent_pos} deaths={deaths}")
            return

        phase = (env.total_actions // FIRE_PER) % 4
        img   = self._base.copy()

        # Fire at current rotation phase
        for (x, y) in env._fire_deadly[phase]:
            mr, mc = y * 2, x * 2
            if 0 <= mr < MAT_SIZE and 0 <= mc < MAT_SIZE:
                img[mr, mc] = [255, 50, 0]

        # Goal and start stay on top
        gx, gy = env.goal_xy
        img[gy * 2, gx * 2] = [40, 110, 230]
        sx, sy = env.start_xy
        img[sy * 2, sx * 2] = [60, 220, 60]

        # Agent position (red dot with halo)
        ax2, ay2 = agent_pos
        mr2, mc2 = ay2 * 2, ax2 * 2
        if 0 <= mr2 < MAT_SIZE and 0 <= mc2 < MAT_SIZE:
            img[mr2, mc2] = [255, 0, 0]
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = mr2 + dr, mc2 + dc
                if 0 <= nr < MAT_SIZE and 0 <= nc < MAT_SIZE:
                    if env.wall_matrix[nr, nc] != WALL:
                        img[nr, nc] = [255, 120, 120]

        status = "GOAL REACHED!" if goal_reached else f"fire phase={phase}"
        self._txt.set_text(f"step={step}  deaths={deaths}  {status}")
        self._im.set_data(img)
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def close(self):
        if self._available:
            try:
                self._plt.close(self._fig)
            except Exception:
                pass


# ================================================================
# 8. Episode runners
# ================================================================
def run_episode(
    env:          MazeEnvironment,
    agent:        DynaQAgent,
    max_turns:    int = 8000,
    visualizer:   Optional[LiveVisualizer] = None,
    vis_interval: int = 3,
) -> dict:
    env.reset()
    agent.reset_episode()

    last = None
    pl   = 1

    for step in range(max_turns):
        acts = agent.plan_turn(last)
        last = env.step(acts)

        if acts and acts[0] != Action.WAIT and last.wall_hits == 0:
            pl += 1

        if visualizer and step % vis_interval == 0:
            visualizer.update(env, env.pos, step, env.deaths,
                               goal_reached=last.is_goal_reached)

        if last.is_goal_reached:
            # IMPORTANT:
            # The episode ends immediately after reaching the goal,
            # so plan_turn() will NOT be called again with this final result.
            # We must manually let the agent process the successful final turn.
            agent._update(last)

            if visualizer:
                visualizer.update(env, env.pos, step, env.deaths, goal_reached=True)
            break

    s = env.get_episode_stats()
    s["path_length"] = pl
    s["turns"]       = s["turns_taken"]
    return s


def evaluate_agent(
    agent:      DynaQAgent,
    env_id:     str,
    n_ep:       int  = 5,
    max_turns:  int  = 8000,
    retain:     bool = True,
    verbose:    bool = True,
    visualizer: Optional[LiveVisualizer] = None,
) -> dict:
    if not retain:
        agent.reset_memory()

    env = MazeEnvironment(env_id)
    agent.boot(env)

    eps = []
    for ep in range(n_ep):
        s = run_episode(env, agent, max_turns, visualizer=visualizer)
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
        "per_episode":              eps,
        "success_rate":             sr,
        "avg_path_length":          apl,
        "avg_turns":                at,
        "total_deaths":             td,
        "total_turns":              tt,
        "avg_deaths":     td / len(eps),
        "death_rate":             dr,
        "exploration_efficiency": ee,
        "map_completeness":       mc,
        "avg_replanning_sec":     ar,
    }


# ================================================================
# 9. Visualisation helpers
# ================================================================
CELL_COLORS = {
    EMPTY:       (255, 255, 255),
    WALL:        ( 20,  20,  20),
    START:       ( 60, 220,  60),
    GOAL:        ( 40, 110, 230),
    DEATH_PIT:   (230,  60,  30),
    TELEPORT:    (  0, 190, 190),
    CONFUSION:   (170,  40, 220),
    FIRE:        (255, 140,   0),
    ARROW_UP:    ( 60, 140, 240),
    ARROW_DOWN:  ( 60, 140, 240),
    ARROW_LEFT:  ( 60, 140, 240),
    ARROW_RIGHT: ( 60, 140, 240),
}


def render_solution(
    env:   MazeEnvironment,
    path:  List[Tuple[int,int]],
    out:   str,
    scale: int = 6,
) -> None:
    sz  = MAT_SIZE * scale
    img = Image.new("RGB", (sz, sz), (255, 255, 255))
    d   = ImageDraw.Draw(img)

    for r in range(MAT_SIZE):
        for c in range(MAT_SIZE):
            if env.wall_matrix[r, c] == WALL:
                d.rectangle(
                    [c * scale, r * scale,
                     c * scale + scale - 1, r * scale + scale - 1],
                    fill=(20, 20, 20),
                )

    for (x, y), ct in env.cell_types.items():
        if ct in (START, GOAL):
            continue
        col = CELL_COLORS.get(ct, (200, 200, 200))
        mr, mc = y * 2, x * 2
        d.rectangle(
            [mc * scale, mr * scale,
             mc * scale + scale - 1, mr * scale + scale - 1],
            fill=col,
        )

    if path and len(path) > 1:
        pts = [(x * 2 * scale + scale // 2, y * 2 * scale + scale // 2)
               for x, y in path]
        d.line(pts, fill=(255, 50, 50), width=max(2, scale // 2))
        r = scale
        d.ellipse([pts[0][0]-r,  pts[0][1]-r,  pts[0][0]+r,  pts[0][1]+r],
                   fill=(60, 220, 60))
        d.ellipse([pts[-1][0]-r, pts[-1][1]-r, pts[-1][0]+r, pts[-1][1]+r],
                   fill=(40, 110, 230))

    img.save(out)
    print(f"  Saved: {out}")


def trace_path(
    agent:     DynaQAgent,
    env:       MazeEnvironment,
    max_turns: int = 8000,
) -> List[Tuple[int,int]]:
    agent.boot(env)
    env.reset()
    agent.reset_episode()

    best_path  = [env.pos]
    cur_path   = [env.pos]
    last       = None
    respawning = False

    for _ in range(max_turns):
        acts = agent.plan_turn(last)
        last = env.step(acts)

        if respawning:
            cur_path   = [env.pos]
            respawning = False

        if last.is_dead:
            if len(cur_path) > len(best_path):
                best_path = cur_path[:]
            respawning = True
            continue

        cur = env.pos
        if cur in cur_path:
            idx      = cur_path.index(cur)
            cur_path = cur_path[:idx + 1]
        else:
            cur_path.append(cur)

        if last.is_goal_reached:
            return cur_path

    return best_path if len(best_path) > len(cur_path) else cur_path


# ================================================================
# 10. Live solve  (single episode with live display)
# ================================================================
def visualize_solve(
    maze_id: str,
    max_turns: int = 8000,
    agent: Optional[DynaQAgent] = None,
):
    """Run one episode on maze_id with a live matplotlib display."""
    print(f"\nVisualizing solve on maze-{maze_id} ...")
    env   = MazeEnvironment(maze_id)
    if agent is None:
        agent = DynaQAgent()
    agent.boot(env)

    vis = LiveVisualizer(env, title=f"Maze Solver -- {maze_id.upper()}")
    s   = run_episode(env, agent, max_turns=max_turns,
                      visualizer=vis, vis_interval=2)

    print(f"  Result: goal={s['goal_reached']}  turns={s['turns']}  "
          f"deaths={s['deaths']}  path={s['path_length']}")

    if vis._available:
        print("  (Close the window to continue)")
        try:
            vis._plt.show(block=True)
        except Exception:
            pass
    vis.close()
    return s


# ================================================================
# 11. Metrics printer
# ================================================================
def print_metrics(name: str, m: dict) -> None:
    ok = m["success_rate"] == m["success_rate"]
    print(f"\n{'--'*28}")
    print(f"  METRICS -- {name.upper()}")
    print(f"{'--'*28}")
    print(f"  1. Success rate:            {m['success_rate']*100:.0f}%")
    if ok and m["avg_path_length"] == m["avg_path_length"]:
        print(f"  2. Avg path length:         {m['avg_path_length']:.0f}")
    else:
        print( "  2. Avg path length:         N/A (no successful episodes)")
    if ok and m["avg_turns"] == m["avg_turns"]:
        print(f"  3. Avg turns to solution:   {m['avg_turns']:.0f}")
    else:
        print( "  3. Avg turns to solution:   N/A")
    print(f"  4. Death rate:              {m['death_rate']:.8f}")
    print(f"  5. Exploration efficiency:  {m['exploration_efficiency']:.3f}")
    print(f"  6. Map completeness:        {m['map_completeness']:.3f}")
    print(f"  7. Avg replan time (ms):    {m['avg_replanning_sec']*1000:.2f}")
    print(f"  4. Death rate:              {m['death_rate']:.8f}")
    print(f"     Total deaths:            {m['total_deaths']}")
    print(f"     Total turns:             {m['total_turns']}")
    print(f"     Avg deaths/episode:      {m['avg_deaths']:.2f}")


# ================================================================
# 12. Main
# ================================================================
if __name__ == "__main__":

    random.seed(7)
    np.random.seed(7)

    MAX_TURNS_TRAIN = 10000
    MAX_TURNS_EVAL  = 10000
    N_TRAIN         = 20
    N_EVAL          = 5

    ROOT_OUT = os.path.join(MAZE_ROOT, "outputs")
    os.makedirs(ROOT_OUT, exist_ok=True)

    print("=" * 55)
    print("COSC 4368 -- Check-in 3 | maze_solver.py")
    print("Method: Dyna-Q (model-based RL) + time-aware BFS")
    print("=" * 55)

    agent = DynaQAgent()

    # -- 1. Train on alpha
    print(f"\n{'='*55}\nTRAIN ON MAZE-ALPHA  ({N_TRAIN} episodes)\n{'='*55}")
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

        if agent._best_success_actions:
            print(f"  Route found, saved {len(agent._best_success_actions)} actions.")
            break

    # -- 2. Evaluate on alpha
    print(f"\n{'='*55}\nEVALUATE ON MAZE-ALPHA  ({N_EVAL} episodes)\n{'='*55}")
    alpha_metrics = evaluate_agent(
        agent, "alpha", n_ep=N_EVAL, max_turns=MAX_TURNS_EVAL,
        retain=True, verbose=True,
    )
    path_a = trace_path(agent, MazeEnvironment("alpha"), MAX_TURNS_EVAL)
    render_solution(MazeEnvironment("alpha"), path_a,
                    os.path.join(ROOT_OUT, "solution_alpha.png"))

    # -- 3. Evaluate on beta (zero-shot, no training)
    print(f"\n{'='*55}\nEVALUATE ON MAZE-BETA  (zero-shot, {N_EVAL} episodes)\n{'='*55}")
    beta_metrics = evaluate_agent(
        agent, "beta", n_ep=N_EVAL, max_turns=MAX_TURNS_EVAL,
        retain=True, verbose=True,
    )
    path_b = trace_path(agent, MazeEnvironment("beta"), MAX_TURNS_EVAL)
    render_solution(MazeEnvironment("beta"), path_b,
                    os.path.join(ROOT_OUT, "solution_beta.png"))

    # -- 4. Evaluate on gamma (extra credit)
    print(f"\n{'='*55}\nEVALUATE ON MAZE-GAMMA  (extra credit)\n{'='*55}")
    try:
        gamma_metrics = evaluate_agent(
            agent, "gamma", n_ep=N_EVAL, max_turns=MAX_TURNS_EVAL,
            retain=True, verbose=True,
        )
        path_g = trace_path(agent, MazeEnvironment("gamma"), MAX_TURNS_EVAL)
        render_solution(MazeEnvironment("gamma"), path_g,
                        os.path.join(ROOT_OUT, "solution_gamma.png"))
    except Exception as exc:
        print(f"  Gamma failed: {exc}")
        gamma_metrics = None

    # -- 5. Summary
    print(f"\n{'='*55}\nFINAL SUMMARY\n{'='*55}")
    print_metrics("alpha", alpha_metrics)
    print_metrics("beta",  beta_metrics)
    if gamma_metrics:
        print_metrics("gamma", gamma_metrics)
    print()
