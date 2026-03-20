from PIL import Image, ImageDraw
import numpy as np
from collections import deque
import colorsys
from typing import List, Tuple
from enum import Enum

# ================================================================
# Constants
# ================================================================
BORDER    = 2
CELL_SIZE = 14
STRIDE    = 16        # pixels per cell (CELL_SIZE + WALL_SIZE)
NUM_CELLS = 64        # 64x64 maze grid
MAT_SIZE  = 128       # 128x128 internal matrix (cells + passages)

# Cell type codes  (spec Appendix 11.2)
EMPTY, WALL, START, GOAL, DEATH_PIT, TELEPORT, CONFUSION = 0, 1, 2, 3, 4, 5, 6

CELL_NAMES = {
    EMPTY: "Empty", WALL: "Wall", START: "Start", GOAL: "Goal",
    DEATH_PIT: "Death Pit", TELEPORT: "Teleport", CONFUSION: "Confusion",
}

CELL_COLORS = {          # RGB used when rendering the matrix preview
    EMPTY:     (255, 255, 255),
    WALL:      (0,   0,   0  ),
    START:     (0,   210, 0  ),
    GOAL:      (30,  100, 220),
    DEATH_PIT: (220, 50,  0  ),
    TELEPORT:  (0,   190, 190),
    CONFUSION: (160, 0,   200),
}

# ================================================================
# API classes  (spec Section 6.1)
# ================================================================
class Action(Enum):
    MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, WAIT = 0, 1, 2, 3, 4

class TurnResult:
    def __init__(self):
        self.wall_hits        = 0
        self.current_position = (0, 0)
        self.is_dead          = False
        self.is_confused      = False
        self.is_goal_reached  = False
        self.teleported       = False
        self.actions_executed = 0

    def __repr__(self):
        return (f"pos={self.current_position}  dead={self.is_dead}  "
                f"goal={self.is_goal_reached}  confused={self.is_confused}  "
                f"teleported={self.teleported}  wall_hits={self.wall_hits}")

# ================================================================
# 1. Maze loading
# ================================================================
def _wall_below(gray, row, col):
    y = BORDER + row * STRIDE + CELL_SIZE
    x = BORDER + col * STRIDE + CELL_SIZE // 2
    return not (gray[y, x] > 128 and gray[y+1, x] > 128)

def _wall_right(gray, row, col):
    y = BORDER + row * STRIDE + CELL_SIZE // 2
    x = BORDER + col * STRIDE + CELL_SIZE
    return not (gray[y, x] > 128 and gray[y, x+1] > 128)

def load_maze(image_path, with_hazards=False):
    """
    Parse maze PNG into a 128x128 matrix.
    Even positions (r*2, c*2) are real cells; odd positions are wall/passage slots.
    If with_hazards=True, colour markers are detected and overlaid.
    """
    gray   = np.array(Image.open(image_path).convert("L"))
    matrix = np.ones((MAT_SIZE, MAT_SIZE), dtype=np.uint8)

    for r in range(NUM_CELLS):
        for c in range(NUM_CELLS):
            matrix[r*2, c*2] = EMPTY
            if r < NUM_CELLS - 1:
                matrix[r*2+1, c*2] = WALL if _wall_below(gray, r, c) else EMPTY
            if c < NUM_CELLS - 1:
                matrix[r*2, c*2+1] = WALL if _wall_right(gray, r, c) else EMPTY
            if r < NUM_CELLS - 1 and c < NUM_CELLS - 1:
                matrix[r*2+1, c*2+1] = WALL

    if with_hazards:
        for (r, c), cell_type in detect_hazards(image_path).items():
            matrix[r*2, c*2] = cell_type
        counts = {}
        for v in matrix[0::2, 0::2].flatten():   # count only real cells
            if v > WALL:
                counts[v] = counts.get(v, 0) + 1
        for t, n in sorted(counts.items()):
            print(f"  {n:3d}  {CELL_NAMES[t]}")

    return matrix

# ================================================================
# 2. Hazard detection via colour analysis
# ================================================================
def detect_hazards(image_path):
    """
    Scan every cell for a coloured marker by sampling a 9x9 pixel window
    around the cell centre.  Returns {(row64, col64): cell_type}.

    Hue thresholds calibrated to MAZE_1 icon colours:
        < 0.095  orange-red flames  -> DEATH_PIT
        < 0.22   gold circle        -> GOAL
        < 0.30   orange skull icons -> DEATH_PIT
        < 0.50   green shapes       -> TELEPORT
        < 0.68   blue               -> START
        < 0.80   purple (large)     -> START  |  purple (small star) -> CONFUSION
        else     deep purple        -> CONFUSION
    """
    rgb     = np.array(Image.open(image_path).convert("RGB"))
    hazards = {}

    for r in range(NUM_CELLS):
        for c in range(NUM_CELLS):
            cy = BORDER + r * STRIDE + CELL_SIZE // 2
            cx = BORDER + c * STRIDE + CELL_SIZE // 2

            # Collect saturated pixels in the 9x9 neighbourhood
            colored = []
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    y, x = cy + dy, cx + dx
                    if 0 <= y < rgb.shape[0] and 0 <= x < rgb.shape[1]:
                        h, s, v = colorsys.rgb_to_hsv(
                            rgb[y,x,0]/255, rgb[y,x,1]/255, rgb[y,x,2]/255)
                        if s > 0.30 and v > 0.25:
                            colored.append((h, s, v))

            if not colored:
                continue

            avg_h = sum(p[0] for p in colored) / len(colored)
            npix  = len(colored)

            if   avg_h < 0.095 or avg_h >= 0.94: cell_type = DEATH_PIT
            elif avg_h < 0.22:                    cell_type = GOAL
            elif avg_h < 0.30:                    cell_type = DEATH_PIT
            elif avg_h < 0.50:                    cell_type = TELEPORT
            elif avg_h < 0.68:                    cell_type = START
            elif avg_h < 0.80:
                cell_type = START if npix >= 60 else CONFUSION
            else:
                cell_type = CONFUSION

            hazards[(r, c)] = cell_type

    return hazards

# ================================================================
# 3. Start / Goal detection
# ================================================================
def find_start_and_goal(image_path):
    """Find start and goal by locating openings in the outer border wall."""
    gray   = np.array(Image.open(image_path).convert("L"))
    top    = [c for c in range(gray.shape[1]) if gray[1,  c] > 128]
    bottom = [c for c in range(gray.shape[1]) if gray[-2, c] > 128]
    tc = (top[   len(top)    // 2] - BORDER) // STRIDE * 2
    bc = (bottom[len(bottom) // 2] - BORDER) // STRIDE * 2
    return (0, tc), (NUM_CELLS*2 - 2, bc)

def start_goal_from_markers(matrix):
    """Read START / GOAL positions directly from colour markers in the matrix."""
    starts = [(r,c) for r in range(0, MAT_SIZE, 2)
                    for c in range(0, MAT_SIZE, 2) if matrix[r,c] == START]
    goals  = [(r,c) for r in range(0, MAT_SIZE, 2)
                    for c in range(0, MAT_SIZE, 2) if matrix[r,c] == GOAL]
    return (starts[0] if starts else None), (goals[0] if goals else None)

# ================================================================
# 4. BFS solver
# ================================================================
def solve(matrix, start, goal, blocked={WALL}):
    """
    BFS on the 128x128 matrix.  Returns the path as a list of (row, col)
    positions, or None if no path exists.
    blocked: set of cell-type values treated as impassable.
    """
    queue     = deque([start])
    came_from = {start: None}

    while queue:
        cur = queue.popleft()
        if cur == goal:
            path, node = [], cur
            while node is not None:
                path.append(node)
                node = came_from[node]
            return path[::-1]
        r, c = cur
        for nr, nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
            if (0 <= nr < MAT_SIZE and 0 <= nc < MAT_SIZE
                    and (nr, nc) not in came_from
                    and matrix[nr, nc] not in blocked):
                came_from[(nr, nc)] = cur
                queue.append((nr, nc))

    return None

def path_length(path):
    """Count real cells (even row AND even col) in a matrix path."""
    return sum(1 for r, c in path if r % 2 == 0 and c % 2 == 0)

def path_to_actions(path) -> List[Action]:
    """Convert a 128x128 BFS path to a list of Actions."""
    cells  = [(r,c) for r,c in path if r%2==0 and c%2==0]
    deltas = {(-2,0):Action.MOVE_UP, (2,0):Action.MOVE_DOWN,
              (0,-2):Action.MOVE_LEFT, (0,2):Action.MOVE_RIGHT}
    return [deltas[r2-r1, c2-c1]
            for (r1,c1),(r2,c2) in zip(cells, cells[1:])
            if (r2-r1, c2-c1) in deltas]

# ================================================================
# 5. MazeEnvironment  (spec Section 6.1)
# ================================================================
def build_teleport_map(matrix):
    """
    Pair teleport pads deterministically: pad[0]<->pad[1], pad[2]<->pad[3], etc.
    Returns {(mat_row, mat_col): (dst_mat_row, dst_mat_col)}.
    """
    pads = [(r, c) for r in range(0, MAT_SIZE, 2)
                   for c in range(0, MAT_SIZE, 2)
                   if matrix[r, c] == TELEPORT]
    mapping = {}
    for i in range(0, len(pads), 2):
        if i + 1 < len(pads):
            mapping[pads[i]]   = pads[i+1]
            mapping[pads[i+1]] = pads[i]
    return mapping


class MazeEnvironment:
    """Simulates the maze with full hazard mechanics."""

    def __init__(self, maze_id: str):
        if maze_id == "training":
            image = "MAZE_1.png"
        elif maze_id == "testing":
            image = "TEST_MAZE.png"
        else:
            raise ValueError(f"Unknown maze_id: '{maze_id}'")

        self.matrix    = load_maze(image, with_hazards=True)
        start, goal    = start_goal_from_markers(self.matrix)
        if not start:
            start, goal = find_start_and_goal(image)

        self.start_mat    = start
        self.goal_mat     = goal
        self.start_xy     = (start[1]//2, start[0]//2)
        self.goal_xy      = (goal[1]//2,  goal[0]//2)
        self.teleport_map = build_teleport_map(self.matrix)
        self.reset()

    def reset(self) -> Tuple[int, int]:
        self.pos           = self.start_xy
        self.confused_left = 0   # full turns of confusion remaining
        self.deaths        = 0
        self.confused_hits = 0
        self.turns         = 0
        self.explored      = set()
        self.goal_reached  = False
        return self.pos

    def _cell_type(self, x, y):
        return int(self.matrix[y*2, x*2])

    def _try_move(self, action: Action, confused: bool):
        """
        Attempt one move.  If confused, UP<->DOWN and LEFT<->RIGHT are swapped.
        Returns (new_x, new_y, hit_wall).
        """
        dx, dy = {Action.MOVE_UP:(0,-1), Action.MOVE_DOWN:(0,1),
                  Action.MOVE_LEFT:(-1,0), Action.MOVE_RIGHT:(1,0),
                  Action.WAIT:(0,0)}[action]
        if confused and action != Action.WAIT:
            dx, dy = -dx, -dy

        x, y = self.pos
        nx, ny = x + dx, y + dy

        # Bounds check
        if not (0 <= nx < NUM_CELLS and 0 <= ny < NUM_CELLS):
            return x, y, True

        # Wall check via the passage cell in the 128x128 matrix
        if self.matrix[y*2 + dy, x*2 + dx] == WALL:
            return x, y, True

        return nx, ny, False

    def step(self, actions: List[Action]) -> TurnResult:
        if not actions or len(actions) > 5:
            raise ValueError("Need 1-5 actions per turn.")

        res = TurnResult()

        # Carry-over confusion from the previous turn
        turn_confused = self.confused_left > 0
        if turn_confused:
            self.confused_left -= 1
            res.is_confused = True

        got_confused = False   # becomes True if we step on a confusion pad this turn

        for action in actions:
            nx, ny, wall_hit = self._try_move(action, turn_confused or got_confused)

            if wall_hit:
                res.wall_hits        += 1
                res.actions_executed += 1
                continue

            self.pos = (nx, ny)
            self.explored.add(self.pos)
            res.actions_executed += 1

            cell = self._cell_type(nx, ny)

            if cell == DEATH_PIT:
                res.is_dead          = True
                res.current_position = (nx, ny)   # show pit location, not respawn
                self.deaths         += 1
                self.pos             = self.start_xy
                break

            elif cell == TELEPORT:
                key = (ny*2, nx*2)
                if key in self.teleport_map:
                    dst             = self.teleport_map[key]
                    self.pos        = (dst[1]//2, dst[0]//2)
                    res.teleported  = True
                    if self._cell_type(*self.pos) == GOAL:
                        res.is_goal_reached = True
                        self.goal_reached   = True
                        break

            elif cell == CONFUSION:
                if not got_confused:
                    got_confused       = True
                    self.confused_left = 1   # next turn also confused
                    res.is_confused    = True
                    self.confused_hits += 1

            elif cell == GOAL:
                res.is_goal_reached = True
                self.goal_reached   = True
                break

        if not res.is_dead:
            res.current_position = self.pos
        self.turns += 1
        return res

    def get_episode_stats(self) -> dict:
        return {
            "turns":          self.turns,
            "deaths":         self.deaths,
            "confused":       self.confused_hits,
            "cells_explored": len(self.explored),
            "goal_reached":   self.goal_reached,
        }

# ================================================================
# 6. Agent interface  (spec Section 6.1)
# ================================================================
class Agent:
    """Base class — students must subclass this and implement plan_turn()."""

    def __init__(self):
        self.memory = {}   # students can structure as needed

    def plan_turn(self, last_result: TurnResult) -> List[Action]:
        raise NotImplementedError("Students must implement plan_turn()")

    def reset_episode(self):
        pass


class SimpleAgent(Agent):
    """Minimal demo agent — always moves right (not a real strategy)."""

    def plan_turn(self, _last_result: TurnResult) -> List[Action]:
        return [Action.MOVE_RIGHT]


# ================================================================
# 7. Visualisation
# ================================================================
def save_matrix_image(matrix, out_path, scale=8, solution=None):
    """Render 128x128 matrix as a colour PNG, optionally with solution dots."""
    size = MAT_SIZE * scale
    img  = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    for r in range(MAT_SIZE):
        for c in range(MAT_SIZE):
            color = CELL_COLORS.get(int(matrix[r,c]), (200, 0, 200))
            draw.rectangle([c*scale, r*scale, c*scale+scale-1, r*scale+scale-1], fill=color)
    if solution:
        s = scale // 3
        for r, c in solution:
            cx, cy = c*scale + scale//2, r*scale + scale//2
            draw.ellipse([cx-s, cy-s, cx+s, cy+s], fill=(220, 0, 0))
    img.save(out_path)
    print(f"  Saved: {out_path}")

def save_solution_image(image_path, solution, out_path):
    """Overlay the solution path as a red line on the original maze PNG."""
    img  = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    def to_px(r, c):
        return (BORDER + (c/2)*STRIDE + CELL_SIZE/2,
                BORDER + (r/2)*STRIDE + CELL_SIZE/2)

    draw.line([to_px(r,c) for r,c in solution], fill=(220, 50, 50), width=3)
    for pos, color in [(solution[0], (0,210,0)), (solution[-1], (30,100,220))]:
        x, y = to_px(*pos)
        draw.ellipse([x-7, y-7, x+7, y+7], fill=color)
    img.save(out_path)
    print(f"  Saved: {out_path}")

# ================================================================
# Helper: navigate to the nearest cell of a given type
# ================================================================
def navigate_to_hazard(env, matrix, start_mat, target_type, label, verbose=True):
    """
    Navigate to nearest hazard, execute actions, and clearly demonstrate behavior.
    Includes before/after state logging and confusion verification.
    """

    print(f"\n=== {label} DEMO ===")

    # BFS to nearest target
    q, seen = deque([start_mat]), {start_mat: None}
    target = None

    while q:
        cur = q.popleft()
        r, c = cur

        if matrix[r, c] == target_type and cur != start_mat:
            target = cur
            break

        for nr, nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
            if (0 <= nr < MAT_SIZE and 0 <= nc < MAT_SIZE
                    and (nr,nc) not in seen
                    and matrix[nr,nc] != WALL):
                seen[(nr,nc)] = cur
                q.append((nr,nc))

    if target is None:
        print(f"No reachable {label} found.")
        return

    # Reconstruct path
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = seen[node]
    path.reverse()

    actions = path_to_actions(path)

    # Reset environment
    env.reset()
    print(f"Start position: {env.pos}")

    # Execute in turns
    for i in range(0, len(actions), 5):
        chunk = actions[i:i+5]

        print(f"\nActions: {chunk}")
        before = env.pos

        result = env.step(chunk)
        after = result.current_position

        print(f"Before: {before} -> After: {after}")
        print(f"Result: {result}")

        # --- HAZARD DETECTION ---
        if result.is_dead:
            print("Death Pit triggered correctly")
            print(f"Respawn position: {env.pos}")
            return

        if result.teleported:
            print("Teleport triggered correctly")
            print(f"New position after teleport: {env.pos}")
            return

        if result.is_confused:
            print("Confusion triggered")

            # --- CONFUSION VERIFICATION ---
            print("\nTesting confusion effect (movement inversion)...")

            test_before = env.pos
            test_result = env.step([Action.MOVE_UP])
            test_after = test_result.current_position

            print(f"Tried MOVE_UP from {test_before}")
            print(f"Ended at {test_after}")

            if test_after[1] > test_before[1]:
                print("Movement inverted correctly (UP became DOWN)")
            else:
                print("Unexpected movement behavior")

            return

        if result.is_goal_reached:
            print("Reached goal before hazard (unexpected for this test)")
            return

    print("Reached target but no hazard triggered (check configuration)")


# ================================================================
# CHECKPOINT 1 — Load MAZE_0, solve with BFS, visualise
# ================================================================
print("\n=== Checkpoint 1: MAZE_0 (no hazards) ===")

matrix0       = load_maze("MAZE_0.png")
start0, goal0 = find_start_and_goal("MAZE_0.png")
print(f"Start: {start0}   Goal: {goal0}")

path0 = solve(matrix0, start0, goal0, blocked={WALL})
print(f"BFS solution: {path_length(path0)} cells")

save_matrix_image(matrix0, "MAZE_0_preview.png", solution=path0)
save_solution_image("MAZE_0.png", path0, "MAZE_0_solved.png")
Image.open("MAZE_0_preview.png").show()
Image.open("MAZE_0_solved.png").show()


# ================================================================
# CHECKPOINT 2 — Load MAZE_1 with hazards, visualise
# ================================================================
print("\n=== Checkpoint 2: MAZE_1 (with hazards) ===")
print("Detected hazards:")

env = MazeEnvironment("training")
save_matrix_image(env.matrix, "MAZE_1_preview_hazards.png")
Image.open("MAZE_1_preview_hazards.png").show()


# ================================================================
# CHECKPOINT 3 — Demonstrate hazard mechanics via MazeEnvironment
# ================================================================
print("\n=== Checkpoint 3: Hazard Mechanics Demo ===")
print(f"Start: {env.start_xy}   Goal: {env.goal_xy}")

navigate_to_hazard(env, env.matrix, env.start_mat, DEATH_PIT, "Death Pit")
navigate_to_hazard(env, env.matrix, env.start_mat, CONFUSION,  "Confusion")
navigate_to_hazard(env, env.matrix, env.start_mat, TELEPORT,   "Teleport")

print("\n=== Episode Stats After Demos ===")
for k, v in env.get_episode_stats().items():
    print(f"{k}: {v}")


# ================================================================
# Episode loop  (spec Section 5.2)
# ================================================================
print("\n=== Episode loop (SimpleAgent demo) ===")

agent = SimpleAgent()
env2  = MazeEnvironment("training")
last  = None

for turn in range(10000):
    actions = agent.plan_turn(last)
    last    = env2.step(actions)
    if last.is_goal_reached:
        print(f"Goal reached in {turn + 1} turns!")
        break
else:
    print("Episode ended: max turns reached")

print("Stats:", env2.get_episode_stats())
