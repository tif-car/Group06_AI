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
STRIDE    = 16
NUM_CELLS = 64
MAT_SIZE  = 128

# Cell type codes
EMPTY, WALL, START, GOAL, DEATH_PIT, TELEPORT, CONFUSION = 0, 1, 2, 3, 4, 5, 6

CELL_NAMES = {
    EMPTY: "Empty",
    WALL: "Wall",
    START: "Start",
    GOAL: "Goal",
    DEATH_PIT: "Death Pit",
    TELEPORT: "Teleport",
    CONFUSION: "Confusion",
}

CELL_COLORS = {
    EMPTY:     (255, 255, 255),
    WALL:      (0,   0,   0),
    START:     (0,   210, 0),
    GOAL:      (30,  100, 220),
    DEATH_PIT: (220, 50,  0),
    TELEPORT:  (0,   190, 190),
    CONFUSION: (160, 0,   200),
}


# ================================================================
# API classes
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
# Popup helper
# ================================================================
def popup_image(path, title=None):
    Image.open(path).show()


# ================================================================
# 1. Maze loading
# ================================================================
def _wall_below(gray, row, col):
    y = BORDER + row * STRIDE + CELL_SIZE
    x = BORDER + col * STRIDE + CELL_SIZE // 2
    return not (gray[y, x] > 128 and gray[y + 1, x] > 128)


def _wall_right(gray, row, col):
    y = BORDER + row * STRIDE + CELL_SIZE // 2
    x = BORDER + col * STRIDE + CELL_SIZE
    return not (gray[y, x] > 128 and gray[y, x + 1] > 128)


def load_maze(image_path):
    """
    Parse the clean maze PNG into a 128x128 matrix.
    Even positions are real cells; odd positions are wall/passage slots.
    """
    gray = np.array(Image.open(image_path).convert("L"))
    matrix = np.ones((MAT_SIZE, MAT_SIZE), dtype=np.uint8)

    for r in range(NUM_CELLS):
        for c in range(NUM_CELLS):
            matrix[r * 2, c * 2] = EMPTY

            if r < NUM_CELLS - 1:
                matrix[r * 2 + 1, c * 2] = WALL if _wall_below(gray, r, c) else EMPTY

            if c < NUM_CELLS - 1:
                matrix[r * 2, c * 2 + 1] = WALL if _wall_right(gray, r, c) else EMPTY

            if r < NUM_CELLS - 1 and c < NUM_CELLS - 1:
                matrix[r * 2 + 1, c * 2 + 1] = WALL

    return matrix


# ================================================================
# 2. Hazard detection
# ================================================================
def detect_hazards(image_path):
    """
    Scan every cell for a coloured marker by sampling a 9x9 pixel window
    around the cell centre. Returns {(row64, col64): cell_type}.
    """
    rgb = np.array(Image.open(image_path).convert("RGB"))
    hazards = {}

    for r in range(NUM_CELLS):
        for c in range(NUM_CELLS):
            cy = BORDER + r * STRIDE + CELL_SIZE // 2
            cx = BORDER + c * STRIDE + CELL_SIZE // 2

            colored = []
            dark_pixels = 0

            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    y, x = cy + dy, cx + dx
                    if 0 <= y < rgb.shape[0] and 0 <= x < rgb.shape[1]:
                        rr, gg, bb = rgb[y, x]
                        h, s, v = colorsys.rgb_to_hsv(rr / 255, gg / 255, bb / 255)

                        if s > 0.28 and v > 0.25:
                            colored.append((h, s, v, rr, gg, bb))

                        if rr < 95 and gg < 95 and bb < 95:
                            dark_pixels += 1

            if not colored:
                continue

            hues = [p[0] for p in colored]
            avg_h = sum(p[0] for p in colored) / len(colored)
            avg_s = sum(p[1] for p in colored) / len(colored)
            avg_v = sum(p[2] for p in colored) / len(colored)
            npix = len(colored)
            hue_span = max(hues) - min(hues)
            dark_ratio = dark_pixels / 81.0

            # Keep your original logic if it was already tuned for your images
            if 0.25 <= avg_h <= 0.45:
                hazards[(r, c)] = TELEPORT
                continue

            if 0.72 <= avg_h <= 0.90:
                hazards[(r, c)] = TELEPORT
                continue

            if dark_ratio > 0.12 and npix >= 18:
                hazards[(r, c)] = CONFUSION
                continue

            if npix >= 26 and hue_span < 0.12 and avg_v > 0.65 and avg_s > 0.40:
                hazards[(r, c)] = TELEPORT
            else:
                hazards[(r, c)] = DEATH_PIT

    return hazards


# ================================================================
# 3. Merge hazards into the maze matrix
# ================================================================
def merge_hazards_into_matrix(base_matrix, hazards_dict):
    """
    Write hazard values directly into the real cell positions of the base matrix.
    hazards_dict keys are in 64x64 cell coordinates.
    """
    merged = base_matrix.copy()

    for (r, c), hazard_type in hazards_dict.items():
        merged[r * 2, c * 2] = hazard_type

    return merged


def build_hazard_maze_matrix(base_maze_path, hazard_maze_path):
    """
    Build maze structure from the clean/base maze,
    then inject hazard cells detected from the hazard image.
    """
    base_matrix = load_maze(base_maze_path)
    hazards_dict = detect_hazards(hazard_maze_path)
    merged_matrix = merge_hazards_into_matrix(base_matrix, hazards_dict)
    return merged_matrix, hazards_dict


# ================================================================
# 4. Start / Goal detection
# ================================================================
def find_start_and_goal(image_path):
    gray = np.array(Image.open(image_path).convert("L"))
    top = [c for c in range(gray.shape[1]) if gray[1, c] > 128]
    bottom = [c for c in range(gray.shape[1]) if gray[-2, c] > 128]

    tc = ((top[len(top) // 2] - BORDER) // STRIDE) * 2
    bc = ((bottom[len(bottom) // 2] - BORDER) // STRIDE) * 2

    return (0, tc), (NUM_CELLS * 2 - 2, bc)


# ================================================================
# 5. BFS solver
# ================================================================
def solve(matrix, start, goal, blocked={WALL}):
    queue = deque([start])
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
        for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if (0 <= nr < MAT_SIZE and 0 <= nc < MAT_SIZE
                    and (nr, nc) not in came_from
                    and matrix[nr, nc] not in blocked):
                came_from[(nr, nc)] = cur
                queue.append((nr, nc))

    return None


def path_length(path):
    return sum(1 for r, c in path if r % 2 == 0 and c % 2 == 0)


def path_to_actions(path) -> List[Action]:
    cells = [(r, c) for r, c in path if r % 2 == 0 and c % 2 == 0]

    deltas = {
        (-2, 0): Action.MOVE_UP,
        (2, 0): Action.MOVE_DOWN,
        (0, -2): Action.MOVE_LEFT,
        (0, 2): Action.MOVE_RIGHT,
    }

    return [
        deltas[(r2 - r1, c2 - c1)]
        for (r1, c1), (r2, c2) in zip(cells, cells[1:])
        if (r2 - r1, c2 - c1) in deltas
    ]


# ================================================================
# 6. MazeEnvironment
# ================================================================
def build_teleport_map(matrix):
    pads = [
        (r, c)
        for r in range(0, MAT_SIZE, 2)
        for c in range(0, MAT_SIZE, 2)
        if matrix[r, c] == TELEPORT
    ]

    mapping = {}
    for i in range(0, len(pads), 2):
        if i + 1 < len(pads):
            mapping[pads[i]] = pads[i + 1]
            mapping[pads[i + 1]] = pads[i]

    return mapping


class MazeEnvironment:
    """
    Uses MAZE_0 as the base maze structure,
    then injects hazards detected from the hazard image.
    """
    def __init__(self, maze_id: str):
        if maze_id == "training":
            base_image = "MAZE_0.png"
            hazard_image = "MAZE_1.png"
        elif maze_id == "testing":
            # Change these if your testing set uses different files
            base_image = "MAZE_0.png"
            hazard_image = "TEST_MAZE.png"
        else:
            raise ValueError(f"Unknown maze_id: '{maze_id}'")

        self.base_image_path = base_image
        self.image_path = hazard_image

        # IMPORTANT: build the maze from the clean maze,
        # then paste hazards into the matrix
        self.matrix, self.hazards_dict = build_hazard_maze_matrix(base_image, hazard_image)

        self.fire_cells = set()
        for r in range(0, MAT_SIZE, 2):
            for c in range(0, MAT_SIZE, 2):
                if self.matrix[r, c] == DEATH_PIT:
                    self.fire_cells.add((c // 2, r // 2))

        start, goal = find_start_and_goal(base_image)

        self.start_mat = start
        self.goal_mat = goal
        self.start_xy = (start[1] // 2, start[0] // 2)
        self.goal_xy = (goal[1] // 2, goal[0] // 2)
        self.teleport_map = build_teleport_map(self.matrix)
        self.reset()

    def reset(self) -> Tuple[int, int]:
        self.pos = self.start_xy
        self.confused_left = 0
        self.deaths = 0
        self.confused_hits = 0
        self.turns = 0
        self.explored = set()
        self.goal_reached = False
        self.fire_rotation_degrees = 0
        return self.pos

    def _cell_type(self, x, y):
        return int(self.matrix[y * 2, x * 2])

    def rotate_fire_icon(self):
        self.fire_rotation_degrees = (self.fire_rotation_degrees + 90) % 360

    def _try_move(self, action: Action, confused: bool):
        dx, dy = {
            Action.MOVE_UP: (0, -1),
            Action.MOVE_DOWN: (0, 1),
            Action.MOVE_LEFT: (-1, 0),
            Action.MOVE_RIGHT: (1, 0),
            Action.WAIT: (0, 0),
        }[action]

        if confused and action != Action.WAIT:
            dx, dy = -dx, -dy

        x, y = self.pos
        nx, ny = x + dx, y + dy

        if not (0 <= nx < NUM_CELLS and 0 <= ny < NUM_CELLS):
            return x, y, True

        if self.matrix[y * 2 + dy, x * 2 + dx] == WALL:
            return x, y, True

        return nx, ny, False

    def step(self, actions: List[Action]) -> TurnResult:
        if not actions or len(actions) > 5:
            raise ValueError("Need 1-5 actions per turn.")

        res = TurnResult()

        turn_confused = self.confused_left > 0
        if turn_confused:
            self.confused_left -= 1
            res.is_confused = True

        got_confused = False

        for action in actions:
            nx, ny, wall_hit = self._try_move(action, turn_confused or got_confused)

            if wall_hit:
                res.wall_hits += 1
                res.actions_executed += 1
                continue

            self.pos = (nx, ny)
            self.explored.add(self.pos)
            res.actions_executed += 1

            cell = self._cell_type(nx, ny)

            if cell == DEATH_PIT:
                res.is_dead = True
                res.current_position = (nx, ny)
                self.deaths += 1
                self.pos = self.start_xy
                break

            if cell == TELEPORT:
                key = (ny * 2, nx * 2)
                if key in self.teleport_map:
                    dst = self.teleport_map[key]
                    self.pos = (dst[1] // 2, dst[0] // 2)
                    res.teleported = True

            elif cell == CONFUSION:
                if not got_confused:
                    got_confused = True
                    self.confused_left = 1
                    res.is_confused = True
                    self.confused_hits += 1

            if self.pos == self.goal_xy:
                res.is_goal_reached = True
                self.goal_reached = True
                break

        self.rotate_fire_icon()

        if not res.is_dead:
            res.current_position = self.pos

        self.turns += 1
        return res

    def get_episode_stats(self) -> dict:
        return {
            "turns": self.turns,
            "deaths": self.deaths,
            "confused": self.confused_hits,
            "cells_explored": len(self.explored),
            "goal_reached": self.goal_reached,
        }


# ================================================================
# 7. Agent interface
# ================================================================
class Agent:
    def __init__(self):
        self.memory = {}

    def plan_turn(self, last_result: TurnResult) -> List[Action]:
        raise NotImplementedError("Students must implement plan_turn()")

    def reset_episode(self):
        pass


class SimpleAgent(Agent):
    def plan_turn(self, _last_result: TurnResult) -> List[Action]:
        return [Action.MOVE_RIGHT]


# ================================================================
# 8. Visualisation
# ================================================================
def save_matrix_image(matrix, out_path, scale=8, solution=None):
    size = MAT_SIZE * scale
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)

    for r in range(MAT_SIZE):
        for c in range(MAT_SIZE):
            color = CELL_COLORS.get(int(matrix[r, c]), (200, 0, 200))
            draw.rectangle(
                [c * scale, r * scale, c * scale + scale - 1, r * scale + scale - 1],
                fill=color
            )

    if solution:
        s = scale // 3
        for r, c in solution:
            cx, cy = c * scale + scale // 2, r * scale + scale // 2
            draw.ellipse([cx - s, cy - s, cx + s, cy + s], fill=(220, 0, 0))

    img.save(out_path)


def save_solution_image(image_path, solution, out_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    def to_px(r, c):
        return (
            BORDER + (c / 2) * STRIDE + CELL_SIZE / 2,
            BORDER + (r / 2) * STRIDE + CELL_SIZE / 2
        )

    draw.line([to_px(r, c) for r, c in solution], fill=(220, 50, 50), width=3)

    for pos, color in [(solution[0], (0, 210, 0)), (solution[-1], (30, 100, 220))]:
        x, y = to_px(*pos)
        draw.ellipse([x - 7, y - 7, x + 7, y + 7], fill=color)

    img.save(out_path)


def _icon_mask_for_patch(patch_rgb):
    h, w, _ = patch_rgb.shape
    mask = np.zeros((h, w), dtype=bool)

    for y in range(h):
        for x in range(w):
            rr, gg, bb = patch_rgb[y, x]
            hh, ss, vv = colorsys.rgb_to_hsv(rr / 255, gg / 255, bb / 255)
            if ss > 0.22 and vv > 0.22:
                mask[y, x] = True

    return mask


def _extract_icon_patch(src_img, x, y, crop_size=14):
    half = crop_size // 2
    cx = BORDER + x * STRIDE + CELL_SIZE // 2
    cy = BORDER + y * STRIDE + CELL_SIZE // 2

    left = max(0, cx - half)
    top = max(0, cy - half)
    right = min(src_img.width, cx + half)
    bottom = min(src_img.height, cy + half)

    patch = src_img.crop((left, top, right, bottom)).convert("RGBA")
    patch_np = np.array(patch)
    mask = _icon_mask_for_patch(patch_np[:, :, :3])

    icon_only = patch_np.copy()
    icon_only[:, :, 3] = 0
    icon_only[mask, 3] = 255

    return Image.fromarray(icon_only, mode="RGBA")


def _paste_icon_patch(base_img, patch, x, y, crop_size=14):
    half = crop_size // 2
    cx = BORDER + x * STRIDE + CELL_SIZE // 2
    cy = BORDER + y * STRIDE + CELL_SIZE // 2
    left = max(0, cx - half)
    top = max(0, cy - half)
    base_img.alpha_composite(patch, dest=(left, top))


def save_hazards_from_matrix_image(clean_maze_path, hazard_maze_path, hazards_dict, out_path, crop_size=14):
    """
    Visual output only:
    Build a clean maze image with hazard icons pasted on top.
    """
    base = Image.open(clean_maze_path).convert("RGBA")
    src = Image.open(hazard_maze_path).convert("RGBA")

    for (r, c), hazard_type in hazards_dict.items():
        patch = _extract_icon_patch(src, c, r, crop_size=crop_size)
        _paste_icon_patch(base, patch, c, r, crop_size=crop_size)

    base.convert("RGB").save(out_path)


def save_part5_rotated_in_place_image(clean_maze_path, hazard_maze_path, hazards_dict, fire_cells, rotation_degrees, out_path, crop_size=14):
    """
    Build a visual image where non-fire hazards are pasted normally
    and fire icons are rotated in place.
    """
    base = Image.open(clean_maze_path).convert("RGBA")
    src = Image.open(hazard_maze_path).convert("RGBA")

    for (r, c), hazard_type in hazards_dict.items():
        if hazard_type != DEATH_PIT:
            patch = _extract_icon_patch(src, c, r, crop_size=crop_size)
            _paste_icon_patch(base, patch, c, r, crop_size=crop_size)

    for (x, y) in fire_cells:
        flame_patch = _extract_icon_patch(src, x, y, crop_size=crop_size)

        if rotation_degrees == 90:
            rotated_patch = flame_patch.rotate(-90, expand=True)
        elif rotation_degrees == 180:
            rotated_patch = flame_patch.rotate(180, expand=True)
        elif rotation_degrees == 270:
            rotated_patch = flame_patch.rotate(90, expand=True)
        else:
            rotated_patch = flame_patch.copy()

        _paste_icon_patch(base, rotated_patch, x, y, crop_size=crop_size)

    base.convert("RGB").save(out_path)


# ================================================================
# 9. Helper demos
# ================================================================
def save_part5_image(env, hazards_dict):
    out_path = "MAZE_1_part5.png"
    save_part5_rotated_in_place_image(
        "MAZE_0.png",
        env.image_path,
        hazards_dict,
        env.fire_cells,
        env.fire_rotation_degrees,
        out_path
    )
    return out_path


def demo_fire_rotation(env, hazards_dict):
    env.reset()
    env.step([Action.WAIT])
    out_path = save_part5_image(env, hazards_dict)
    Image.open(out_path).show()


def navigate_to_hazard(env, matrix, start_mat, target_type):
    q, seen = deque([start_mat]), {start_mat: None}
    target = None

    while q:
        cur = q.popleft()
        r, c = cur

        if matrix[r, c] == target_type and cur != start_mat:
            target = cur
            break

        for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if (0 <= nr < MAT_SIZE and 0 <= nc < MAT_SIZE
                    and (nr, nc) not in seen
                    and matrix[nr, nc] != WALL):
                seen[(nr, nc)] = cur
                q.append((nr, nc))

    if target is None:
        return

    path = []
    node = target
    while node is not None:
        path.append(node)
        node = seen[node]
    path.reverse()

    actions = path_to_actions(path)
    env.reset()

    for i in range(0, len(actions), 5):
        chunk = actions[i:i + 5]
        result = env.step(chunk)

        if result.is_dead or result.teleported or result.is_confused or result.is_goal_reached:
            return


# ================================================================
# CHECKPOINT 1 — Load MAZE_0, solve with BFS, visualise
# ================================================================
matrix0 = load_maze("MAZE_0.png")
start0, goal0 = find_start_and_goal("MAZE_0.png")
path0 = solve(matrix0, start0, goal0, blocked={WALL})

save_matrix_image(matrix0, "MAZE_0_preview.png", solution=path0)
save_solution_image("MAZE_0.png", path0, "MAZE_0_solved.png")
popup_image("MAZE_0_solved.png", "MAZE 0 Solved")


# ================================================================
# CHECKPOINT 2 — Build FINAL hazard matrix correctly
# ================================================================
matrix1, hazards_dict = build_hazard_maze_matrix("MAZE_0.png", "MAZE_1.png")
save_matrix_image(matrix1, "MAZE_1_matrix_with_hazards.png")

save_hazards_from_matrix_image(
    "MAZE_0.png",
    "MAZE_1.png",
    hazards_dict,
    "MAZE_1_hazards.png"
)

popup_image("MAZE_1_hazards.png", "MAZE 1 Hazards")
popup_image("MAZE_1_matrix_with_hazards.png", "Matrix With Hazards")


# ================================================================
# CHECKPOINT 3 — Demonstrate hazard mechanics
# ================================================================
env = MazeEnvironment("training")

demo_fire_rotation(env, hazards_dict)
navigate_to_hazard(env, env.matrix, env.start_mat, DEATH_PIT)
navigate_to_hazard(env, env.matrix, env.start_mat, CONFUSION)
navigate_to_hazard(env, env.matrix, env.start_mat, TELEPORT)


# ================================================================
# Episode loop
# ================================================================
agent = SimpleAgent()
env2 = MazeEnvironment("training")
last = None

for _ in range(10000):
    actions = agent.plan_turn(last)
    last = env2.step(actions)
    if last.is_goal_reached:
        break
