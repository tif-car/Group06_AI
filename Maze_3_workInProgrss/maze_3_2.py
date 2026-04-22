from PIL import Image, ImageDraw
import numpy as np
from collections import deque
import colorsys
from typing import List
from enum import Enum
import math
import random

# ================================================================
# Constants
# ================================================================
BORDER = 2
CELL_SIZE = 14
STRIDE = 16
NUM_CELLS = 64
MAT_SIZE = 128

# Matrix codes
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
    EMPTY: (255, 255, 255),
    WALL: (0, 0, 0),
    START: (0, 200, 0),
    GOAL: (30, 100, 220),
    DEATH_PIT: (220, 50, 0),
    TELEPORT: (0, 190, 190),
    CONFUSION: (160, 0, 200),
}


# ================================================================
# Basic classes
# ================================================================
class Action(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    WAIT = 4


class TurnResult:
    def __init__(self):
        self.wall_hits = 0
        self.current_position = (0, 0)
        self.is_dead = False
        self.is_confused = False
        self.is_goal_reached = False
        self.teleported = False
        self.actions_executed = 0

    def __repr__(self):
        return (
            f"pos={self.current_position} dead={self.is_dead} "
            f"goal={self.is_goal_reached} confused={self.is_confused} "
            f"teleported={self.teleported} wall_hits={self.wall_hits} "
            f"actions_executed={self.actions_executed}"
        )


# ================================================================
# Small helper
# ================================================================
def popup_image(path):
    Image.open(path).show()


# ================================================================
# 1. Convert MAZE_0 image into matrix data
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
    Build the maze matrix from the clean maze image.
    Real cells are at even-even coordinates.
    The spaces between them store walls/passages.
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
# 2. Detect hazards from MAZE_1 image
# ================================================================
def detect_hazards(image_path):
    """
    Returns:
        {(row64, col64): hazard_type}
    using 64x64 maze cell coordinates.
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

            # Green / cyan-ish = teleport
            if 0.25 <= avg_h <= 0.45:
                hazards[(r, c)] = TELEPORT
                continue

            # Purple-ish = confusion
            if 0.72 <= avg_h <= 0.90:
                hazards[(r, c)] = CONFUSION
                continue

            # Dark + colored icon also likely confusion
            if dark_ratio > 0.12 and npix >= 18:
                hazards[(r, c)] = CONFUSION
                continue

            # Very bright, tight color grouping can still be teleport
            if npix >= 26 and hue_span < 0.12 and avg_v > 0.65 and avg_s > 0.40:
                hazards[(r, c)] = TELEPORT
            else:
                hazards[(r, c)] = DEATH_PIT

    return hazards


# ================================================================
# 3. Merge hazards into the matrix
# ================================================================
def merge_hazards_into_matrix(base_matrix, hazards_dict):
    merged = base_matrix.copy()

    for (r, c), hazard_type in hazards_dict.items():
        merged[r * 2, c * 2] = hazard_type

    return merged


def build_hazard_maze_matrix(base_maze_path, hazard_maze_path):
    base_matrix = load_maze(base_maze_path)
    hazards_dict = detect_hazards(hazard_maze_path)
    merged_matrix = merge_hazards_into_matrix(base_matrix, hazards_dict)
    return merged_matrix, hazards_dict


# ================================================================
# 4. Find start and goal
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
def solve(matrix, start, goal, blocked=None):
    if blocked is None:
        blocked = {WALL}

    queue = deque([start])
    came_from = {start: None}

    while queue:
        cur = queue.popleft()

        if cur == goal:
            path = []
            node = cur
            while node is not None:
                path.append(node)
                node = came_from[node]
            return path[::-1]

        r, c = cur
        for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if (
                0 <= nr < MAT_SIZE
                and 0 <= nc < MAT_SIZE
                and (nr, nc) not in came_from
                and matrix[nr, nc] not in blocked
            ):
                came_from[(nr, nc)] = cur
                queue.append((nr, nc))

    return None


def path_to_actions(path) -> List[Action]:
    cells = [(r, c) for r, c in path if r % 2 == 0 and c % 2 == 0]

    deltas = {
        (-2, 0): Action.MOVE_UP,
        (2, 0): Action.MOVE_DOWN,
        (0, -2): Action.MOVE_LEFT,
        (0, 2): Action.MOVE_RIGHT,
    }

    actions = []
    for (r1, c1), (r2, c2) in zip(cells, cells[1:]):
        dr = r2 - r1
        dc = c2 - c1
        if (dr, dc) in deltas:
            actions.append(deltas[(dr, dc)])

    return actions


# ================================================================
# 6. Draw matrix back into image
# ================================================================
def save_matrix_image(matrix, out_path, scale=8, solution=None):
    size = MAT_SIZE * scale
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)

    for r in range(MAT_SIZE):
        for c in range(MAT_SIZE):
            val = int(matrix[r, c])
            color = CELL_COLORS.get(val, (255, 0, 255))
            draw.rectangle(
                [c * scale, r * scale, c * scale + scale - 1, r * scale + scale - 1],
                fill=color
            )

    if solution:
        dot = max(2, scale // 3)
        for r, c in solution:
            cx = c * scale + scale // 2
            cy = r * scale + scale // 2
            draw.ellipse([cx - dot, cy - dot, cx + dot, cy + dot], fill=(255, 255, 0))

    img.save(out_path)


def save_matrix_image_with_labels(matrix, out_path, scale=8):
    start, goal = find_start_and_goal("MAZE_0.png")
    matrix_copy = matrix.copy()
    matrix_copy[start[0], start[1]] = START
    matrix_copy[goal[0], goal[1]] = GOAL
    save_matrix_image(matrix_copy, out_path, scale=scale)


def save_solution_image_on_original(image_path, solution, out_path):
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


# ================================================================
# 7. Build teleport map
# ================================================================
def build_teleport_map(matrix):
    """
    Temporary deterministic pairing for check-in demo.
    Pairs teleports in scan order.
    """
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


# ================================================================
# 8. Environment uses the matrix directly
# ================================================================
class MazeEnvironment:
    def __init__(self, maze_id: str):
        if maze_id == "training":
            base_image = "MAZE_0.png"
            hazard_image = "MAZE_1.png"
        else:
            raise ValueError(f"Unknown maze_id: {maze_id}")

        self.base_image_path = base_image
        self.hazard_image_path = hazard_image

        self.matrix, self.hazards_dict = build_hazard_maze_matrix(base_image, hazard_image)

        self.start_mat, self.goal_mat = find_start_and_goal(base_image)
        self.teleport_map = build_teleport_map(self.matrix)

        self.start_xy = (self.start_mat[1] // 2, self.start_mat[0] // 2)
        self.goal_xy = (self.goal_mat[1] // 2, self.goal_mat[0] // 2)

        self.fire_cells = set()
        for r in range(0, MAT_SIZE, 2):
            for c in range(0, MAT_SIZE, 2):
                if self.matrix[r, c] == DEATH_PIT:
                    self.fire_cells.add((c // 2, r // 2))

        self.reset()

    def reset(self):
        self.pos = self.start_xy
        self.confused_left = 0
        self.turns = 0
        self.deaths = 0
        self.confused_hits = 0
        self.explored = set()
        self.goal_reached = False
        self.fire_rotation_degrees = 0
        self.pending_respawn = False
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

        # Respawn at start on next turn after dying
        if self.pending_respawn:
            self.pos = self.start_xy
            self.pending_respawn = False

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
                self.pending_respawn = True
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


# ================================================================
# 9. Evolutionary Computing / Genetic Algorithm
# ================================================================
ACTION_LIST = [
    Action.MOVE_UP,
    Action.MOVE_DOWN,
    Action.MOVE_LEFT,
    Action.MOVE_RIGHT,
    Action.WAIT,
]

# Faster and more practical settings

POPULATION_SIZE = 12
CHROMOSOME_LENGTH = 500
GENERATIONS = 12
MUTATION_RATE = 0.15
ELITE_COUNT = 2
TOURNAMENT_SIZE = 3


def manhattan_distance(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return abs(x1 - x2) + abs(y1 - y2)


def random_chromosome(length=CHROMOSOME_LENGTH):
    return [random.randint(0, len(ACTION_LIST) - 1) for _ in range(length)]


def decode_chromosome(chromosome):
    return [ACTION_LIST[g] for g in chromosome]


def evaluate_chromosome(env_template, chromosome):
    """
    Runs one candidate path and returns:
    fitness, route, solved, stats
    """
    env = MazeEnvironment("training")
    env.reset()

    route = [(env.pos[1] * 2, env.pos[0] * 2)]
    visited = {env.pos}
    total_wall_hits = 0
    repeated_steps = 0
    best_distance = manhattan_distance(env.pos, env.goal_xy)

    for step_idx, gene in enumerate(chromosome):
        action = ACTION_LIST[gene]
        result = env.step([action])
        total_wall_hits += result.wall_hits
        route.append((result.current_position[1] * 2, result.current_position[0] * 2))

        if result.current_position in visited:
            repeated_steps += 1
        visited.add(result.current_position)

        dist = manhattan_distance(env.pos, env.goal_xy)
        best_distance = min(best_distance, dist)

        if result.is_goal_reached:
            fitness = (
                200000
                - env.turns * 10
                - env.deaths * 800
                - total_wall_hits * 20
                - repeated_steps * 3
            )
            stats = {
                "turns": env.turns,
                "deaths": env.deaths,
                "wall_hits": total_wall_hits,
                "unique_cells": len(visited),
                "best_distance": best_distance,
            }
            return fitness, route, True, stats

        # early stop for obviously bad candidates
        if step_idx > 80 and len(visited) < 15:
            break

    final_pos = env.pos
    dist = manhattan_distance(final_pos, env.goal_xy)

    fitness = 0
    fitness -= dist * 60
    fitness -= best_distance * 40
    fitness -= env.deaths * 800
    fitness -= total_wall_hits * 15
    fitness -= repeated_steps * 3
    fitness += len(visited) * 8

    stats = {
        "turns": env.turns,
        "deaths": env.deaths,
        "wall_hits": total_wall_hits,
        "unique_cells": len(visited),
        "best_distance": best_distance,
    }
    return fitness, route, False, stats


def tournament_selection(population, fitnesses, k=TOURNAMENT_SIZE):
    indices = random.sample(range(len(population)), k)
    best_idx = max(indices, key=lambda idx: fitnesses[idx])
    return population[best_idx][:]


def crossover(parent1, parent2):
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have same length.")
    point = random.randint(1, len(parent1) - 2)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(chromosome, mutation_rate=MUTATION_RATE):
    child = chromosome[:]
    for i in range(len(child)):
        if random.random() < mutation_rate:
            child[i] = random.randint(0, len(ACTION_LIST) - 1)
    return child


def evolve_population():
    population = [random_chromosome() for _ in range(POPULATION_SIZE)]

    best_overall = None
    best_fitness = -float("inf")
    best_route = None
    best_solved = False
    best_stats = None

    for generation in range(GENERATIONS):
        evaluations = [evaluate_chromosome(None, chrom) for chrom in population]
        fitnesses = [ev[0] for ev in evaluations]

        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_fit, gen_best_route, gen_best_solved, gen_best_stats = evaluations[gen_best_idx]

        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit
            best_overall = population[gen_best_idx][:]
            best_route = gen_best_route
            best_solved = gen_best_solved
            best_stats = gen_best_stats

        print(
            f"Generation {generation + 1}: "
            f"best_fitness={gen_best_fit:.2f} "
            f"solved={gen_best_solved} "
            f"turns={gen_best_stats['turns']} "
            f"deaths={gen_best_stats['deaths']} "
            f"unique_cells={gen_best_stats['unique_cells']} "
            f"best_distance={gen_best_stats['best_distance']}"
        )

        if gen_best_solved:
            print("Solved maze during evolution.")
            break

        elite_indices = np.argsort(fitnesses)[-ELITE_COUNT:]
        new_population = [population[i][:] for i in elite_indices]

        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)

            new_population.append(child1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        population = new_population

    return best_overall, best_fitness, best_route, best_solved, best_stats


# ================================================================
# 10. Matrix-based Part 5 image
# ================================================================
def save_part5_from_matrix(matrix, fire_cells, rotation_degrees, out_path, scale=8):
    size = MAT_SIZE * scale
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)

    for r in range(MAT_SIZE):
        for c in range(MAT_SIZE):
            val = int(matrix[r, c])
            color = CELL_COLORS.get(val, (255, 0, 255))
            draw.rectangle(
                [c * scale, r * scale, c * scale + scale - 1, r * scale + scale - 1],
                fill=color
            )

    angle = math.radians(rotation_degrees)
    line_len = max(3, scale)

    for (x, y) in fire_cells:
        mr = y * 2
        mc = x * 2

        cx = mc * scale + scale // 2
        cy = mr * scale + scale // 2

        dx = int(line_len * math.cos(angle))
        dy = int(line_len * math.sin(angle))

        draw.line([cx, cy, cx + dx, cy + dy], fill=(255, 255, 0), width=2)

        dot = max(2, scale // 3)
        draw.ellipse([cx - dot, cy - dot, cx + dot, cy + dot], fill=(255, 255, 0))

    img.save(out_path)


# ================================================================
# 11. Hazard debug helpers
# ================================================================
def print_hazards(hazards_dict):
    counts = {DEATH_PIT: 0, TELEPORT: 0, CONFUSION: 0}
    for h in hazards_dict.values():
        if h in counts:
            counts[h] += 1

    print("Hazard counts:")
    print("Death pits:", counts[DEATH_PIT])
    print("Teleports :", counts[TELEPORT])
    print("Confusion :", counts[CONFUSION])


def demo_hazard_cells(env):
    print("\nSample hazards detected:")
    shown = 0
    for (r, c), h in env.hazards_dict.items():
        print(f"Cell ({r}, {c}) -> {CELL_NAMES[h]}")
        shown += 1
        if shown >= 10:
            break


def find_open_neighbor_for_hazard(matrix, hr, hc):
    """
    Given a hazard at 64x64 cell coords (hr, hc),
    find an adjacent open cell and the action needed to step into hazard.
    """
    options = [
        (hr - 1, hc, Action.MOVE_DOWN),
        (hr + 1, hc, Action.MOVE_UP),
        (hr, hc - 1, Action.MOVE_RIGHT),
        (hr, hc + 1, Action.MOVE_LEFT),
    ]

    for nr, nc, action in options:
        if 0 <= nr < NUM_CELLS and 0 <= nc < NUM_CELLS:
            if matrix[nr * 2, nc * 2] == EMPTY:
                dy = hr - nr
                dx = hc - nc
                if matrix[nr * 2 + dy, nc * 2 + dx] != WALL:
                    return (nc, nr), action

    return None, None


def demo_specific_hazard(env, hazard_type):
    for (hr, hc), h in env.hazards_dict.items():
        if h != hazard_type:
            continue

        start_pos, action = find_open_neighbor_for_hazard(env.matrix, hr, hc)
        if start_pos is None:
            continue

        env.pos = start_pos
        env.pending_respawn = False
        env.confused_left = 0

        print(f"\nTesting {CELL_NAMES[hazard_type]} at cell ({hr}, {hc})")
        print("Agent starting at:", env.pos)
        print("Action:", action.name)

        result = env.step([action])
        print("Result:", result)
        print("Agent end pos:", env.pos)
        return

    print(f"\nNo reachable {CELL_NAMES[hazard_type]} found for demo.")


def demo_wall_hit(env):
    x, y = env.start_xy
    env.pos = (x, y)
    env.pending_respawn = False
    env.confused_left = 0

    for action in [Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.MOVE_UP, Action.MOVE_DOWN]:
        env.pos = (x, y)
        result = env.step([action])
        if result.wall_hits > 0:
            print("\nTesting wall behavior")
            print("Start:", (x, y))
            print("Action:", action.name)
            print("Result:", result)
            return

    print("\nCould not demonstrate wall hit from start cell.")


# ================================================================
# 12. Main demo
# ================================================================
if __name__ == "__main__":
    # A) Build base maze matrix from MAZE_0
    matrix0 = load_maze("MAZE_0.png")

    # B) Save base computation matrix
    save_matrix_image_with_labels(matrix0, "MAZE_0_matrix.png", scale=8)

    # C) Detect hazards from MAZE_1 and merge into matrix
    matrix1, hazards_dict = build_hazard_maze_matrix("MAZE_0.png", "MAZE_1.png")
    print_hazards(hazards_dict)

    # D) Save final computation matrix with hazards
    save_matrix_image_with_labels(matrix1, "MAZE_1_matrix_with_hazards.png", scale=8)

    # E) Environment + hazard demos
    env = MazeEnvironment("training")
    demo_hazard_cells(env)
    demo_wall_hit(env)
    demo_specific_hazard(env, DEATH_PIT)
    demo_specific_hazard(env, TELEPORT)
    demo_specific_hazard(env, CONFUSION)

    # F) Run evolutionary computing
    print("\nRunning evolutionary computing...")
    best_chromosome, best_fitness, best_route, solved, best_stats = evolve_population()

    print("\nBest overall result:")
    print("Solved:", solved)
    print("Best fitness:", best_fitness)
    print("Turns:", best_stats["turns"])
    print("Deaths:", best_stats["deaths"])
    print("Unique cells:", best_stats["unique_cells"])
    print("Best distance:", best_stats["best_distance"])

    # Save best route even if partial
    if best_route is not None:
        save_matrix_image(env.matrix, "EC_best_route_matrix.png", scale=8, solution=best_route)
        save_solution_image_on_original("MAZE_1.png", best_route, "EC_best_route_original.png")
        if solved:
            print("Saved solved EC route images.")
        else:
            print("Saved best partial EC route images (did not fully solve).")
    else:
        print("No route image could be produced.")

    # G) Save Part 5 image
    save_part5_from_matrix(
        env.matrix,
        env.fire_cells,
        env.fire_rotation_degrees,
        "MAZE_1_part5_from_matrix.png",
        scale=8
    )

    print("\nSaved image files:")
    print(" - MAZE_0_matrix.png")
    print(" - MAZE_1_matrix_with_hazards.png")
    print(" - MAZE_1_part5_from_matrix.png")
    if best_route is not None:
        print(" - EC_best_route_matrix.png")
        print(" - EC_best_route_original.png")