from PIL import Image
import numpy as np
import math, random, colorsys, threading, time
import tkinter as tk
from enum import Enum
from typing import List, Tuple


# ================================================================
# 1. Official API Specifications (From Document Section 6)
# ================================================================
class Action(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    WAIT = 4


class TurnResult:
    def __init__(self):
        self.wall_hits: int = 0
        self.current_position: Tuple[int, int] = (0, 0)
        self.is_dead: bool = False
        self.is_confused: bool = False
        self.is_goal_reached: bool = False
        self.teleported: bool = False
        self.actions_executed: int = 0


class Agent:
    """Base class for student implementations"""

    def __init__(self):
        self.memory = {}

    def plan_turn(self, last_result: TurnResult) -> List[Action]:
        raise NotImplementedError("Students must implement this method")

    def reset_episode(self):
        pass


# ================================================================
# Configuration & Constants
# ================================================================
BORDER, CELL_SIZE, STRIDE, NUM_CELLS, MAT_SIZE = 2, 14, 16, 64, 128
EMPTY, WALL, DEATH_PIT, T_GREEN, T_YELLOW, T_PURPLE, CONFUSION = 0, 1, 4, 7, 8, 9, 6
FIRE = 10


# ================================================================
# 2. Image Loading & Robust Hazard Detection
# ================================================================
def load_maze(path):
    img = Image.open(path)
    gray = np.array(img.convert("L"))
    rgb = np.array(img.convert("RGB"))
    mat = np.zeros((MAT_SIZE, MAT_SIZE), dtype=np.uint8)

    for r in range(NUM_CELLS):
        for c in range(NUM_CELLS):
            if r < 63: mat[r * 2 + 1, c * 2] = WALL if gray[
                                                           BORDER + r * STRIDE + CELL_SIZE, BORDER + c * STRIDE + CELL_SIZE // 2] < 128 else EMPTY
            if c < 63: mat[r * 2, c * 2 + 1] = WALL if gray[
                                                           BORDER + r * STRIDE + CELL_SIZE // 2, BORDER + c * STRIDE + CELL_SIZE] < 128 else EMPTY
            if r < 63 and c < 63: mat[r * 2 + 1, c * 2 + 1] = WALL

            cy, cx = BORDER + r * STRIDE + CELL_SIZE // 2, BORDER + c * STRIDE + CELL_SIZE // 2
            hues = []
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    rr, gg, bb = rgb[cy + dy, cx + dx]
                    h, s, v = colorsys.rgb_to_hsv(rr / 255, gg / 255, bb / 255)
                    if s > 0.3 and v > 0.3: hues.append(h)

            if hues:
                avg_h = sum(hues) / len(hues)
                if 0.12 <= avg_h <= 0.22:
                    hz = T_YELLOW
                elif 0.25 <= avg_h <= 0.45:
                    hz = T_GREEN
                elif 0.70 <= avg_h <= 0.90:
                    hz = T_PURPLE
                else:
                    hz = DEATH_PIT
                mat[r * 2, c * 2] = hz
    return mat


# ================================================================
# 3. Fire Orbit Logic (Exact Pixels, TRUE Clockwise)
# ================================================================
def build_fire_orbits(mat):
    fires = [(c, r) for r in range(0, MAT_SIZE, 2) for c in range(0, MAT_SIZE, 2) if mat[r, c] == DEATH_PIT]
    clusters = []

    for fx, fy in fires:
        placed = False
        for cl in clusters:
            if any(math.hypot(fx - px, fy - py) <= 8 for px, py in cl["pts"]):
                cl["pts"].append((fx, fy))
                placed = True;
                break
        if not placed: clusters.append({"pts": [(fx, fy)]})

    orbits = []
    for cl in clusters:
        pts = cl["pts"]
        if len(pts) <= 2:
            px, py = pts[0]
        else:
            max_d = -1
            t1, t2 = pts[0], pts[0]
            for p1 in pts:
                for p2 in pts:
                    d = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                    if d > max_d: max_d = d; t1, t2 = p1, p2

            pivot = max(pts,
                        key=lambda p: math.hypot(p[0] - t1[0], p[1] - t1[1]) + math.hypot(p[0] - t2[0], p[1] - t2[1]))
            px, py = pivot

        mems = [(x - px, y - py) for x, y in pts]

        # Replace original DEATH_PIT with FIRE in the environment matrix
        for x, y in pts:
            mat[y, x] = FIRE

        orbits.append({"px": px, "py": py, "mems": mems})
    return orbits


def rotate_fires(mat, orbits):
    for r in range(0, MAT_SIZE, 2):
        for c in range(0, MAT_SIZE, 2):
            if mat[r, c] == FIRE: mat[r, c] = EMPTY

    # TRUE CW Rotation Sequence: V -> > -> ^ -> <
    for o in orbits:
        new_mems = []
        for dx, dy in o["mems"]:
            nx, ny = dy, -dx  # Clockwise
            new_mems.append((nx, ny))
            fx, fy = o["px"] + nx, o["py"] + ny
            if 0 <= fx < MAT_SIZE and 0 <= fy < MAT_SIZE:
                mat[int(fy), int(fx)] = FIRE
        o["mems"] = new_mems


# ================================================================
# 4. Compliant Maze Environment
# ================================================================
class MazeEnvironment:
    def __init__(self, path):
        self.mat = load_maze(path)
        self.orbits = build_fire_orbits(self.mat)
        self.start, self.goal = (32, 0), (32, 63)
        self.pos = self.start
        self.pending_respawn = False

    def reset(self) -> Tuple[int, int]:
        self.pos = self.start
        self.pending_respawn = False
        return self.pos

    def step(self, actions: List[Action]) -> TurnResult:
        res = TurnResult()

        if self.pending_respawn:
            self.pos = self.start
            self.pending_respawn = False

        for act in actions:
            if act == Action.WAIT:
                res.actions_executed += 1
                continue

            dx, dy = \
            {Action.MOVE_UP: (0, -1), Action.MOVE_DOWN: (0, 1), Action.MOVE_LEFT: (-1, 0), Action.MOVE_RIGHT: (1, 0)}[
                act]
            nx, ny = self.pos[0] + dx, self.pos[1] + dy

            if 0 <= nx < 64 and 0 <= ny < 64:
                if self.mat[self.pos[1] * 2 + dy, self.pos[0] * 2 + dx] == WALL:
                    res.wall_hits += 1
                else:
                    self.pos = (nx, ny)
                    hz = self.mat[ny * 2, nx * 2]
                    if hz == DEATH_PIT or hz == FIRE:
                        res.is_dead = True
                        res.actions_executed += 1
                        res.current_position = (nx, ny)
                        self.pending_respawn = True
                        break

                    if self.pos == self.goal:
                        res.is_goal_reached = True
                        res.actions_executed += 1
                        res.current_position = self.pos
                        break
            else:
                res.wall_hits += 1
            res.actions_executed += 1

        if not res.is_dead and not res.is_goal_reached:
            res.current_position = self.pos

        rotate_fires(self.mat, self.orbits)
        return res


# ================================================================
# 5. GA Solver & Blind Agent Implementation
# ================================================================
GA_POP, GA_GENS, GA_LEN = 100, 20, 60


def simulate(mat, start, goal, chrom, heatmap):
    x, y = start
    path = [(x, y)]
    visited = {(x, y)}
    pen = 0
    for move in chrom:
        if move == Action.WAIT:
            pen += 100  # Increased wait penalty
            continue
        dx, dy = \
        {Action.MOVE_UP: (0, -1), Action.MOVE_DOWN: (0, 1), Action.MOVE_LEFT: (-1, 0), Action.MOVE_RIGHT: (1, 0)}[move]
        nx, ny = x + dx, y + dy
        if 0 <= nx < 64 and 0 <= ny < 64:
            if mat[y * 2 + dy, x * 2 + dx] != WALL:
                x, y = nx, ny
                path.append((x, y))

                if (x, y) in visited:
                    pen += 200
                visited.add((x, y))

                # BOREDOM PENALTY: Drastically increased to force exploration over cowering
                pen += heatmap[y, x] * 200

                hz = mat[y * 2, x * 2]
                if hz == DEATH_PIT:
                    pen += 2000  # INSTANT BREAK
                    break
                elif hz == FIRE:
                    pen += 100  # DRASTICALLY REDUCED. Agent is willing to risk it now.

                if (x, y) == goal:
                    pen -= 5000  # MASSIVE BONUS for reaching goal
                    break
            else:
                pen += 20

    dist_to_goal = abs(x - goal[0]) + abs(y - goal[1])
    dist_from_start = abs(x - start[0]) + abs(y - start[1])

    # DRASTICALLY increased goal gravity
    score = (dist_to_goal * 50.0) - (dist_from_start * 10.0) + pen
    return score, path


def solve(mat, start, goal, heatmap):
    valid_moves = [Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.WAIT]
    pop = [[random.choice(valid_moves) for _ in range(GA_LEN)] for _ in range(GA_POP)]
    best_chrom, best_score = pop[0], float('inf')

    for gen in range(GA_GENS):
        scored = sorted([(c, simulate(mat, start, goal, c, heatmap)[0]) for c in pop], key=lambda x: x[1])
        if scored[0][1] < best_score:
            best_chrom, best_score = scored[0]

        new_pop = [c for c, s in scored[:10]]
        while len(new_pop) < GA_POP:
            p1, p2 = random.choice(scored[:20])[0], random.choice(scored[:20])[0]
            cp = random.randint(1, GA_LEN - 1)
            child = p1[:cp] + p2[cp:]
            if random.random() < 0.1: child[random.randint(0, GA_LEN - 1)] = random.choice(valid_moves)
            new_pop.append(child)
        pop = new_pop
    return best_chrom


class GAAgent(Agent):
    def __init__(self, start, goal):
        super().__init__()
        self.start = start
        self.goal = goal
        self.known_mat = np.zeros((MAT_SIZE, MAT_SIZE), dtype=np.uint8)
        self.heatmap = np.zeros((64, 64), dtype=int)
        self.pos = start
        self.queue = []
        self.last_act = None
        self.was_fire_target = False

    def plan_turn(self, last_result: TurnResult) -> List[Action]:
        if last_result:
            if last_result.is_dead:
                dx, dy = last_result.current_position
                if not self.was_fire_target:
                    self.known_mat[dy * 2, dx * 2] = DEATH_PIT
                self.pos = self.start
                self.queue.clear()
            else:
                if last_result.wall_hits > 0 and self.last_act and self.last_act != Action.WAIT:
                    dx, dy = {Action.MOVE_UP: (0, -1), Action.MOVE_DOWN: (0, 1), Action.MOVE_LEFT: (-1, 0),
                              Action.MOVE_RIGHT: (1, 0)}[self.last_act]
                    self.known_mat[self.pos[1] * 2 + dy, self.pos[0] * 2 + dx] = WALL
                    self.queue.clear()
                else:
                    self.pos = last_result.current_position

        self.heatmap[self.pos[1], self.pos[0]] += 1

        if not self.queue:
            self.queue = solve(self.known_mat, self.pos, self.goal, self.heatmap)

        act = self.queue.pop(0) if self.queue else Action.WAIT
        self.last_act = act

        if act != Action.WAIT:
            dx, dy = \
            {Action.MOVE_UP: (0, -1), Action.MOVE_DOWN: (0, 1), Action.MOVE_LEFT: (-1, 0), Action.MOVE_RIGHT: (1, 0)}[
                act]
            target_y, target_x = self.pos[1] + dy, self.pos[0] + dx
            if 0 <= target_x < 64 and 0 <= target_y < 64:
                self.was_fire_target = (self.known_mat[target_y * 2, target_x * 2] == FIRE)
            else:
                self.was_fire_target = False
        else:
            self.was_fire_target = (self.known_mat[self.pos[1] * 2, self.pos[0] * 2] == FIRE)

        return [act]


# ================================================================
# 6. Visualizer App
# ================================================================
class App:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.root = tk.Tk()
        self.root.title("GA Maze Solver - Blind Agent")
        self.canvas = tk.Canvas(self.root, width=512, height=512, bg="white")
        self.canvas.pack()
        self.path = []
        threading.Thread(target=self.loop, daemon=True).start()
        self.draw()
        self.root.mainloop()

    def loop(self):
        last_res = None
        while True:
            self.agent.known_mat[self.agent.known_mat == FIRE] = EMPTY
            self.agent.known_mat[self.env.mat == FIRE] = FIRE

            actions = self.agent.plan_turn(last_res)
            last_res = self.env.step(actions)

            self.path = \
            simulate(self.agent.known_mat, self.agent.pos, self.agent.goal, self.agent.queue, self.agent.heatmap)[1]

            if last_res.is_goal_reached:
                print("Goal Reached!")
                break

            time.sleep(0.01)

    def draw(self):
        self.canvas.delete("all")
        px = 8

        for x, y in self.path:
            self.canvas.create_rectangle(x * px, y * px, x * px + px, y * px + px, fill="#ffb3c6", outline="")

        for r in range(NUM_CELLS):
            for c in range(NUM_CELLS):
                t = self.env.mat[r * 2, c * 2]
                if t == DEATH_PIT:
                    self.canvas.create_rectangle(c * px, r * px, c * px + px, r * px + px, fill="red", outline="")
                elif t == FIRE:
                    self.canvas.create_rectangle(c * px, r * px, c * px + px, r * px + px, fill="orange", outline="")
                elif t == T_GREEN:
                    self.canvas.create_rectangle(c * px, r * px, c * px + px, r * px + px, fill="green", outline="")
                elif t == T_YELLOW:
                    self.canvas.create_rectangle(c * px, r * px, c * px + px, r * px + px, fill="gold", outline="")
                elif t == T_PURPLE:
                    self.canvas.create_rectangle(c * px, r * px, c * px + px, r * px + px, fill="purple", outline="")

                if r < 63 and self.env.mat[r * 2 + 1, c * 2] == WALL:
                    self.canvas.create_line(c * px, r * px + px, c * px + px, r * px + px, fill="black", width=2)
                if c < 63 and self.env.mat[r * 2, c * 2 + 1] == WALL:
                    self.canvas.create_line(c * px + px, r * px, c * px + px, r * px + px, fill="black", width=2)

        gx, gy = self.env.goal
        self.canvas.create_oval(gx * px, gy * px, gx * px + px, gy * px + px, fill="blue")
        ax, ay = self.env.pos
        self.canvas.create_oval(ax * px + 1, ay * px + 1, ax * px + px - 1, ay * px + px - 1, fill="cyan")

        self.root.after(100, self.draw)


if __name__ == "__main__":
    environment = MazeEnvironment("mazepicUSE.png")
    blind_agent = GAAgent(start=(32, 0), goal=(32, 63))
    App(environment, blind_agent)
