from PIL import Image, ImageDraw
import numpy as np
from collections import deque

BORDER    = 2
CELL_SIZE = 14 #how wide and tall each cell is in the loaded image
WALL_SIZE = 2 #how thick the wall lines are
STRIDE    = 16 #total pixels per cell is CELL_SIZE + WALL_SIZE

NUM_CELLS   = 64 #maze is 64x64 cells
MATRIX_SIZE = 128 #stored as a 128x128 matrix

def is_wall_below(pixels, row, col):
    """
    Check if there is a wall between the cell (row, col) and the cell below it.
    """
    wall_start_y = BORDER + row * STRIDE + CELL_SIZE #wall pixels start right after cell ends
    sample_x     = BORDER + col * STRIDE + CELL_SIZE // 2

    top_pixel    = pixels[wall_start_y,     sample_x]
    bottom_pixel = pixels[wall_start_y + 1, sample_x]

    return not (top_pixel > 128 and bottom_pixel > 128) #if both pixels are light, passage is open

def is_wall_to_right(pixels, row, col):
    """
    Check if there is a wall between the cell (row, col) and the cell to its right.
    """
    wall_start_x = BORDER + col * STRIDE + CELL_SIZE
    sample_y     = BORDER + row * STRIDE + CELL_SIZE // 2

    left_pixel  = pixels[sample_y, wall_start_x    ]
    right_pixel = pixels[sample_y, wall_start_x + 1]

    return not (left_pixel > 128 and right_pixel > 128)

def load_maze(image_path):
    """
    Reads maze PNG and returns a 128x128 numpy matrix where:
      0 = open space
      1 = wall
    """
    img = Image.open(image_path).convert("L")  #open as grayscale
    pixels = np.array(img)

    height, width = pixels.shape
    if height != 1026 or width != 1026:
        raise ValueError(f"Expected a 1026x1026 image but got {width}x{height}.")

    #start with everything as walls, then open up passages found on the way
    matrix = np.ones((MATRIX_SIZE, MATRIX_SIZE), dtype=np.uint8)

    for row in range(NUM_CELLS):
        for col in range(NUM_CELLS):
            matrix[row * 2, col * 2] = 0 #every actual cell is always open

            if row < NUM_CELLS - 1: #check if there's a row below
                if is_wall_below(pixels, row, col):
                    matrix[row * 2 + 1, col * 2] = 1  #wall
                else:
                    matrix[row * 2 + 1, col * 2] = 0  #open passage

            if col < NUM_CELLS - 1: #check if there's a column to the right
                if is_wall_to_right(pixels, row, col):
                    matrix[row * 2, col * 2 + 1] = 1
                else:
                    matrix[row * 2, col * 2 + 1] = 0

            if row < NUM_CELLS - 1 and col < NUM_CELLS - 1: #corner slot between 4 cells is always a wall
                matrix[row * 2 + 1, col * 2 + 1] = 1

    return matrix

def find_start_and_goal(image_path):
    """
    Scan the top and bottom borders of the maze image for openings (white pixels).
    This works for any maze image rather than assuming a fixed start/goal position.
    """
    img = Image.open(image_path).convert("L")
    pixels = np.array(img)

    #scan just inside the top and bottom borders for white/open pixels
    top_open = [col for col in range(pixels.shape[1]) if pixels[1,  col] > 128]
    bottom_open = [col for col in range(pixels.shape[1]) if pixels[-2, col] > 128]

    if not top_open or not bottom_open:
        raise ValueError("Could not find openings in the maze border")

    #take the middle pixel of each opening and convert it to a matrix column
    top_col    = (top_open[len(top_open) // 2] - BORDER) // STRIDE * 2
    bottom_col = (bottom_open[len(bottom_open) // 2] - BORDER) // STRIDE * 2

    start = (0, top_col)
    goal  = (NUM_CELLS * 2 - 2, bottom_col)

    print(f"  Start found at matrix position {start}")
    print(f"  Goal found at matrix position  {goal}")

    return start, goal

def solve_maze(matrix, start, goal):
    """
    Find the shortest path from start to goal using BFS.
    BFS explores all positions one step away, then two steps away,
    and so on. This guarantees the first time we reach the goal is through the shortest path.
    """
    queue = deque([start]) #queue holds positions needed to explore

    came_from = {start: None} #remembers how we got to each position visited in order to retrace path at the end

    while queue:
        current = queue.popleft()

        if current == goal:
            return rebuild_path(came_from, goal)

        row, col = current
        neighbors = [
            (row - 1, col), #up
            (row + 1, col), #down
            (row, col - 1), #left
            (row, col + 1), #right
        ]

        for next_pos in neighbors:
            next_row, next_col = next_pos

            if not (0 <= next_row < MATRIX_SIZE and 0 <= next_col < MATRIX_SIZE): #skips if out of bounds
                continue

            if next_pos in came_from:  #skips if already visited
                continue

            if matrix[next_row, next_col] != 0: #skips walls and hazards
                continue

            came_from[next_pos] = current #adds valid moves to queue
            queue.append(next_pos)

    return None #no path has been found

def rebuild_path(came_from, goal):
    """
    Walk backwards through the came_from map to reconstruct the path from start to goal.
    """
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()  #flips it so it goes from start to goal
    return path

def save_matrix_as_image(matrix, output_path, scale=8):
    """
    Save the 128x128 matrix as an image so we can visually verify it loaded correctly.
    Each matrix cell is drawn as a small square (scale x scale pixels).
      White is an open space
      Black is the wall
    """
    color_map = {
        0: (255, 255, 255),  #open path is white
        1: (0,   0,   0  ),  #wall is black
        2: (255, 140, 0  ),  #hazard color
    }

    img_size = MATRIX_SIZE * scale
    img = Image.new("RGB", (img_size, img_size), "white")
    draw = ImageDraw.Draw(img)

    for row in range(MATRIX_SIZE):
        for col in range(MATRIX_SIZE):
            cell_value = matrix[row, col]
            color = color_map.get(cell_value, (200, 0, 200))  #purple is an unknown value

            x1 = col * scale
            y1 = row * scale
            x2 = x1 + scale - 1
            y2 = y1 + scale - 1
            draw.rectangle([x1, y1, x2, y2], fill=color)

    img.save(output_path)
    print(f"Matrix preview saved to: {output_path}")

def save_solution_on_original(original_image_path, path, output_path):
    """
    Draw the solution path as a red line on top of the original maze image. Convert each position in matrix coordinates back to pixel coordinates before drawing.
    """
    img  = Image.open(original_image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    def matrix_pos_to_pixel(matrix_row, matrix_col):
        #matrix position is the logical cell position, then pixel coordinates at the center of that spot
        pixel_y = BORDER + (matrix_row / 2) * STRIDE + CELL_SIZE / 2
        pixel_x = BORDER + (matrix_col / 2) * STRIDE + CELL_SIZE / 2
        return (pixel_x, pixel_y)

    pixel_path = [matrix_pos_to_pixel(r, c) for r, c in path] #converts all path positions to pixels

    draw.line(pixel_path, fill=(220, 50, 50), width=4) #draws a red line for the solution

    def draw_dot(matrix_row, matrix_col, color): #green dot at the start and blue dot at the goal
        x, y = matrix_pos_to_pixel(matrix_row, matrix_col)
        draw.ellipse([x - 7, y - 7, x + 7, y + 7], fill=color)

    draw_dot(*path[0],  color=(0, 200, 0))
    draw_dot(*path[-1], color=(0, 80, 220))

    img.save(output_path)
    print(f"Solved maze image saved to: {output_path}")


IMAGE_PATH = "MAZE_0.png"

print("Loading maze...")
matrix = load_maze(IMAGE_PATH)

open_count = int((matrix == 0).sum())
wall_count = int((matrix == 1).sum())
print(f"  Open spaces : {open_count}")
print(f"  Walls       : {wall_count}")

print("\nTop-left corner of the matrix (16x16 preview):")
for row in matrix[:16, :16]:
    print("  " + " ".join(str(v) for v in row))

print("\nFinding start and goal...")
START, GOAL = find_start_and_goal(IMAGE_PATH)

print(f"\nSolving from {START} to {GOAL}...")
path = solve_maze(matrix, START, GOAL)

if path is None:
    print("No solution found!")
else:
    cell_visits = sum(1 for r, c in path if r % 2 == 0 and c % 2 == 0)
    print(f"Solution found! Path length: {cell_visits} cells")

print("\nSaving images...")
save_matrix_as_image(matrix, "maze_preview.png")
Image.open("maze_preview.png").show()

if path:
    save_solution_on_original(IMAGE_PATH, path, "maze_solved.png")
    Image.open("maze_solved.png").show()