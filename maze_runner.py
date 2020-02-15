import copy

from cell import Cell
from queue import PriorityQueue
from prioritizeditem import PrioritizedItem
import math
import random
from queue import Queue
from queue import LifoQueue
import time
import sys
from multiprocessing import Process
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

# TODO: Create Hard Maze
# TODO: Fire Maze Own Strategy
# TODO: Plot Data for heuristics
# TODO: improved DFS

# Colors
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
BLACK = (0, 0, 0)
RED = (255, 0, 00)
GREEN = (0, 250, 0)
BLUE = (0, 0, 255)

sys.setrecursionlimit(10000)
board = []
movement_symbol = "\u2705"
# movement_symbol = ">"
block_symbol = "\u274C"
# block_symbol = "x"
fire_symbol = "\U0001F525"
# fire_symbol = "#"
optimal_dim = 60
optimal_p = 0.225
p0 = 0.3
optimal_q = .1
p0 = .3

pygame.init()
width = 8
screen_size = (optimal_dim * width + 10, optimal_dim * width +10 )
screen = pygame.display.set_mode(screen_size)
# screen.fill(BLUE)
# pygame.display.flip()


max_fringe_size = 0
num_nodes_explored = 0
nodes_explored = []
fire_locations = []

def create_maze(dim, p):
    global board
    board = []
    for row in range(dim):
        board.append([])
        for col in range(dim):
            board[row].append(Cell(row, col))
            if (row == 0 and col == 0) or (row == dim - 1 and col == dim - 1):
                continue
            else:
                if random.random() < p:
                    board[row][col].set_block_status(True)
    assign_board_neighbors(dim)


def create_hard_maze(dim, p):
    return None


def assign_board_neighbors(dim):
    for row in range(dim):
        for col in range(dim):
            if row != 0:
                if not board[row - 1][col].is_blocked:
                    board[row][col].add_neighbor(board[row - 1][col])
            if row != (dim - 1):
                if not board[row + 1][col].is_blocked:
                    board[row][col].add_neighbor(board[row + 1][col])
            if col != 0:
                if not board[row][col - 1].is_blocked:
                    board[row][col].add_neighbor(board[row][col - 1])
            if col != (dim - 1):
                if not board[row][col + 1].is_blocked:
                    board[row][col].add_neighbor(board[row][col + 1])


def dfs(start,goal):
    fringe = LifoQueue(-1)
    discovered = [start]
    backward_mapping = dict()
    fringe.put(start)

    while not fringe.empty():
        current = fringe.get()
        if current == goal:
            return back_track(backward_mapping, start, current)
        for neighbor in current.neighbors:
            if neighbor not in discovered:
                discovered.append(neighbor)
                backward_mapping[neighbor] = current
                fringe.put(neighbor)


def bfs(start, goal):
    fringe = Queue(-1)
    discovered = [start]
    backward_mapping = dict()
    fringe.put(start)

    while not fringe.empty():
        current = fringe.get()
        if current == goal:
            return back_track(backward_mapping, start, current)
        for neighbor in current.neighbors:
            if neighbor not in discovered:
                discovered.append(neighbor)
                backward_mapping[neighbor] = current
                fringe.put(neighbor)


def back_track(backward_mapping, start, current):
    path = [current]
    while current != start:
        current = backward_mapping[current]
        path.insert(0, current)
    return path


def astar(start, goal, hFunc):
    global max_fringe_size
    global num_nodes_explored
    global nodes_explored
    fringeq = PriorityQueue(-1)
    backward_mapping = dict()

    gscores = dict()
    gscores[start] = 0
    fscores = dict()
    fscores[start] = gscores[start] + hFunc(start.row, start.col, goal.row, goal.col)

    fringeq.put(PrioritizedItem(fscores[start], start))
    while not fringeq.empty():
        current = fringeq.get().item
        nodes_explored.append(current)
        num_nodes_explored += 1
        if current == goal:
            return back_track(backward_mapping, start, current)
        for neighbor in current.neighbors:
            if neighbor.on_fire:
                continue
            try:
                gscores[neighbor]
            except KeyError:
                gscores[neighbor] = math.inf
            recalc_g = gscores[current] + 1
            if recalc_g < gscores[neighbor]:
                backward_mapping[neighbor] = current
                gscores[neighbor] = recalc_g
                fscores[neighbor] = gscores[neighbor] + hFunc(neighbor.row, neighbor.col, goal.row, goal.col)

                for node in fringeq.queue:
                    if node.item == neighbor:
                        fringeq.queue.remove(node)
                        break
                fringeq.put(PrioritizedItem(fscores[neighbor], neighbor))
                if(fringeq.qsize() > max_fringe_size):
                    max_fringe_size = fringeq.qsize()
    return None


def euclidean_dist(x1, y1, x2, y2):
    return math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))


def manhattan_dist(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def bfsBD(start, goal):
    global nodes_explored
    fringe_front = Queue(-1)
    fringe_back = Queue(-1)
    discovered_front = [start]
    discovered_back = [goal]
    backward_mapping = dict()
    forward_mapping = dict()
    fringe_front.put(start)
    fringe_back.put(goal)

    while not (fringe_front.empty() or fringe_back.empty()):
        if not fringe_front.empty():
            current_front = fringe_front.get()
            nodes_explored.append(current_front)
            for neighbor in current_front.neighbors:
                if neighbor not in discovered_front:
                    discovered_front.append(neighbor)
                    backward_mapping[neighbor] = current_front
                    fringe_front.put(neighbor)
            intersect = intersection(discovered_front, discovered_back)
            if intersect is not None:
                disc = intersect[0]
                path_front_to_intersection = back_track(backward_mapping, start, disc)
                path_back_to_intersection = back_track(forward_mapping, goal, disc)
                path_back_to_intersection.reverse()
                path_back_to_intersection.remove(intersect[0])
                return path_front_to_intersection + path_back_to_intersection
        if not fringe_back.empty():
            current_back = fringe_back.get()
            nodes_explored.append(current_back)
            for neighbor in current_back.neighbors:
                if neighbor not in discovered_back:
                    discovered_back.append(neighbor)
                    forward_mapping[neighbor] = current_back
                    fringe_back.put(neighbor)
            intersect = intersection(discovered_front, discovered_back)
            if intersect is not None:
                disc = intersect[0]
                path_front_to_intersection = back_track(backward_mapping, start, disc)
                path_back_to_intersection = back_track(forward_mapping, goal, disc)
                path_back_to_intersection.reverse()
                path_back_to_intersection.remove(intersect[0])
                return path_front_to_intersection + path_back_to_intersection
    return None


def intersection(list1, list2):
    list2_as_set = set(list2)
    intersect = [value for value in list1 if value in list2_as_set]
    if len(intersect) == 0:
        return None
    return intersect

def create_fire_maze():
    global fire_locations
    fire_locations.clear()
    while True:
        create_maze(optimal_dim, optimal_p)
        start = board[0][0]
        goal = board[optimal_dim - 1][optimal_dim - 1]
        fire_loc = (random.randint(0, optimal_dim - 1), random.randint(0, optimal_dim - 1))
        if board[fire_loc[0]][fire_loc[1]].is_blocked or fire_loc == (0, 0) \
                or fire_loc == (optimal_dim - 1, optimal_dim - 1):
            continue
        else:
            if astar(start, board[fire_loc[0]][fire_loc[1]], manhattan_dist) is None \
                    or astar(start, goal, manhattan_dist) is None:
                continue
            board[fire_loc[0]][fire_loc[1]].set_fire_status(True)

        path = astar(start, goal, manhattan_dist)

        if path is None:
            continue
        fire_locations.append((fire_loc[0], fire_loc[1]))
        return path


def fire_strat_1(q, num_tests, display):
    pygame.display.set_caption('Fire Strategy 1')
    global board
    fail_counter = 0
    for i in range(num_tests):
        print("\r Running Test " + str(i), end="")
        path = create_fire_maze()
        goal = board[len(board) - 1][len(board) - 1]
        for cell in path:
            if cell.on_fire:
                fail_counter += 1
                break
            compute_fire_movement(q)
            if display:
                draw_maze([cell])
                pygame.event.get()
        board = []
    print("\r", end="")
    return 100 - ((fail_counter / num_tests) * 100)


def fire_strat_2(q, num_tests):
    pygame.display.set_caption('Fire Strategy 2')
    fail_counter = 0
    for i in range(num_tests):
        print("\r Running Test " + str(i), end="")
        path = create_fire_maze()
        goal = board[len(board) - 1][len(board) - 1]
        while True:
            if path is None or path[0].on_fire:
                fail_counter += 1
                break
            elif path[1] == goal:
                break
            compute_fire_movement(q)
            draw_maze([path[0]])
            pygame.event.get()
            if path[1].on_fire:
                sys.exit(1)
            path = astar(path[1], goal, euclidean_dist)
    print("\r", end="")
    return 100 - ((fail_counter / num_tests) * 100)


# Multi Process version To go quicker but does not display the maze!!
def fire_strat_2_multi_proc(q, num_tests):
    fail_counter = 0

    processes = []
    # print("starting ", end="")
    for i in range(num_tests):
        print("\r Running Test " + str(i), end="")
        processes.append(Process(target=fire_strat_2_helper, args=(q,)))
        processes[i].start()
        # print("\r" + str(i), end="")
    # print("\nJoining ", end="")
    for p in processes:
        p.join()
        # print("#", end="")
        # print(p.exitcode)
        if p.exitcode == 1:
            fail_counter += 1
    # print("Fail Counter: " + str(fail_counter))
    print("\r", end="")
    return 100 - ((fail_counter / num_tests) * 100)


def fire_strat_2_helper(q):
    path = create_fire_maze()
    goal = board[len(board) - 1][len(board) - 1]
    while True:
        if path is None or path[1].on_fire:
            sys.exit(1)
            # return
        if path[1] == goal:
            sys.exit(0)
            # return
        compute_fire_movement(q)
        if path[1].on_fire:
            sys.exit(1)
        path = astar(path[1], goal, euclidean_dist)


def fire_strat_custom(q, num_tests, steps_ahead):
    pygame.display.set_caption('Fire Strategy Custom')
    fail_counter = 0
    for i in range(num_tests):
        print("\r Running Test " + str(i), end="")
        create_fire_maze()
        path = fire_search(board[0][0], board[optimal_dim - 1][optimal_dim - 1])
        goal = board[len(board) - 1][len(board) - 1]
        while True:
            if path is None or path[0].on_fire:
                fail_counter += 1
                break

            current = path[0]
            draw_maze([current])
            pygame.event.get()

            if path[1] == goal:
                break
            compute_fire_movement(q)
            if fire_distance(current) < 20:
                fire_reset = []
                step_length = []
                for j in range(steps_ahead):
                    temp = compute_fire_movement(q)
                    step_length.append(len(temp))
                    fire_reset = fire_reset + temp

                while True:
                    path = astar(current, goal, euclidean_dist)
                    if path is not None:
                        for cell in fire_reset:
                            cell.on_fire = False
                        break
                    if len(fire_reset) == 0:
                        break
                    if path is None:
                        fire_reset[len(fire_reset) - 1].on_fire = False
                        fire_locations.remove(
                            (fire_reset[len(fire_reset) - 1].row, fire_reset[len(fire_reset) - 1].col))
                        fire_reset.pop(len(fire_reset) - 1)
            if path is not None:
                path = path[1:]
    print("\r", end="")
    return 100 - ((fail_counter / num_tests) * 100)

# Multi Process version To go quicker but does not display the maze!!
def fire_strat_custom_multi_proc(q, num_tests,steps_ahead):
    fail_counter = 0

    processes = []
    for i in range(num_tests):
        processes.append(Process(target=fire_strat_custom_helper, args=(q,steps_ahead)))
        processes[i].start()
    for p in processes:
        p.join()
        if p.exitcode == 1:
            fail_counter += 1
    return 100 - ((fail_counter / num_tests) * 100)


def fire_strat_custom_helper(q, steps_ahead):
    path = create_fire_maze()
    goal = board[len(board) - 1][len(board) - 1]
    while True:
        if path is None or path[1].on_fire:
            sys.exit(1)
            # return
        current = path[1]
        if current == goal:
            sys.exit(0)
            # return
        # draw_maze(path)
        # pygame.event.get()
        compute_fire_movement(q)
        if fire_distance(current) < 2:
            fire_steps = []
            for i in range(steps_ahead):
                fire_steps.append(compute_fire_movement(q))
            path = astar(current, goal, euclidean_dist)
            if path is None and len(fire_steps) != 0:
                reset_fire_prediction(fire_steps[-1])
                del fire_steps[-1]
                path = astar(current, goal, euclidean_dist)
        else:
            path = astar(current, goal, euclidean_dist)


def fire_search(start, goal):
    fringe = PriorityQueue(-1)
    discovered = [start]
    backward_mapping = dict()
    score = (10 * euclidean_dist(start.row, start.col, goal.row, goal.col)) #+ (fire_distance(start) * -.5)
    fringe.put(PrioritizedItem(score, start))

    if goal.on_fire:
        return None
    if astar(start, goal, euclidean_dist) is None:
        return None
    while not fringe.empty():
        current = fringe.get().item
        if current == goal:
            return back_track(backward_mapping, start, current)
        for neighbor in current.neighbors:
            if neighbor not in discovered and not neighbor.on_fire:
                discovered.append(neighbor)
                backward_mapping[neighbor] = current
                score = (10 * euclidean_dist(start.row, start.col, goal.row, goal.col)) #+ (fire_distance(start) * -.5)
                fringe.put(PrioritizedItem(score, neighbor))
    return None

def fire_distance(start):
    global fire_locations
    m = manhattan_dist(start.row, start.col, fire_locations[0][0], fire_locations[0][1])
    for locs in fire_locations:
        if manhattan_dist(start.row, start.col, locs[0], locs[1]) < m:
            m = manhattan_dist(start.row, start.col, locs[0], locs[1])
    return m

def reset_fire_prediction(step):
    for cell in step:
        cell.on_fire = False

def compute_fire_movement(q):
    new_on_fire = []
    for row in board:
        for cell in row:
            if cell.on_fire:
                continue
            on_fire_count = 0
            for neighbor in cell.neighbors:
                if neighbor.on_fire:
                    on_fire_count += 1
            if on_fire_count >= 1:
                p = 1 - ((1 - q) ** on_fire_count)
                if random.random() < p:
                    cell.set_fire_status(True)
                    fire_locations.append((cell.row, cell.col))
                    # fire_locations.sort(key=lambda x: x[0])
                    new_on_fire.append(cell)
    return new_on_fire

def print_maze(path):
    print("  \t", end='')
    for i in range(len(board)):
        print("\t" + str(i) + "\t", end='')
    print()
    for row in range(len(board)):
        print("    ", end='')
        for i in range(len(board)):
            print("________", end='')
        print()
        for col in range(len(board[0]) + 1):
            if col == 0:
                print(str(row) + ":\t|", end='')
            else:
                print("|", end='')
            if col < len(board[0]) and board[row][col] in path:
                print("\t" + movement_symbol + "\t", end='')
            else:
                if col < len(board[0]) and board[row][col].is_blocked:
                    print("\t" + block_symbol + "\t", end='')
                elif col < len(board[0]) and board[row][col].on_fire:
                    print("\t" + fire_symbol + "\t", end='')
                else:
                    print("\t \t", end='')
        print()
    print("    ", end='')
    for i in range(len(board)):
        print("________", end='')
    print()


def print_maze_nopath():
    print("  \t", end='')
    for i in range(len(board)):
        print("\t" + str(i) + "\t", end='')
    print()
    for row in range(len(board)):
        print("    ", end='')
        for i in range(len(board)):
            print("________", end='')
        print()
        for col in range(len(board[0]) + 1):
            if col == 0:
                print(str(row) + ":\t|", end='')
            else:
                print("|", end='')
            if col < len(board[0]) and board[row][col].is_blocked:
                print("\t" + block_symbol + "\t", end='')
            elif col < len(board[0]) and board[row][col].on_fire:
                print("\t" + fire_symbol + "\t", end='')
            else:
                print("\t \t", end='')
        print()
    print("    ", end='')
    for i in range(len(board)):
        print("________", end='')
    print()


def draw_maze(path):
    if path is None:
        return
    for row in board:
        for c in row:
            if c.is_blocked:
                pygame.draw.rect(screen, BLACK, (c.row * width, c.col * width, width, width))
            elif c.on_fire:
                pygame.draw.rect(screen, RED, (c.row * width, c.col * width, width, width))
            elif c in path:
                pygame.draw.rect(screen, GREEN, (c.row * width, c.col * width, width, width))
            else:
                pygame.draw.rect(screen, WHITE, (c.row * width, c.col * width, width, width))
    pygame.display.update()

if __name__ == '__main__':

    if input("\n\nWelcome to our Maze Runner! To run the algorithm we used to find our dim "
             "\npress the enter key press to continue or any other key followed by enter to continue\n") == "":
        # Find a map size (dim) that is large enough to produce maps that require some work to solve,
        # but small enough that you can run each algorithm multiple times for a range of possible p values.
        # How did you pick a dim? We picked Dim by looping through a bunch of different dims with a bunch of different
        # p values we picked our dim based on which dim took a decent amount of average time to complete for each p value
        # Optimal Value finder: We Selected -- Dim: 105 p: 0.225 Fail: 20.0 AVG Time: 0.1652039000000002
        results = []
        for dim in range(15, 70):
            if not dim % 5 == 0:
                continue
            for p in [.2, .225, .25, .275, .3]:
                fail_counter = 0
                tests = 50
                total_time = 0
                for i in range(tests):
                    # print(i)
                    create_maze(dim, p)
                    t0 = time.process_time()
                    if bfs(board[0][0], board[dim - 1][dim - 1]) is None:
                        fail_counter += 1
                    dfs(board[0][0], board[dim - 1][dim - 1])
                    astar(board[0][0], board[dim - 1][dim - 1], euclidean_dist)
                    astar(board[0][0], board[dim - 1][dim - 1], manhattan_dist)
                    bfsBD(board[0][0], board[dim - 1][dim - 1])
                    total_time += (time.process_time() - t0)
                avgtime = (total_time / tests / 5)
                percent = (fail_counter / tests) * 100
                print(
                    "Dim: " + str(dim) + " p: " + str(p) + " Fail: " + str(percent) + " AVG Time: " + str(avgtime))
                results.append((dim, p, percent, avgtime))
                if percent > 25:
                    break
        print("\n\n\n\n\n")
        #print(results)

    if input("\n\nTo generate shortest paths for dim = optimal (60), p = .2 "
             "\npress enter, press any other key followed by enter to continue. "
             "\nTo move to the next algorithm press the X to close the window. "
             "\nThe Title box of the screen will tell you which algorithm\n") == "":
        bfspath = None
        while True:
            create_maze(optimal_dim, optimal_p)
            bfspath = bfs(board[0][0], board[optimal_dim - 1][optimal_dim - 1])
            if bfspath is not None:
                break

        dfspath = dfs(board[0][0], board[optimal_dim-1][optimal_dim-1])
        astarEDpath = astar(board[0][0], board[optimal_dim-1][optimal_dim-1], euclidean_dist)
        astarMDpath = astar(board[0][0], board[optimal_dim-1][optimal_dim-1], manhattan_dist)
        bfsBDpath = bfsBD(board[0][0], board[optimal_dim-1][optimal_dim-1])

        print("BFS Path Length: " + str(len(bfspath)))
        print("DFS Path Length: " + str(len(dfspath)))
        print("A* ED Path Length: " + str(len(astarEDpath)))
        print("A* MD Path Length: " + str(len(astarMDpath)))
        print("BFSBD Path Length: " + str(len(bfsBDpath)))

        pygame.display.set_caption('BFS')
        running = True
        while running:
            draw_maze(bfspath)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        pygame.display.set_caption('DFS')
        running = True
        while running:
            draw_maze(dfspath)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        pygame.display.set_caption('Astar, Euclidean Dist')
        running = True
        while running:
            draw_maze(astarEDpath)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        pygame.display.set_caption('Astar, Manhattan Dist')
        running = True
        while running:
            draw_maze(astarMDpath)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        pygame.display.set_caption('Bidirectional BFS')
        running = True
        while running:
            draw_maze(bfsBDpath)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

    screen.fill(BLUE)
    pygame.display.flip()

    try:
        dim = int(input("\n\nTo run maze solvability depending on p enter the dim you would like to "
                              "run our algorithm with. Enter the Dim to start enter anything else to continue\n"))
        num_tests = 50
        p = 0
        while p <= 1:
            fail_counter = 0
            for i in range(num_tests):
                create_maze(dim, p)
                path = astar(board[0][0],board[dim-1][dim-1],euclidean_dist)
                if path is None:
                    fail_counter += 1
            print(str(p) + "\t" + str(100 - ((fail_counter/num_tests) *100)))
            p += .005
            p = round(p, 3)
    except ValueError:
        None


    try:
        dim = int(input("\n\nTo run path length vs maze density from [0,p0] enter the dim you would like to"
                        "run out algorithm with. Enter the Dim to start, enter anything else to continue \n"))
        num_tests = 50
        p = 0
        while p <= p0:
            length = 0
            for i in range(num_tests):
                path = None
                while path is None:
                    create_maze(dim, p)
                    path = astar(board[0][0],board[dim-1][dim-1],euclidean_dist)
                length += len(path)
            print(str(p) + "\t" + str(round(length/num_tests, 2)))
            p += .005
            p = round(p, 3)
    except ValueError:
        None

    if input("\n\nTo run our way of measuring which heuristic is better press enter,"
             "\notherwise press any other key followed by enter\n") == "":
        num_tests = 1000
        max_fringe_ed = 0
        max_fringe_md = 0
        total_time_ed = 0
        total_time_md = 0
        total_nodes_ed = 0
        total_nodes_md = 0
        average_max_fringe_ed = []
        average_time_ed = []
        average_total_nodes_ed = []
        average_max_fringe_md = []
        average_time_md = []
        average_total_nodes_md = []

        dim = 25
        while dim <= 100:
            for i in range(num_tests):
                print("\rRunning Test: "+str(i), end = "")
                create_maze(dim, optimal_p)

                t0 = time.process_time()
                astar(board[0][0], board[dim-1][dim-1],euclidean_dist)
                total_time_ed += (time.process_time() - t0)

                max_fringe_ed += max_fringe_size
                total_nodes_ed += num_nodes_explored

                max_fringe_size = 0
                num_nodes_explored = 0

                t0 = time.process_time()
                astar(board[0][0], board[dim - 1][dim - 1], manhattan_dist)
                total_time_md += (time.process_time() - t0)

                max_fringe_md += max_fringe_size
                total_nodes_md += num_nodes_explored

                max_fringe_size = 0
                num_nodes_explored = 0

            average_max_fringe_ed.append((dim,max_fringe_ed/num_tests))
            average_time_ed.append((dim,total_time_ed/num_tests))
            average_total_nodes_ed.append((dim,total_nodes_ed/num_tests))

            average_max_fringe_md.append((dim,max_fringe_md/num_tests))
            average_time_md.append((dim,total_time_md/num_tests))
            average_total_nodes_md.append((dim,total_nodes_md/num_tests))


            print("\rAverages for Dim: " + str(dim) +
                "\naverage max fringe size\n"
                "Euclidean Distance: " + str(max_fringe_ed/num_tests) + "\n"
                "Manhattan Distance: " + str(max_fringe_md/num_tests) + "\n"
                "\naverage time to complete\n"
                "Euclidean Distance: " + str(total_time_ed /num_tests) + "\n"
                "Manhattan Distance: " + str(total_time_md /num_tests) + "\n"
                "\naverage nodes visited\n"
                "Euclidean Distance: " + str(total_nodes_ed /num_tests) + "\n"
                "Manhattan Distance: " + str(total_nodes_md /num_tests) + "\n")
            dim += 25

        print("average max fringe for ED Data")
        for i in average_max_fringe_ed:
            print(str(i[0]) + "\t" + str(i[1]))

        print("average max fringe for MD Data")
        for i in average_max_fringe_md:
            print(str(i[0]) + "\t" + str(i[1]))

        print("average time for ED Data")
        for i in average_time_ed:
            print(str(i[0]) + "\t" + str(i[1]))

        print("average time for MD Data")
        for i in average_time_md:
            print(str(i[0]) + "\t" + str(i[1]))

        print("average nodes explored for ED Data")
        for i in average_total_nodes_ed:
            print(str(i[0]) + "\t" + str(i[1]))

        print("average nodes explored for MD Data")
        for i in average_total_nodes_md:
            print(str(i[0]) + "\t" + str(i[1]))

    screen.fill(BLUE)
    pygame.display.flip()

    if input("\n\nTo run our testing on if on the same map there are some nodes that BDBFS expands that A* does not"
             "Press enter, otherwise press any other key followed by enter\n") == "":
        num_tests = 100
        for i in range(num_tests):
            create_maze(optimal_dim,optimal_p)
            start = board[0][0]
            goal = board[optimal_dim-1][optimal_dim-1]

            astar_nodes_md = []
            bfsbd_nodes = []

            astar(start, goal, manhattan_dist)
            astar_nodes_md = copy.copy(nodes_explored)
            nodes_explored = []

            bfsBD(start, goal)
            bfsbd_nodes = copy.copy(nodes_explored)
            nodes_explored = []

            diff_bfs_MD = list(set(bfsbd_nodes) - set(astar_nodes_md))
            diff_MD_bfs = list(set(astar_nodes_md) - set(bfsbd_nodes))

            if len(diff_bfs_MD) != 0 or len(diff_MD_bfs) != 0:
                pygame.display.set_caption('astar nodes explored (MD)')
                running = True
                while running:
                    draw_maze(astar_nodes_md)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                pygame.display.set_caption('bfsbd nodes explored')
                running = True
                while running:
                    draw_maze(bfsbd_nodes)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                pygame.display.set_caption('Difference in nodes explored BFS - MD')
                running = True
                while running:
                    draw_maze(diff_bfs_MD)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                pygame.display.set_caption('Difference in nodes explored MD - BFS')
                running = True
                while running:
                    draw_maze(diff_MD_bfs)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                break


    q = 0.0
    q_increment = 0.025
    q_max = .5

    output = None
    try:
        num_tests = int(input("\n\nTo run fire Strategy 1 enter the number of times you want it to run "
                              "\nand press enter otherwise to continue hit any other key followed by enter \n"))


        if input("\nWould you like to display all the tests? (it runs significantly slower when displaying) "
                 "enter \"y\" for Yes and enter anything else for No") == "y":
            while q <= q_max:
                output = fire_strat_1(q, num_tests, True)
                print(str(q) + "\t" + str(output))
                q += q_increment
                q = round(q, 3)
        else:
            while q <= q_max:
                output = fire_strat_1(q, num_tests, False)
                print(str(q) + "\t" + str(output))
                q += q_increment
                q = round(q, 3)
    except ValueError:
        None

    screen.fill(BLUE)
    pygame.display.flip()

    q = 0.0
    try:
        num_tests = int(input("\n\nTo run fire Strategy 2 enter the number of times you want it to run "
                              "\nand press enter otherwise to continue hit any other key followed by enter \n"))


        if input("\nWould you like to display all the tests? (it runs significantly slower when displaying) "
                 "enter \"y\" for Yes and enter anything else for No\n") == "y":
            while q <= q_max:
                output = fire_strat_2(q, num_tests)
                print(str(q) + "\t" + str(output))
                q += q_increment
                q = round(q, 3)
        else:
            while q <= q_max:
                output = fire_strat_2_multi_proc(q, num_tests)
                print(str(q) + "\t" + str(output))
                q += q_increment
                q = round(q, 3)
    except ValueError:
        None

    screen.fill(BLUE)
    pygame.display.flip()

    q = 0.0
    try:
        num_tests = int(input("\n\nTo run custom fire Strategy enter the number of times you want it to run "
                              "\nand press enter otherwise to continue hit any other key followed by enter \n"))


        if input("\nWould you like to display all the tests? (it runs significantly slower when displaying) "
                 "enter \"y\" for Yes and enter anything else for No\n") == "y":
            while q <= q_max:
                output = fire_strat_custom(q, num_tests, 4)
                print(str(q) + "\t" + str(output))
                q += q_increment
                q = round(q, 3)
        else:
            while q <= q_max:
                output = fire_strat_custom_multi_proc(q, num_tests, 4)
                print(str(q) + "\t" + str(output))
                q += q_increment
                q = round(q, 3)
    except ValueError:
        None


    num_tests = int(input("enter num tests"))

    while q <= .325:
        output = fire_strat_2_multi_proc(q, num_tests)
        print(str(q) + "\t" + str(output))
        q += q_increment
        q = round(q, 3)

    q = 0.00
    while q <= .325:
        output = fire_strat_custom_multi_proc(q, num_tests, 3)
        print(str(q) + "\t" + str(output))
        q += q_increment
        q = round(q, 3)