from cell import Cell
from queue import PriorityQueue
from prioritizeditem import PrioritizedItem
import math
import random
from queue import Queue
import time
import sys
from multiprocessing import Process
from multiprocessing import Queue as msQueue
import pygame

#TODO: Create Hard Maze
#TODO: Fire Maze Own Strategy

# Colors
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
BLACK = (0, 0, 0)
RED = (255,0,00)
GREEN = (0,250,0)
BLUE = (0,0,255)

sys.setrecursionlimit(10000)
board = []
movement_symbol = "\u2705"
# movement_symbol = ">"
block_symbol = "\u274C"
# block_symbol = "x"
fire_symbol = "\U0001F525"
# fire_symbol = "#"
optimal_dim = 105
optimal_p = 0.225
optimal_q = .1

pygame.init()
screen_size = (optimal_dim*10, optimal_dim*10)
width = 8
screen = pygame.display.set_mode(screen_size)

def create_maze(dim, p):
    global board
    board = []
    for row in range(dim):
        board.append([])
        for col in range(dim):
            board[row].append(Cell(row, col))
            if (row == 0 and col == 0) or (row == dim-1 and col == dim-1):
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
            if row != (dim-1):
                if not board[row + 1][col].is_blocked:
                    board[row][col].add_neighbor(board[row + 1][col])
            if col != 0:
                if not board[row][col - 1].is_blocked:
                    board[row][col].add_neighbor(board[row][col - 1])
            if col != (dim-1):
                if not board[row][col + 1].is_blocked:
                    board[row][col].add_neighbor(board[row][col + 1])

def dfs(start, goal):
    return dfs_helper(start, goal, [])

def dfs_helper(current, goal,path):
    path.append(current)
    if current == goal:
        return path
    for neighbor in current.neighbors:
        if neighbor not in path:
            dfs_helper(neighbor,goal,path)



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
    fringeq = PriorityQueue(-1)
    backward_mapping = dict()

    gscores = dict()
    gscores[start] = 0
    fscores = dict()
    fscores[start] = gscores[start] + hFunc(start.row, start.col, goal.row, goal.col)

    fringeq.put(PrioritizedItem(fscores[start], start))
    while not fringeq.empty():
        current = fringeq.get().item
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
    return None


def euclidean_dist(x1,y1,x2,y2):
    return math.sqrt(((x1-x2)**2) + ((y1-y2)**2))


def manhattan_dist(x1, y1, x2, y2):
    return abs(x1-x2) + abs(y1-y2)

# def bfsBD(start,goal):
#     fringSt = PriorityQueue()
#     fringEn = PriorityQueue()
#     gscoresSt = dict();
#     gscoresEn = dict();
#     gscoresSt[start] = 0
#     gscoresEn[goal] = 0
#     backward_mappingF = dict()
#     backward_mappingB = dict()
#     fringSt.put(PrioritizedItem(gscoresSt[start], start))
#     fringEn.put(PrioritizedItem(gscoresEn[goal], goal))
#     while not fringSt.empty() and not fringEn.empty():
#         currentF = fringSt.get().item
#         currentB = fringEn.get().item
#         for neighbor in currentF.neighbors:
#             try:
#                 gscoresSt[neighbor]
#             except KeyError:
#                 gscoresSt[neighbor] = math.inf
#             try:
#                 gscoresEn[neighbor]
#             except KeyError:
#                 gscoresEn[neighbor] = math.inf
#             recalc_g = gscoresSt[currentF] + 1
#             if gscoresSt[neighbor] != math.inf and gscoresEn[neighbor] != math.inf:
#                 path = [neighbor]
#                 path.insert(0, currentF)
#                 currentTemp = neighbor
#
#                 while currentF != start:
#                     currentF = backward_mappingF[currentF]
#                     path.insert(0, currentF)
#                 while currentTemp != goal:
#                     currentTemp = backward_mappingB[currentTemp]
#                     path.insert(len(path), currentTemp)
#                 return path
#             if recalc_g < gscoresSt[neighbor]:
#                 backward_mappingF[neighbor] = currentF
#                 gscoresSt[neighbor] = recalc_g
#
#                 for node in fringSt.queue:
#                     if node.item == neighbor:
#                         fringSt.queue.remove(node)
#                         break
#                 fringSt.put(PrioritizedItem(gscoresSt[neighbor], neighbor))
#         for neighbor in currentB.neighbors:
#             try:
#                 gscoresSt[neighbor]
#             except KeyError:
#                 gscoresSt[neighbor] = math.inf
#             try:
#                 gscoresEn[neighbor]
#             except KeyError:
#                 gscoresEn[neighbor] = math.inf
#             recalc_g = gscoresEn[currentB] + 1
#
#             if recalc_g < gscoresEn[neighbor]:
#                 backward_mappingB[neighbor] = currentB
#                 gscoresEn[neighbor] = recalc_g
#
#                 for node in fringEn.queue:
#                     if node.item == neighbor:
#                         fringEn.queue.remove(node)
#                         break
#                 fringEn.put(PrioritizedItem(gscoresEn[neighbor], neighbor))
#
#
#     return []


def bfsBD(start,goal):
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
            for neighbor in current_front.neighbors:
                if neighbor not in discovered_front:
                    discovered_front.append(neighbor)
                    backward_mapping[neighbor] = current_front
                    fringe_front.put(neighbor)
            intersect = intersection(discovered_front,discovered_back)
            if intersect is not None:
                disc = intersect[0]
                path_front_to_intersection = back_track(backward_mapping, start, disc)
                path_back_to_intersection = back_track(forward_mapping, goal, disc)
                path_back_to_intersection.reverse()
                return path_front_to_intersection + path_back_to_intersection
        if not fringe_back.empty():
            current_back = fringe_back.get()
            for neighbor in current_back.neighbors:
                if neighbor not in discovered_back:
                    discovered_back.append(neighbor)
                    forward_mapping[neighbor] = current_back
                    fringe_back.put(neighbor)
            intersect = intersection(discovered_front,discovered_back)
            if intersect is not None:
                disc = intersect[0]
                path_front_to_intersection = back_track(backward_mapping, start, disc)
                path_back_to_intersection = back_track(forward_mapping, goal, disc)
                path_back_to_intersection.reverse()
                return path_front_to_intersection + path_back_to_intersection
    return None


def intersection(list1, list2):
    list2_as_set = set(list2)
    intersect = [value for value in list1 if value in list2_as_set]
    if len(intersect) == 0:
        return None
    return intersect

def fire_strat_1(q,num_tests):
    global board
    fail_counter = 0
    for i in range(num_tests):
        print("\r Running Test "+str(i), end="")
        start = None
        goal = None
        fire_loc = None
        path = None
        while True:
            create_maze(optimal_dim, optimal_p)
            start = board[0][0]
            goal = board[optimal_dim - 1][optimal_dim - 1]
            fire_loc = (random.randint(0, optimal_dim - 1), random.randint(0, optimal_dim - 1))
            if board[fire_loc[0]][fire_loc[1]].is_blocked or fire_loc == (0, 0) \
                    or fire_loc == (optimal_dim-1, optimal_dim-1):
                continue
            else:
                if astar(start, board[fire_loc[0]][fire_loc[1]], euclidean_dist) is None \
                        or astar(start, goal, euclidean_dist) is None:
                    continue
                board[fire_loc[0]][fire_loc[1]].set_fire_status(True)

            path = astar(start, goal, euclidean_dist)

            if path is None:
                continue
            break

        # print_maze(path)
        for cell in path:
            if cell.on_fire:
                fail_counter += 1
                # print("FAILLLL")
                break
            compute_fire_movement(q)
            draw_maze([path[0]])
            path = path[1:]
            # print_maze(path)
        board = []
    print()
    return (fail_counter/num_tests) * 100

def fire_strat_2(q, num_tests):
    start = None
    goal = None
    fire_loc = None
    path = None
    fail_counter = 0
    for i in range(num_tests):
        while True:
            create_maze(optimal_dim, optimal_p)
            start = board[0][0]
            goal = board[optimal_dim - 1][optimal_dim - 1]
            fire_loc = (random.randint(0, optimal_dim - 1), random.randint(0, optimal_dim - 1))
            if board[fire_loc[0]][fire_loc[1]].is_blocked or fire_loc == (0, 0) \
                    or fire_loc == (optimal_dim - 1, optimal_dim - 1):
                continue
            else:
                if astar(start, board[fire_loc[0]][fire_loc[1]], euclidean_dist) is None \
                        or astar(start, goal, euclidean_dist) is None:
                    continue
                board[fire_loc[0]][fire_loc[1]].set_fire_status(True)

            path = astar(start, goal, euclidean_dist)

            if path is None:
                continue
            break
        while True:
            if path is None or path[0].on_fire:
                fail_counter += 1
            if path[1] == goal:
                break
            compute_fire_movement(q)
            draw_maze([path[0]])
            path = astar(path[1], goal, euclidean_dist)



# Multi Process version To go quicker but does not display the maze!!
# def fire_strat_2(q, num_tests):
#     fail_counter = 0
#
#     processes = []
#     # print("starting ", end="")
#     for i in range(num_tests):
#         processes.append(Process(target=fire_strat_2_helper, args=(q,)))
#         processes[i].start()
#         # print("\r" + str(i), end="")
#     # print("\nJoining ", end="")
#     for p in processes:
#         p.join()
#         # print("#", end="")
#         # print(p.exitcode)
#         if p.exitcode == 1:
#             fail_counter += 1
#     # print("Fail Counter: " + str(fail_counter))
#     return (fail_counter/num_tests) * 100
#
#
# def fire_strat_2_helper(q):
#     start = None
#     goal = None
#     fire_loc = None
#     path = None
#     while True:
#         create_maze(optimal_dim, optimal_p)
#         start = board[0][0]
#         goal = board[optimal_dim - 1][optimal_dim - 1]
#         fire_loc = (random.randint(0, optimal_dim - 1), random.randint(0, optimal_dim - 1))
#         if board[fire_loc[0]][fire_loc[1]].is_blocked or fire_loc == (0, 0) \
#                 or fire_loc == (optimal_dim - 1, optimal_dim - 1):
#             continue
#         else:
#             if astar(start, board[fire_loc[0]][fire_loc[1]], euclidean_dist) is None \
#                     or astar(start, goal, euclidean_dist) is None:
#                 continue
#             board[fire_loc[0]][fire_loc[1]].set_fire_status(True)
#
#         path = astar(start, goal, euclidean_dist)
#
#         if path is None:
#             continue
#         break
#     while True:
#         if path is None or path[1].on_fire:
#             sys.exit(1)
#             # return
#         if path[1] == goal:
#             sys.exit(0)
#             # return
#         compute_fire_movement(q)
#         path = astar(path[1], goal, euclidean_dist)


def compute_fire_movement(q):
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


def print_maze(path):
    print("  \t", end='')
    for i in range(len(board)):
        print("\t" + str(i) + "\t", end='')
    print()
    for row in range(len(board)):
        print("    ",  end='')
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
        print("    ",  end='')
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
          "\npress the enter key press to continue or any other key followed by enter to continue\n") == "" :
    # Find a map size (dim) that is large enough to produce maps that require some work to solve,
    # but small enough that you can run each algorithm multiple times for a range of possible p values.
    # How did you pick a dim? We picked Dim by looping through a bunch of different dims with a bunch of different
    # p values we picked our dim based on which dim took a decent amount of average time to complete for each p value
    #Optimal Value finder: We Selected -- Dim: 105 p: 0.225 Fail: 20.0 AVG Time: 0.1652039000000002
        results = []
        for dim in range(15, 125):
            if not dim % 5 == 0:
                continue
            for p in [.2, .225, .25, .275, .3]:
                fail_counter = 0
                tests = 50
                t0 = time.process_time()
                for i in range(tests):
                    # print(i)
                    create_maze(dim, p)
                    if astar(board[0][0], board[dim - 1][dim - 1], euclidean_dist) is None:
                        fail_counter += 1
                percent = (fail_counter/tests) * 100
                avgtime = ((time.process_time()-t0)/tests)
                print("Dim: " + str(dim) + " p: " + str(p / 10) + " Fail: " + str(percent) + " AVG Time: " + str(avgtime))
                results.append((dim, p/10, percent, avgtime))
                if percent > 20:
                    break
        print("\n\n\n\n\n")
        print(results)

    if input("\n\nTo generate shortest paths for dim = optimal (105), p = .2 "
             "\npress enter, press any other key followed by enter to continue. "
             "\nTo move to the next algorithm press the X to close the window. "
             "\nThe Title box of the screen will tell you which algorithm\n") == "" :
        create_maze(optimal_dim,optimal_p)

        bfspath = bfs(board[0][0], board[104][104])
        dfspath = dfs(board[0][0], board[104][104])
        astarEDpath = astar(board[0][0], board[104][104], euclidean_dist)
        astarMDpath = astar(board[0][0], board[104][104], manhattan_dist)
        bfsBDpath = bfsBD(board[0][0], board[104][104])

        pygame.display.set_caption('BFS')
        draw_maze(bfspath)
        running = True
        while running:
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


    # create_maze(optimal_dim, optimal_p)
    # running = True
    # path = bfsBD(board[0][0], board[104][104])
    # print_maze(path)
    # while running:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
    #     draw_maze(path)

