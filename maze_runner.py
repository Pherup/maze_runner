from cell import Cell
from queue import PriorityQueue
from prioritizeditem import PrioritizedItem
import math
import random
from queue import Queue
import time
import sys

#TODO: Create Hard Maze
#TODO: Bidirectional BFS
#TODO: Fire Movement Do this before Fire Strategies
#TODO: Fire startegy 1
#TODO: Fire startegy 2


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
optimal_q = 0.1


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


def bfsBD():
    return None


def fire_strat_1(q):
    global board
    fail_counter = 0
    num_tests = 30
    for i in range(num_tests):
        print(i)
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
                print("FAILLLL")
                break
            compute_fire_movement(q)
            # print_maze(path)
        board = []
    return (fail_counter/num_tests) * 100



def fire_strat_2(q):
    global board
    fail_counter = 0
    num_tests = 30
    for i in range(num_tests):
        print("\r" + str(i))
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
        count = 0 #Delete later
        while True:
            if path is None or path[1].on_fire:
                fail_counter += 1
                print("\rFAILLLL")
                break
            if path[1] == goal:
                break
            count += 1
            compute_fire_movement(optimal_q)
            print("\rcomputing new path " + str(count), end="")
            path = astar(path[1], goal, euclidean_dist)
        board = []
    return (fail_counter / num_tests) * 100


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


def main():
    print(fire_strat_2(optimal_q))





    #Optimal Value finder: We Selected -- Dim: 105 p: 0.225 Fail: 20.0 AVG Time: 0.1652039000000002
    # results = []
    # for dim in range(15, 125):
    #     if not dim % 5 == 0:
    #         continue
    #     for p in [2, 2.25, 2.5, 2.75, 3]:
    #         fail_counter = 0
    #         tests = 50
    #         t0 = time.process_time()
    #         for i in range(tests):
    #             # print(i)
    #             create_maze(dim, p/10)
    #             if astar(board[0][0], board[dim - 1][dim - 1], euclidean_dist) is None:
    #                 fail_counter += 1
    #         percent = (fail_counter/tests) * 100
    #         avgtime = ((time.process_time()-t0)/tests)
    #         print("Dim: " + str(dim) + " p: " + str(p / 10) + " Fail: " + str(percent) + " AVG Time: " + str(avgtime))
    #         results.append((dim, p/10, percent, avgtime))
    #         if percent > 20:
    #             break
    # print("\n\n\n\n\n")
    # print(results)

    # create_maze(2, 0.0)
    # print_maze_nopath()
    # output = astar(board[0][0],board[1][1],euclidean_dist)
    # if not (output == None):
    #     print_maze(output)
    #     for item in output:
    #         print(str(item))

main()


