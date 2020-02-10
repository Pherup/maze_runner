from cell import Cell
from queue import PriorityQueue
from prioritizeditem import PrioritizedItem
import math
import random
from queue import Queue
from queue import LifoQueue

board = []


def create_maze(dim, p):
    for row in range(dim):
        board.append([])
        for col in range(dim):
            board[row].append(Cell(row, col))
            if (row == 0 and col == 0) or (row == dim-1 and col == dim-1):
                continue
            else:
                if random.random() < p:
                    board[row][col].set_block_status(True)
    # for row in range(dim):
    #     for col in range(dim):
    #         if (row == 0 and col == 0) or (row == dim-1 and col == dim-1):
    #             continue
    #         else:
    #             if random.random() < p:
    #                 board[row][col].set_block_status(True)
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


# def dfs(start, goal):
#     fringe = LifoQueue(-1)
#     discovered = [start]
#     backward_mapping = dict()
#     fringe.put(start)
#
#     while not fringe.empty():
#         current = fringe.get()
#         if current == goal:
#             return back_track(backward_mapping, start, current)
#         for neighbor in current.neighbors:
#             if neighbor not in discovered:
#                 discovered.append(neighbor)
#                 backward_mapping[neighbor] = current
#                 fringe.put(neighbor)

def dfs(current, goal,path):
    path.append(current)
    if current == goal:
        return path
    for neighbor in current.neighbors:
        if neighbor not in path:
            dfs(neighbor,goal,path)



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
    fscores = dict();
    fscores[start] = gscores[start] + hFunc(start.row, start.col, goal.row, goal.col)

    fringeq.put(PrioritizedItem(fscores[start], start))
    while not fringeq.empty():
        current = fringeq.get().item
        if current == goal:
            return back_track(backward_mapping, start, current)
        for neighbor in current.neighbors:
            try:
                gscores[neighbor]
            except KeyError:
                gscores[neighbor] = math.inf
            recalc_g = gscores[current] + 1
            if recalc_g < gscores[neighbor]:
                backward_mapping[neighbor] = current
                gscores[neighbor] = recalc_g
                fscores[neighbor] = gscores[neighbor] + hFunc(neighbor.row, neighbor.col, goal.row, goal.col)

                node_in_q = False
                for node in fringeq.queue:
                    if node.item == neighbor:
                        fringeq.queue.remove(node)
                        break
                fringeq.put(PrioritizedItem(fscores[neighbor], neighbor))
    return []


def euclidean_dist(x1,y1,x2,y2):
    return math.sqrt(((x1-x2)**2) + ((y1-y2)**2))


def manhattan_dist(x1,y1,x2,y2):
    return abs(x1-x2) + abs(y1-y2)


def bfsBD():
    return None


def fire_strat_1():
    return None


def fire_strat_2():
    return None


def compute_fire_movement():
    return None


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
                print("\t\u2023\t", end='')
            else:
                if col < len(board[0]) and board[row][col].is_blocked:
                    print("\tx\t", end='')
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
                print("\tx\t", end='')
            else:
                print("\t \t", end='')
        print()
    print("    ", end='')
    for i in range(len(board)):
        print("________", end='')
    print()


def main():
    create_maze(20, 0.3)
    print_maze_nopath()
    output = bfs(board[0][0],board[19][19])
    print_maze(output)
    for item in output:
         print(str(item))

main()


