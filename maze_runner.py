from cell import Cell
from queue import PriorityQueue
from prioritizeditem import PrioritizedItem
import math

board = []

def create_board(dim):
    for row in range(dim):
        board.append([])
        for col in range(dim):
            board[row].append(Cell(row,col))


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

def populate_board():
    return None

def dfs():
    return None

def bfs():
    return None

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
            path = [current]
            while current != start:
                current = backward_mapping[current]
                path.insert(0, current)
            return path
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

def main():
    create_board(4)
    assign_board_neighbors(4)
    output = astar(board[0][0],board[3][3],euclidean_dist)
    for item in output:
        print(str(item))

main()


