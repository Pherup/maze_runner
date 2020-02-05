from Cell import Cell



def create_board(dim):
    global board = []
    for row in range(dim):
        board.append([])
        for col in range(dim):
            board[row].append(Cell(row,col))
    for row in range(dim):
        for col in range(dim):
            if row != 0:
                board[row][col].add_neighbor(board[row - 1][col])
            if row != dim:
                board[row][col].add_neighbor(board[row + 1][col])
            if col != 0:
                board[row][col].add_neighbor(board[row][col - 1])
            if col != dim:
                board[row][col].add_neighbor(board[row][col + 1])

def populate_board():

def dfs():

def bfs():

def astarED():

def astarMD():

def bfsBD():

