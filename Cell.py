class Cell:

    def __init__(self,row,col):
        self.row = row
        self.col = col
        self.isBlocked = False
        self.onFire = False
        self.neighbors = []

    def __str__(self):
        return "(" + str(self.row) + ", " + str(self.col) + ")"

    def set_block_status(self, status):
        self.isBlocked = status

    def set_fire_status(self, status):
        self.onFire = status

    def set_neighbors(self,neighbors):
        self.neighbors = neighbors

    def add_neighbor(self,neighbor):
        self.neighbors.append(neighbor)
