class Cell:

    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.isBlocked = False
        self.onFire = False


    def setBlockStatus(self, status):
        self.isBlocked = status

    def setFireStatus(self, status):
        self.onFire = status
