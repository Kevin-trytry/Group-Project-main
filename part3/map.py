import numpy as np

EMPTY = 0
AGENT = 1
CARGO = 2

class Map:
    def __init__(self, width = 8, height = 8, rng = None):
        # initialize the map
        self.width = width
        self.height = height
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.grid.fill(EMPTY)
        
    def is_valid_position(self, x, y):
        # check if hit the wall
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return True
    
    def get_random_pos(self):
        # generate a empty positoin (for agent or cargo)
        while True:
            x = self.rng.integers(0, self.width)
            y = self.rng.integers(0, self.height)

            if self.is_valid_position(x, y) and self.grid[y, x] is EMPTY:
                return (x, y)