class Robot:
    image_file = "bot.png"

    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.score = 0
        self.step_count = 0
        self.name = "Robot"
    
    @property
    def pos(self):
        return (self.row, self.col)
    
    def add_score(self, points):
        self.score += points
    
    def get_next_position(self, action):
        next_row, next_col = self.row, self.col

        # 0:上, 1:下, 2:左, 3:右
        if action == 0: # Up
            next_row -= 1
        elif action == 1: # Down
            next_row += 1
        elif action == 2: # Left
            next_col -= 1
        elif action == 3: # Right
            next_col += 1
        
        return (next_row, next_col)

    def move_to(self, row, col):
        self.row = row
        self.col = col
        self.step_count += 1