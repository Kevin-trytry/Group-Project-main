class Cargo:
    image_file = "cargo.png"

    """The base class for different types of cargo."""
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.pos = (row, col)
        self.active = True # The cargo is on the map

    def get_reward(self) :
        """When the robot collects the cargo, it receives a score."""
        self.active = False # The cargo is collected
        return 0
    
    def update(self):
        """Update the state of the cargo when every round starts."""
        pass

    def get_position(self):
        """Get the current position of the cargo."""
        return self.pos
    
class GoodCargo(Cargo):
    image_file = "good.png"

    """The cargo can increase the score of the robot."""
    def __init__(self, row, col):
        super().__init__(row, col)
        self.value = 30
        self.name = 'good'
    
    def get_reward(self):
        """When the robot collects the cargo, it receives a score."""
        self.active = False # The cargo is collected
        return self.value
    
class BadCargo(Cargo):
    image_file = "bad.png"

    """The cargo can decrease the score of the robot."""
    def __init__(self, row, col):
        super().__init__(row, col)
        self.value = -20
        self.name = 'bad'
        self.image_file = "bad.png"
    
    def get_reward(self):
        self.active = False # The cargo is collected
        return self.value
    
class LimitedCargo(Cargo):
    image_file = "limit.png"

    """The cargo can increase the score of the robot, but only within a limited time."""
    def __init__(self, row, col, lifetime = 40):
        super().__init__(row, col)
        self.value = 100
        self.name = 'limit'
        self.image_file = "limit.png"
        self.remain_lifetime = lifetime
        self.lifetime = lifetime

    def get_reward(self):
        # if the cargo is still active, give the score
        if self.active:
            self.active = False 
            return self.value
        else:
            return 0 
    
    def update(self):
        if not self.active: 
            return
        
        self.remain_lifetime -= 1
        
        if self.remain_lifetime <= 0:
            self.active = False # The cargo expires
            print(f"The time is up, limited cargo is expired.")