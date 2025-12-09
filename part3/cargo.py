import random

class Cargo:
    """The base class for different types of cargo."""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.active = True # The cargo is on the map
        self.name = "General Cargo"
        self.symbol = "C"

    def get_reward(self) :
        """When the robot collects the cargo, it receives a score."""
        self.active = False # The cargo is collected
        return 0
    
    def update(self):
        """Update the state of the cargo when every round starts."""
        pass

    def get_position(self):
        """Get the current position of the cargo."""
        return (self.x, self.y)
    
class GoodCargo(Cargo):
    """The cargo can increase the score of the robot."""
    def __init__(self, x, y):
        super().__init__(x, y)
        self.value = 10
        self.name = "Gold"
        self.symbol = "G"
    
    def get_reward(self):
        """When the robot collects the cargo, it receives a score."""
        self.active = False # The cargo is collected
        return self.value
    
class BadCargo(Cargo):
    """The cargo can decrease the score of the robot."""
    def __init__(self, x, y):
        super().__init__(x, y)
        self.value = -10
        self.name = "Trash"
        self.symbol = "T"
    
    def get_reward(self):
        self.active = False # The cargo is collected
        return self.value
    
class LimitedCargo(Cargo):
    """The cargo can increase the score of the robot, but only within a limited time."""
    def __init__(self, x, y, lifetime = 20):
        super().__init__(x, y)
        self.value = 30
        self.name = "Dimond"
        self.symbol = "D"
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
        
        self.lifetime -= 1
        
        if self.lifetime <= 0:
            self.active = False # The cargo expires
            print(f"The time is up, {self.name} is expired.")