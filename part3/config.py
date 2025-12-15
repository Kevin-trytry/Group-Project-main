class Config:
    def __init__(self):
        self.MAP_SIZE = 8
        self.BATCH_SIZE = 64
        self.LR = 0.00025          # Learning Rate
        self.GAMMA = 0.9        # 折扣因子 (重視未來程度)
        self.EPSILON_START = 1.0 # 初始探索率
        self.EPSILON_END = 0.05  # 最終探索率
        self.EPSILON_DECAY = 0.998
        self.MEMORY_CAPACITY = 50000
        self.TARGET_UPDATE = 10  # 每幾回合更新一次 Target Network
        self.MAX_STEPS = 100      # 每回合最大步數 (防止死循環)
        self.EPISODES = 3000      # 總訓練回合數  
        self.STEP_COST = -0.5      # 每走一步的消耗 (鼓勵最短路徑)
        
        # 各類貨物的數量
        self.CARGO_TYPES = 3
        self.NUM_LIMIT = 1
        self.NUM_GOOD = 3
        self.NUM_BAD = 2