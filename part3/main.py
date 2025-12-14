import torch
import time
from environment import Environment
from dqn_agent  import DQNAgent
from config     import Config

def train():
    config = Config()
    env = Environment(config)
    agent = DQNAgent(config)
    
    print(f"Start Training on device: {agent.device}")
    print("-" * 30)
    total_steps = 0
    
    for i_episode in range(config.EPISODES):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            total_steps += 1
            if total_steps % 4 == 0:
                agent.update()
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # 更新探索率
        agent.update_epsilon()
        
        # 更新 Target Network
        if i_episode % config.TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        # 顯示進度
        if i_episode % 100 == 0:
            print(f"Episode {i_episode}\t Score: {total_reward:.2f}\t Epsilon: {agent.epsilon:.2f}")

    print("Training Finished!")
    torch.save(agent.policy_net.state_dict(), "cargo_dqn_model.pth")
    print("Model saved as 'cargo_dqn_model.pth'")
    
def run_demo():
    config = Config()
    env = Environment(config, render_mode='human')
    agent = DQNAgent(config)
    
    model_path = "cargo_dqn_model.pth"
    try:
        agent.policy_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        agent.policy_net.eval()
        print("成功載入模型！")
    except FileNotFoundError:
        print(f"找不到模型檔案：{model_path}，請先執行 main.py 進行訓練。")
        return

    agent.epsilon = 0.05
    for i_episode in range(5):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward
            
            # 方便觀察
            time.sleep(0.1) 
            env.render()

        print(f"Demo Episode {i_episode+1} Score: {total_reward:.2f}")
        
if __name__ == "__main__":
    train()
    run_demo()