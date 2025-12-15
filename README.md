# Group Project Setup Guide

# OOP Final Project - Reinforcement Learning Robot

**Team:** Group 22
**Course:** Object-Oriented Programming

## 1. Project Overview
This project explores Reinforcement Learning (RL) across three different environments:
- **Part 1: Mountain Car** (Classic Control)
- **Part 2: Frozen Lake** (Tabular Q-Learning with Reward Shaping)
- **Part 3: Autonomous Cargo Collector** (Deep Q-Network with OOP Architecture)
---

## Installation

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Navigate to the Gymnasium directory
cd group_project/Gymnasium

# 4. Install Gymnasium in editable mode
pip install -e .

# 5. Install additional dependencies
pip install "gymnasium[classic_control]"
pip install matplotlib
```

---

## 🚀 Running the Project

### **Part 1: Mountain Car**
Train and test the reinforcement learning agent:

```bash
# Train the agent
python mountain_car.py --train --episodes 5000

# Render and visualize performance
python mountain_car.py --render --episodes 10
```

### **Part 2: Frozen Lake**
Run the Frozen Lake environment:

```bash
python frozenlake_modified.py
```

### **Part 3: DQN learning robot Project Environment**
The Part 3 entry point is main.py. You can toggle between Training and Demo modes by editing the code.

To Train the Agent:
```bash
#1. Open main.py.
#2. Set the run command to: run(is_training=True, render=False)
#3. Execute:
python main.py
```

To Run the Demo (Test Mode):
```bash
#1. Open main.py.
#2. Set the run command to: run(is_training=False, render=True)
#3. Execute:
python main.py
```

### **Part 4: Contribution list**
```bash
Name       Student ID   Task / Contribution
尤振德      B123040021   Part2 Implementation, Part3 environment.py, robot.py, and prepared slides
林雲翔      B123040033   Part3 Implemented ddqn_agent.py, designed the Neural Network structure, tuned hyperparameters, and prepared slides
莊志文      B123040004   Part3 cargo.py and main.py, Managed GitHub repository wrote README and reflection report, and prepared slides.
```