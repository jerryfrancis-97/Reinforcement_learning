# Reinforcement Learning Repository

A comprehensive collection of reinforcement learning algorithms, implementations, and projects covering various RL techniques from basic Q-learning to advanced meta-learning approaches.

## 🚀 Overview

This repository contains implementations of various reinforcement learning algorithms, from fundamental methods to cutting-edge techniques. It serves as both a learning resource and a practical toolkit for RL practitioners and researchers.

## 📚 Contents

### Core RL Algorithms

#### 1. **Q-Learning & Value-Based Methods**
- **Taxi Environment**: Classic gym environment solved with Q-learning
- **Vanilla DQN**: Deep Q-Network implementation
- **DQN Variants**:
  - Double DQN (DDQN)
  - Dueling DQN
  - DDQN with Dueling architecture

#### 2. **Policy-Based Methods**
- **Policy Gradient**: Implementation with adaptive noise scaling
- **Hill Climbing**: Policy optimization technique
- **Adaptive Noise Scaling**: Dynamic exploration strategy

#### 3. **Actor-Critic Methods**
- **Actor-Critic**: Discrete action space implementation
- **DDPG**: Deep Deterministic Policy Gradient for continuous control

#### 4. **Meta-Learning**
- **MAML**: Model-Agnostic Meta-Learning implementation
- **Sine Wave Experiments**: Meta-learning on regression tasks

### Projects & Applications

#### **Navigation Project**
- Unity-based navigation environment
- DQN agent implementation
- Complete project with checkpoints and results

#### **Gym Environments**
- CartPole-v0 implementations
- LunarLander-v2 experiments
- Custom environment wrappers

#### **Game Development**
- Pygame implementations
- Custom game environments for RL testing

## 🛠️ Technologies & Dependencies

- **Python**: Core programming language
- **PyTorch**: Deep learning framework
- **OpenAI Gym**: RL environment library
- **NumPy**: Numerical computing
- **Matplotlib**: Visualization and plotting
- **Pygame**: Game development for RL testing

## 📁 Repository Structure

```
Reinforcement_learning/
├── Actor_critic/           # Actor-critic implementations
├── adaptive_noise_scaling/ # Adaptive noise scaling algorithms
├── DDPG/                  # Deep Deterministic Policy Gradient
├── DQNs+Variants/         # DQN and its variants
├── Meta-RL/               # Meta-reinforcement learning
├── Policy_grad/           # Policy gradient methods
├── pygame/                # Game implementations
├── RL_project1nav/        # Navigation project
├── RL_Coursera_Specialisation/ # Course materials
└── *.ipynb                # Jupyter notebooks
```

## 🎯 Key Features

- **Comprehensive Coverage**: From basic Q-learning to advanced meta-learning
- **Practical Implementations**: Working code with real environments
- **Multiple Environments**: CartPole, LunarLander, Taxi, custom games
- **Research Ready**: Includes experimental results and visualizations
- **Educational**: Suitable for learning and understanding RL concepts

## 🚦 Getting Started

### Prerequisites
```bash
pip install torch gym numpy matplotlib pygame
```

### Quick Start
1. Clone the repository
2. Navigate to any algorithm folder
3. Run the corresponding Python file or Jupyter notebook

### Example Usage
```python
# For DQN implementation
cd DQNs+Variants/
python ddqn_w_duelling.py

# For policy gradient
cd Policy_grad/
python policy_grad.py
```

## 📊 Results & Visualizations

The repository includes:
- Training curves and performance metrics
- Environment visualizations
- Model checkpoints for reproduction
- Experimental results and comparisons

## 🎓 Learning Resources

- **Coursera Specialization**: Complete RL course materials
- **Jupyter Notebooks**: Interactive learning examples
- **Code Comments**: Detailed explanations in implementations
- **Results Analysis**: Performance comparisons and insights

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs or issues
- Suggest new algorithms or improvements
- Add new environments or implementations
- Improve documentation

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- OpenAI Gym for providing RL environments
- PyTorch team for the deep learning framework
- Coursera for educational materials
- RL research community for algorithm developments

---

**Note**: This repository is actively maintained and updated with new implementations and improvements. Check back regularly for updates!

