# Deep Reinforcement Learning: Implementations and Empirical Studies

Implementations and experimental studies of reinforcement learning methods, spanning value-based learning, policy optimization, planning, exploration, imitation, multi-agent coordination, and adaptation across tasks.

Developed through practical coursework in Deep Reinforcement Learning. Each project combines algorithm implementations, experimental comparisons, and analysis in Jupyter notebooks.

## Projects

| Project | Methods and focus |
| --- | --- |
| [Value-Based Learning](projects/01-value-based-learning/) | DQN, Double DQN, Dueling DQN, prioritized replay, and Half-Rainbow |
| [Policy Gradient Methods](projects/02-policy-gradient-methods/) | REINFORCE, variance reduction, and natural policy gradients |
| [Continuous Control](projects/03-continuous-control/) | DPG, DDPG, and TD3 |
| [Model-Based Planning](projects/04-model-based-planning/) | Dyna-Q, prioritized sweeping, and MCTS with learned models |
| [Contextual Bandits](projects/05-contextual-bandits/) | LinUCB, neural contextual bandits, safe exploration, and off-policy evaluation |
| [Deep Exploration](projects/06-deep-exploration/) | Double DQN, random network distillation, and Bootstrapped DQN |
| [Imitation and Offline Learning](projects/07-imitation-and-offline-learning/) | Behavior cloning, DAgger, GAIL, and DDPG+BC |
| [Multi-Agent Learning](projects/08-multi-agent-learning/) | Fictitious play, MADDPG, MAPPO, IDDPG, and IPPO |
| [Meta-Reinforcement Learning](projects/09-meta-reinforcement-learning/) | MAML, RL², and VariBAD |

## Getting Started

Choose a project under `projects/` and run the following from its directory:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab
```

Run notebook cells in order. Use a separate environment for projects with different simulator or library requirements. Each project provides a short README; [running notes](docs/running.md) cover compatibility and missing dependencies.

## Experimental Artifacts

Notebooks retain their recorded outputs. Available policy videos, numerical results, and supporting material are stored with the corresponding project in `media/`, `results/`, or `reports/`.

The continuous-control and deep-exploration projects require additional local modules absent from the source archive. Dependency files are starting points rather than validated environment lockfiles.
