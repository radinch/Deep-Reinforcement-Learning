# Deep Reinforcement Learning: Implementations and Empirical Studies

Implementations and empirical studies of reinforcement learning algorithms, spanning value-based learning, policy optimization, model-based planning, exploration, imitation, multi-agent coordination, and adaptation across tasks.

Developed through practical coursework in Deep Reinforcement Learning, the repository contains 14 Jupyter notebooks organized into nine thematic projects. The notebooks combine algorithm implementations, methodological explanations, experimental comparisons, and interpretation of results across standard benchmarks and custom environments.

## Projects

| Project | Methods | Experimental focus |
| --- | --- | --- |
| [Value-Based Learning](projects/01-value-based-learning/) | DQN, Double DQN, Dueling DQN, prioritized replay, Half-Rainbow | Compare value estimation, network architecture, and replay strategies on LunarLander. |
| [Policy Gradient Methods](projects/02-policy-gradient-methods/) | REINFORCE, natural policy gradients | Study reward-to-go and baseline variance reduction on CartPole, and policy-gradient geometry in MiniTetris. |
| [Continuous Control](projects/03-continuous-control/) | DPG, DDPG, TD3 | Compare deterministic actor-critic methods in a custom moving-target Reacher environment. |
| [Model-Based Planning](projects/04-model-based-planning/) | Dyna-Q, prioritized sweeping, MCTS | Examine planning and reward shaping on CliffWalking, alongside search with learned model components. |
| [Contextual Bandits](projects/05-contextual-bandits/) | LinUCB, neural contextual bandits, off-policy evaluation | Evaluate text moderation policies through regret, safety constraints, calibration, and logged-feedback estimators. |
| [Deep Exploration](projects/06-deep-exploration/) | Double DQN, random network distillation, Bootstrapped DQN | Investigate sparse-reward exploration and distracting observations in normal and noisy DeepSea. |
| [Imitation and Offline Learning](projects/07-imitation-and-offline-learning/) | Behavior cloning, DAgger, GAIL, DDPG+BC | Study distribution shift, expert-label efficiency, adversarial imitation, and offline behavior regularization. |
| [Multi-Agent Learning](projects/08-multi-agent-learning/) | Fictitious play, MADDPG, MAPPO, IDDPG, IPPO | Analyze strategic learning dynamics and centralized versus decentralized critics in cooperative VMAS Navigation. |
| [Meta-Reinforcement Learning](projects/09-meta-reinforcement-learning/) | MAML, RL², VariBAD | Compare gradient-based, recurrent, and latent-inference adaptation to hidden target velocities in HalfCheetahVel. |

## Experimental Scope

The projects examine how algorithmic choices affect learning dynamics, sample efficiency, exploration, coordination, and adaptation. Depending on the experiment, notebooks include learning curves, ablations, policy visualizations, multi-seed comparisons, or held-out-task evaluation. Experimental settings and written analyses are documented alongside the implementations.

## Repository Organization

Each project contains its notebooks, a short README, and a dependency file. Additional artifacts are included where available:

| Location | Contents |
| --- | --- |
| `projects/` | Thematic projects and their implementations. |
| `media/` | Recorded policy rollouts within the corresponding project. |
| `results/` | Saved exploration runs, summaries, and evaluation metadata. |
| `reports/` | Supporting material associated with a project. |
| `docs/running.md` | Shared setup guidance and project-specific compatibility notes. |

## Getting Started

Choose a project and run the following from its directory, for example:

```bash
cd projects/01-value-based-learning
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open the selected notebook and run its cells in order. Launching JupyterLab from the project directory ensures that relative paths resolve correctly. Use separate environments when switching between legacy Gym, Gymnasium, MuJoCo, and TorchRL dependencies. See the [running notes](docs/running.md) for additional setup details.

## Outputs and Reproducibility

Notebooks retain their recorded outputs for inspection, alongside the available videos and numerical artifacts. Rerunning experiments may overwrite these files and can produce different results across seeds, hardware, and library versions.

The continuous-control and deep-exploration projects require local support modules absent from the source archive; their expected locations and interfaces are documented in the running notes. Dependency files are inferred from the original notebooks rather than validated environment lockfiles. The retained results have not been regenerated in a clean environment.
