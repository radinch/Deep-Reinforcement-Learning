# Running the Projects

## Environment setup

Launch JupyterLab from the selected project directory so relative paths resolve correctly:

```bash
cd projects/01-value-based-learning
python -m pip install -r requirements.txt
python -m jupyterlab
```

A virtual environment per project is recommended when switching between legacy Gym, Gymnasium, MuJoCo, and TorchRL. The notebooks came from different runtimes; the dependency files are inferred from imports and installation cells, not tested lockfiles. Retain the working package versions after a successful run.

Notebook installation cells may duplicate the requirements files. The continuous-control notebook also contains a CUDA-specific PyTorch installation cell; skip or adapt it for the selected machine. Notebook outputs are retained from the source archive and do not represent fresh validation of this repository.

## Missing local modules

### Continuous control

Place the original `HW3_moving_target_reacher_env.py` beside `deterministic_actor_critic.ipynb`. It must provide `make_moving_reacher`, `rollout_to_video`, and `embed_video`. The custom environment is not part of standard Gymnasium. Its original import name is retained for compatibility.

### Deep exploration

Three implementation modules were recovered directly from the notebook's `%%writefile` cells:

- `modules/student_metric.py`
- `modules/student_rnd.py`
- `modules/student_bootstrap.py`

Restore the original benchmark support package into `modules/`, including `metrics`, `rnd_dqn`, `bootstrap_dqn`, `hw_plots`, `hw_experiment`, and `hw_grading`, along with any modules they depend on. These import names remain unchanged to preserve the original API. The notebook can then load the recovered implementations before their corresponding write/reload cells.

`results/` contains the supplied normal/noisy run artifacts, summary files, fixtures, and metadata. Recovering these files does not recover the missing benchmark implementation.

## Project-specific details

| Project | Setup detail |
| --- | --- |
| Value-based learning | LunarLander rendering requires Box2D and OpenCV. Some systems need SWIG to build Box2D dependencies. |
| Policy gradient methods | REINFORCE uses legacy `gym`; MiniTetris uses an environment defined in the notebook. Legacy Gym can require a NumPy version compatible with its API. |
| Continuous control | Requires MuJoCo and the missing custom Reacher wrapper. |
| Model-based planning | Uses Gymnasium; rendering can require additional system libraries. |
| Contextual bandits | Requires access to TweetEval or the local CSV cache described in the notebook. The cache is not supplied. |
| Deep exploration | Restore the local support package described above. |
| Imitation and offline learning | MiniGrid is required for DoorKey; other notebooks use classic-control environments. |
| Multi-agent learning | Cooperative navigation uses TorchRL, TensorDict, and VMAS. Its rendering cells install Linux display libraries and start a virtual display. Adapt them for other systems. |
| Meta-reinforcement learning | Requires MuJoCo. Full multi-seed experiments can be substantially longer than notebook inspection. |

## Working with artifacts

Recorded video files are in each project's `media/` directory. The value-based and REINFORCE notebook paths were adjusted to use this directory. Exploration write cells now target `modules/`; its export cell writes `exploration_artifacts.zip`.

Preserve the checked-in outputs when inspecting the original experiments. Rerunning cells may overwrite videos, result files, or notebook outputs.
