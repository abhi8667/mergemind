# MergeMind 🚗🚙🚧

**MergeMind** is a multi-agent traffic merge benchmark where LLM agents learn cooperative behavior during highway zipper merges using reinforcement learning. This is a semantic coordination environment (not a physics simulator) designed for fast iteration, clear reward shaping, and a judge-friendly demo.

---

## 1) Problem Statement
Traffic merges are one of the most common coordination failures in the real world. When every driver acts selfishly, collisions and deadlocks happen, throughput drops, and aggressive behavior increases. MergeMind asks: *Can LLM agents learn social intelligence and cooperative merging?*

---

## 2) Why This Is Novel
- **Semantic multi-agent benchmark**: merges are modeled as a coordination game with structured observations and action choices.
- **Cooperation-first rewards**: safety, courtesy, and throughput are explicitly rewarded.
- **LLM-compatible**: observations can be serialized into prompts for lightweight PPO/GRPO training.

---

## 3) How The Environment Works
Two lanes merge into a single bottleneck. Each timestep, every car chooses an action (accelerate, brake, merge, yield, etc.). The environment exposes semantic observations:

```json
{
  "lane": "left",
  "speed": 2,
  "front_gap": 3,
  "merge_distance": 1,
  "right_car_waiting": true,
  "cars_behind": 1,
  "can_merge": true
}
```

Key mechanics:
- Right lane ends at the merge point (🚧)
- Collisions end an agent
- Episodes terminate when all cars clear or max steps reached
- Deterministic mode and reproducible seeds are supported

---

## 4) Reward Engineering
Composable reward components balance safety and cooperation:
- **Safety**: `-10` collision
- **Efficiency**: `+1` progress, `+2` successful merge
- **Courtesy**: `+3` correct yield
- **Anti-toxic**: `-2` blocking, `-1` tailgating, `-1` brake spam
- **Global**: `+5` when all cars clear efficiently

Safeguards:
progress is capped, repeated stalling is penalized, and aggressive blocking is discouraged.

---

## 5) Training Setup
### Lightweight baseline (default)
`train.py` includes a Q-learning baseline that trains quickly on CPU and produces a reward curve.

```bash
python train.py --mode qlearn --episodes 60
```

### TRL PPO scaffold (optional)
When `trl`, `transformers`, and `torch` are installed, you can run a minimal PPO loop:

```bash
python train.py --mode trl --episodes 10 --model-name sshleifer/tiny-gpt2
```

This keeps the pipeline extensible for larger LLMs (Phi, Gemma, TinyLlama) in Colab.

---

## 6) Results (Placeholders)
After training:
- `outputs/reward_curve.png`
- `outputs/train_metrics.json`

After evaluation:
- `outputs/eval_results.json`
- `outputs/eval_metrics.png`

---

## 7) Demo Instructions
Launch the Gradio demo:

```bash
python app.py
```

Tabs include:
1. **Live Replay** with emoji animations
2. **Before vs After Training**
3. **Metrics Dashboard**
4. **About MergeMind**

---

## 8) Hugging Face Spaces Deployment
1. Add `app.py` as the entry file.
2. Ensure `requirements.txt` is in the repo root.
3. Set the Space to **Gradio** and **Python 3.11**.
4. Push the repo to your Space.

---

## 9) Why It Matters Beyond Traffic
MergeMind models **real-world coordination failures** relevant to:
- multi-agent robotics
- autonomous systems safety
- cooperative decision-making in crowded environments
- reinforcement learning with social norms

---

## 10) Future Work
- richer negotiation (signals, intent commitments)
- larger maps with on-ramps and exits
- safety stress tests (aggressive agents)
- multi-objective optimization for fairness

---

## Project Layout
```
mergemind/
│── README.md
│── requirements.txt
│── app.py
│── train.py
│── evaluate.py
│── plots.py
│── openenv.yaml
│── .gitignore
│── env/
│   ├── mergemind_env.py
│   ├── rewards.py
│   ├── utils.py
│── ui/
│   ├── replay.py
│   ├── dashboard.py
│── data/
│── models/
│── outputs/
```

---

## Quickstart
```bash
pip install -r requirements.txt
python train.py
python evaluate.py
python app.py
```
