# Rumor Mill

Rumor Mill is a simulated corporate-information environment where an agent must distinguish truth, gossip, distortion, and manipulation.

The environment models a work week inside a noisy company. A learning agent receives Slack-like messages, anonymous forum posts, and 1-on-1 responses from characters with hidden agendas. It must decide who to trust, when to wait, and when to act.

## Project structure

```text
Rumour-Mill/
|-- environment/
|   |-- rumor_env.py
|   |-- characters.py
|   |-- ground_truth.py
|   `-- reward.py
|-- training/
|   |-- train_agent.py
|   `-- config.py
|-- demo/
|   |-- visualize.py
|   `-- sample_episodes.py
|-- evaluation/
|   `-- metrics.py
|-- requirements.txt
|-- pyproject.toml
|-- Dockerfile
`-- README.md
```

## Core idea

The agent never sees ground truth directly.

It only sees noisy social signals from:

- `Spinner`
- `Gossip`
- `Quiet One`
- `Politician`
- `Leaker`

The environment hides the actual event, such as:

- layoffs
- revenue miss
- promotion politics

The reward function balances:

- correct decisions
- timing
- social capital
- end-of-week truth alignment

## Quick start

Follow these commands from:

```powershell
cd c:\Users\pooja\Rumour-Mill
```

### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

This installs only the lighter local demo dependencies.

### 3. Run a sample episode to verify the environment works

```powershell
python demo\sample_episodes.py
```

### 4. Launch the Streamlit demo UI

```powershell
streamlit run demo\visualize.py
```

This should open the UI in your browser. If it does not open automatically, Streamlit will print a local URL in the terminal.

### 5. Optional: run the training script

```powershell
python training\train_agent.py
```

Important:

- the training script is heavier than the demo scripts
- it uses `trl`, `transformers`, and `unsloth`
- you may need to install extra training packages manually before this works
- for many setups, training is better on Colab or a GPU machine
- local development should start with the sample episode and Streamlit UI first

## Teammate setup checklist

If someone on the team is opening the project for the first time, these are the minimum commands they should run:

```powershell
cd c:\Users\pooja\Rumour-Mill
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python demo\sample_episodes.py
streamlit run demo\visualize.py
```

## Training

The starter training script is in [training/train_agent.py](/c:/Users/pooja/Rumour-Mill/training/train_agent.py:1).

It currently uses:

- `TRL`
- `PPOTrainer`
- `Unsloth`

Run it with:

```powershell
python training\train_agent.py
```

Note:

- this is a starter script, not a production RL pipeline
- you will likely run training on Colab or a Linux GPU box
- if `unsloth` is difficult on your local machine, keep local runs for environment debugging and train remotely

## Docker

Build the image:

```powershell
docker build -t rumor-mill .
```

Run the sample demo:

```powershell
docker run --rm rumor-mill python demo/sample_episodes.py
```

Run the Streamlit UI locally:

```powershell
streamlit run demo\visualize.py
```

## Important files

- [environment/rumor_env.py](/c:/Users/pooja/Rumour-Mill/environment/rumor_env.py:1): main environment loop
- [environment/characters.py](/c:/Users/pooja/Rumour-Mill/environment/characters.py:1): NPC logic
- [environment/ground_truth.py](/c:/Users/pooja/Rumour-Mill/environment/ground_truth.py:1): hidden scenario generation
- [environment/reward.py](/c:/Users/pooja/Rumour-Mill/environment/reward.py:1): reward calculation
- [training/config.py](/c:/Users/pooja/Rumour-Mill/training/config.py:1): training constants
- [evaluation/metrics.py](/c:/Users/pooja/Rumour-Mill/evaluation/metrics.py:1): evaluation helpers

## Colab note

If you train in Colab, install from `requirements.txt` selectively and add your Hugging Face token through Colab secrets or environment variables, not directly in code.

## Status

This repo is currently an MVP scaffold:

- environment works as a simple simulation
- training loop is a starter baseline
- demo scripts are CLI-based

Next natural steps:

1. make actions more structured
2. improve reward shaping
3. add OpenEnv integration if you want hosted stateful environment training
4. build a proper Streamlit or Gradio demo
