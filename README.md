# Rocket League ML

An experimental machine learning pipeline for analyzing Rocket League replays, predicting playstyle decisions, and eventually providing feedback on optimal choices.

## Setup rrrocket.exe

This project uses [rrrocket](https://github.com/SaltieRL/rrrocket) to parse Rocket League replays.

1. Download the Windows binary (`rrrocket.exe`) from the rrrocket releases page.
   https://github.com/nickbabcock/rrrocket
2. Place it into the `bin/` folder at the root of this repo:
3. Make sure it is executable (on Linux/macOS you may need `chmod +x rrrocket`).

The Python code will call this binary directly to parse `.replay` files into JSON, which is then converted into feature datasets.

## Quickstart

1. Clone the repo:

```bash
git clone https://github.com/<your-username>/rocketleague-ml.git
cd rocketleague-ml
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv\Scripts\activate
pip install -r requirements.txt
```

3. Place .replay files into data/raw/.
4. Preprocess data:

```bash
python -m scripts.preprocess.py
```

5. Train a model:

```bash
python -m scripts.train_model.py
```

6. Evaluate:

```bash
python -m scripts.evaluate_model.py
```
