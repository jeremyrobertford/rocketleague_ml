# Rocket League ML

An experimental machine learning pipeline for analyzing Rocket League replays, predicting playstyle decisions, and eventually providing feedback on optimal choices.

## Quickstart

1. Clone the repo:

```bash
git clone https://github.com/<your-username>/rocketleague-ml.git
cd rocketleague-ml
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

3. Place .replay files into data/raw/.
4. Preprocess data:

```bash
python scripts/preprocess.py
```

5. Train a model:

```bash
python scripts/train_model.py
```

6. Evaluate:

```bash
python scripts/evaluate_model.py
```
