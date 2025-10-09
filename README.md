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

```powershell
git clone https://github.com/<your-username>/rocketleague-ml.git
cd rocketleague-ml
```

2. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Place .replay files into data/raw/.
4. Preprocess data:

```powershell
python -m scripts.preprocess.py
```

5. Train a model:

```powershell
python -m scripts.train_model.py
```

6. Evaluate:

```powershell
python -m scripts.evaluate_model.py
```
