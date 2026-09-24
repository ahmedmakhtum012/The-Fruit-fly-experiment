# The-Fruit-fly-experiment

This repository contains every information rearding the Fruit fly brainmapping and itx exposure to the X application with limited(but graceful options) provided to the fly's brain to interact on the platform.

## Setting up the Environment

you'll need atlest python3.9=<, neuprint-python and Python-dotenv for that execute

```bash
python -m venv .venv

# if you're on Windows
.venv\Scripts\Activate.bat

# you're using Unix based terminal
source .venv/bin/activate

# then
pip install -r requirements.txt
```

and the environment is ready

## Phase 1(caching the male CNS data)

Just run the script called [fetch_neurons.py](fetching/fetch_neurons.py)

```bash
python fetching/fetch_neuron.py
```

## Phase 2(the connectome sim engine)