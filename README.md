# Macrocircuit NeuroAI: Biological vs. Artificial Control

## Project Overview
This project investigates the impact of Neural Circuit Architectural Priors (NCAP) on the learning and generalization capabilities of reinforcement learning agents. We aim to bridge the gap between biological plausibility and artificial intelligence by comparing two distinct architectures controlling a multi-link swimmer in a physics-based environment (`dm_control`):

### 1. Biological NCAP Agent (The Experiment):
* A highly sparse, interpretable network designed to mimic biological Central Pattern Generators (CPGs).
* It utilizes oscillator-based structures and synaptic constraints to naturally generate rhythmic locomotion, similar to how simple organisms control movement.

### 2. Baseline MLP Agent (The Control):
* A standard, fully connected Multi-Layer Perceptron (MLP).
* This represents the "blank slate" approach common in Deep Learning, which must learn the physics of locomotion from scratch without any architectural guidance.

### The Goal
To determine if biological constraints—specifically sparse coding and pre-defined oscillatory structures—lead to faster learning, better energy efficiency, and more robust swimming behaviors compared to standard "black box" networks.



## Installation

### 1. Clone the repository:
```bash
git clone [https://github.com/YOUR-USERNAME/Macrocircuit-NeuroAI.git](https://github.com/YOUR-USERNAME/Macrocircuit-NeuroAI.git)
cd Macrocircuit-NeuroAI
```



### 2. Install Dependencies:
This project relies on `tonic` for RL and `dm_control` for physics.

```bash
pip install -r requirements.txt
```

> **Note:** If you encounter issues installing Tonic via pip, clone it locally and install it in editable mode:

```bash
git clone [https://github.com/neuromatch/tonic.git](https://github.com/neuromatch/tonic.git)
pip install -e tonic
```

## Usage

### 1. Training the Agents
To reproduce the experiment, run the main script. This will train the MLP Baseline first, followed by the NCAP Biological Agent.

```bash
python main.py
```

Configuration files and checkpoints will be saved to `data/local/experiments/tonic/`. By default, training runs for 100,000 steps. You can adjust this in `main.py` for longer convergence.

### 2. Visualizing the Results
To watch the trained agent and save a video of it swimming:

```bash
python visualize.py --path data/local/experiments/tonic/swimmer-swimmer/mlp_baseline
```

**Arguments:**
* `--path`: Path to the specific experiment folder (e.g., `mlp_baseline` or `ncap_biological`).
* `--checkpoint`: Use `last` to load the final model, or specific step number (e.g., 100000).

## Project Structure
* **model.py:** Contains the NCAP architecture definitions, including the SwimmerModule (Bio-Brain) and SwimmerActor.
* **main.py:** The experiment orchestrator. Configures the environment, defines the agents (MLP vs NCAP), and executes the training loops.
* **visualize.py:** A robust tool for loading trained models, rendering the simulation, and saving compatibility-friendly MP4 videos.
* **requirements.txt:** List of dependencies.
