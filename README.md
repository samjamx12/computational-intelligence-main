# Computational Intelligence Projects

A collection of four self-contained projects implementing classic algorithms from computational intelligence — covering **continuous optimization**, **MCMC sampling**, **evolutionary algorithms**, and **neural networks** — each applied to a small, well-defined problem and analyzed in a Jupyter notebook.

Every algorithm in this repository is implemented from scratch (NumPy / PyTorch only; no high-level optimization libraries) and the notebooks discuss what each method does, how the hyperparameters affect behavior, and what the trade-offs are between methods of the same family.

## What's inside

| # | Notebook | What it covers |
|---|----------|----------------|
| 1 | `01_optimization.ipynb` | **Gradient Descent vs Derivative-Free Optimization** on a non-convex 2D test function. Analytical gradient, step-size sweep, random-search DFO baseline, side-by-side discussion. |
| 2 | `02_sampling.ipynb` | **Metropolis-Hastings and Simulated Annealing** on a bimodal 2D Gaussian mixture. Random-walk proposals, acceptance ratios, logarithmic cooling schedule, hyperparameter sweep over `(std, T0, C)`. |
| 3 | `03_evolutionary_algorithm.ipynb` | **Evolutionary algorithm** for parameter estimation of the **gene repressilator** (a stiff 6-state ODE). Rank selection, arithmetic crossover, (μ + λ) elitist survival, treating the ODE solver as a black-box objective. |
| 4 | `04_neural_networks.ipynb` | **MLP vs CNN** image classifiers in PyTorch on the scikit-learn 8x8 digits dataset. Custom Dataset class, generic classifier wrapper, training loop with early stopping, full evaluation pipeline. |

## Repository layout

```
computational-intelligence-projects/
├── notebooks/
│   ├── 01_optimization.ipynb
│   ├── 02_sampling.ipynb
│   ├── 03_evolutionary_algorithm.ipynb
│   └── 04_neural_networks.ipynb
├── src/
│   ├── optimization.py            # GradientDescent + RandomSearchDFO
│   ├── sampling.py                # MetropolisHastings + SimulatedAnnealing
│   ├── evolutionary/
│   │   ├── ea.py                  # rank selection, crossover, elitist survival
│   │   └── repressilator.py       # ODE model + black-box objective
│   └── classifier/
│       ├── dataset.py             # PyTorch Dataset for the 8x8 digits
│       ├── model.py               # Reshape / Flatten / build_mlp / build_cnn / ClassifierNeuralNet
│       └── train.py               # training loop, evaluation, plotting
├── results/
│   ├── 01_optimization/           # contour + GD + DFO trajectory figures
│   ├── 02_sampling/               # target density + MH samples + SA sweep
│   ├── 03_evolutionary_algorithm/ # data, EA fit, population scatter, convergence
│   └── 04_neural_networks/        # sample digits, validation curves (PNG + PDF), test losses
├── requirements.txt
├── .gitignore
└── README.md
```

## Technologies used

Python 3.10+, NumPy, SciPy, scikit-learn, PyTorch, Matplotlib, Jupyter Notebook.

## How to run it

```bash
git clone <this-repo>
cd computational-intelligence-projects

python -m venv venv
source venv/bin/activate              # on Windows: venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook
```

Then open any of the four notebooks in `notebooks/`. Each notebook is self-contained — you can run them in any order, and you don't need to import anything from `src/` to use them.

You can also import the algorithms directly:

```python
from src.optimization import GradientDescent, grad
from src.sampling import MetropolisHastings, SimulatedAnnealing
from src.evolutionary.ea import EA
from src.evolutionary.repressilator import Repressilator
from src.classifier.dataset import Digits
from src.classifier.model import build_cnn, build_mlp, ClassifierNeuralNet
from src.classifier.train import training, evaluation
```

## Datasets

No external downloads are required.
- Projects 1, 2, and 3 use analytical objectives (a 2D function, a 2D mixture density, and an ODE-generated synthetic dataset with added Gaussian noise).
- Project 4 uses the scikit-learn `digits` dataset, which ships with `scikit-learn` and is loaded via `sklearn.datasets.load_digits()`.

## Results

Every notebook produces figures into the matching `results/` subfolder. A summary of the headline numbers / observations:

**1. Optimization.** Gradient descent with step sizes around 0.05–0.1 reliably reaches the global minimum within ~20 iterations. Step sizes ≥ 0.25 oscillate or diverge along the steeper axis. Random-search DFO eventually finds good regions but converges much more slowly and shows large run-to-run variance — exactly as expected for a method that ignores all gradient information.

**2. Sampling.** Metropolis-Hastings with `std = 0.1` mixes locally and rarely escapes its starting mode within 1500 iterations. With `std = 0.5` the chain traverses both modes, but the acceptance ratio drops noticeably. Simulated annealing with slow cooling (`C = 0.1`, `T0 = 1.0`) produces the best samples in the sweep — both modes are visited and the chain concentrates near the dominant mode at `(-3, -3)` as the temperature drops. Aggressive cooling (`C = 10`) freezes the chain into whatever mode it discovered first.

**3. Evolutionary algorithm.** The EA recovers the oscillation period and amplitude of the observed mRNA channels of the gene repressilator, and the unobserved protein channels — which were not used in the loss — also come out roughly correct, indicating the algorithm found a parameter set close to the true dynamics. Convergence is fast in the first ~10 generations and then plateaus, with some residual phase drift in the fit. This is the diversity-collapse problem flagged in the analysis: without mutation, arithmetic crossover drives the population toward the centroid and the search stalls. Adding a Gaussian mutation operator is the most obvious next step.

**4. Neural networks.** With the default hyperparameters (`M=256`, `num_kernels=32`, `lr=1e-3`, `wd=1e-5`, `patience=20`), one typical run reaches:

| Model | Test NLL | Test classification error |
|-------|----------|----------------------------|
| MLP   | 0.256    | 6.94%                      |
| CNN   | 0.136    | 3.80%                      |

The CNN converges to a lower validation NLL, generalizes about 3 percentage points better than the MLP on classification error, and trains for more epochs before early stopping kicks in. The CNN's translation equivariance and parameter sharing pay off measurably here even at very low resolution.

## What I learned

- **Gradient-based vs gradient-free optimization** — when the gradient is available and the objective is smooth, GD (with a sensible step size) crushes random search. When the objective is non-differentiable, noisy, or only available as a black-box simulator, gradient-free methods are still useful, but pure random search is an unimpressive baseline; smarter DFO methods (CMA-ES, Bayesian optimization, Nelder–Mead) close most of the gap.
- **The mixing-vs-acceptance trade-off in MCMC** is visible directly in the acceptance ratio and the spread of samples — small steps are accepted often but explore little, large steps explore broadly but waste evaluations.
- **Why simulated annealing is fundamentally an optimizer**, not a sampler — its stationary distribution is biased toward the global mode by the temperature schedule.
- **Evolutionary algorithms as black-box optimizers** — phrasing parameter estimation in a stiff biological ODE as a fitness-evaluation problem, and validating a fit on hidden channels (proteins) that were not used in the loss as a practical sanity check.
- **The diversity-vs-exploitation trade-off in EAs** — pure crossover collapses the population toward the centroid; mutation is what keeps an EA exploring.
- **A complete PyTorch training pipeline from scratch** — Dataset + DataLoader + a generic classifier wrapper + a training loop with early stopping + evaluation. This is the kind of skeleton that scales well to bigger problems.
- **Why CNNs beat MLPs on images** — translation equivariance, local connectivity, and parameter sharing — and that the gap is measurable even on 8x8 inputs.

## Future improvements

- **Project 1.** Add momentum / Adam / line-search variants of GD; replace random-search DFO with CMA-ES or Nelder–Mead and compare convergence rates.
- **Project 2.** Add an HMC baseline; quantify mixing with effective sample size (ESS); try a geometric cooling schedule and compare against the logarithmic one.
- **Project 3.** Add Gaussian mutation with a 1/5-success-rule step size, or full CMA-ES; replace arithmetic crossover with BLX-α or SBX; run a population-size sweep and report wall-clock cost vs final fitness.
- **Project 4.** Add data augmentation (small rotations / shifts); scale the same pipeline to MNIST and SVHN; replace Adamax with cosine-annealing SGD; add a confusion matrix and per-class precision/recall to the evaluation report; turn the training script into a CLI entrypoint.
