# Tropical Simplex Algorithm for Certifying Properties of ReLU-based Artificial Neural Networks

This repository contains the official implementation of the Tropical Simplex Algorithm for the formal verification of ReLU-based Deep Neural Networks (DNNs). The framework leverages Tropical Geometry to model the decision boundaries of neural networks as tropical hypersurfaces, transforming the verification problem into a tropical linear programming problem.


## 📦 Dependencies and Environment Setup

The repository is modularly structured. Depending on your goal, you can run the standalone solver or the full neural network abstraction pipeline.

### 1. Simplex Solver Only
To run the custom tropical simplex solver (located in `simplex_python/`), the only required dependency is `numpy`. No external mathematical or optimization libraries are needed.

### 2. Full Verification Pipeline (Recommended)
To run the complete pipeline, which includes training a model, abstracting the neural network, and running the verification solver (e.g., executing `MLP_classic.ipynb`, `MLP_Tropical.ipynb`, and `NN_to_LP.py`), a complete Python environment is needed. 

It is recommended to manage the environment via`conda`:

```bash
conda create -n tropical_env python=3.10
conda activate tropical_env
pip install numpy torch torchvision pandas matplotlib tqdm
```
*(Note: You can also use `pip install -r requirements.txt` if provided).*

## 🚀 Quick Start & Reproducibility

To facilitate the review process, you can immediately test the tropical simplex algorithm without training a new neural network from scratch.

### Option A: Run the Solver on Pre-generated Problems
You can directly run the simplex solver on the mathematical problems already provided in the `simplex_python/problems/` directory. For example:
```bash
python simplex_python/main.py simplex_python/problems/generic_lp_2D.lp
```

### Option B: End-to-End Abstraction and Verification
If you have a pre-trained PyTorch model, saved in `.pt` file extention (e.g., `simple_model.pt`), you can abstract it and generate the linear programming formulation ready for the tropical simplex. 

Here is a ready-to-run example command for abstraction:
```bash
python tropical_abstraction.py simple_model.pt -xlb -1.0 -1.0 -xub 1.0 1.0 -o network.lp
```
*Note: You can find automatically formatted CLI commands with specific bounds generated at the end of the `MLP_classic.ipynb` notebook.*

## 📖 Mapping Code to the Paper Theory

To help reviewers navigate the codebase and verify the claims made in the long paper, here is a mapping between the theoretical concepts and their implementation:

* **Network Abstraction to Tropical Maps (Theorem 1 & Proposition 3):** The logic connecting the neural network weights to tropical rational maps is implemented in `tropical_abstraction.py`. You will find explicit comments referring to the paper's theorems within the code.
* **Tropical Simplex Algorithm:** The core logic of the solver is contained within the `simplex_python/` directory.
* **Phase I / Phase II Perturbations:** The handling of non-generic cases and perturbed linear programming is strictly implemented in `simplex_python/perturbed_lp.py`.

## 📂 Repository Structure

* `MLP_classic.ipynb` / `MLP_Tropical.ipynb`: Notebooks for training and evaluating standard and tropical MLPs (MNIST dataset).
* `NN_to_LP.py` / `tropical_abstraction.py`: Scripts for converting the trained PyTorch neural networks into tropical linear programming formalizations. `NN_to_LP.py` is used for converting tropical MLPs, while `tropical_abstraction.py` is used to convert standard MLPs.
* `simplex_python/`: The core module containing the custom tropical simplex solver.
    * `problems/`: Directory containing pre-formulated `.lp` test problems.
    * `test_classes/`: Unit tests for the algebraic and solver components.
