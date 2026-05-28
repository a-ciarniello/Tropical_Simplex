# Tropical Simplex Algorithm

This repository contains the source code for my thesis project, titled "A tropical simplex algorithm for certifying properties of ReLU-based artificial neural networks". This work explores the application of the Tropical Simplex Algorithm as a novel tool for the formal certification of Neural Network properties. The repository provides a comprehensive suite of tools designed to train both Classic (ReLU) and Tropical Multi-Layer Perceptrons (MLPs), mathematically extract tropical linear programming (TLP) constraints from them, and efficiently solve these constraints relying on a custom Python implementation of the Tropical Simplex algorithm.
## Repository Structure

* **`MLP_classic.ipynb`**: Notebook to train a standard ReLU MLP on a binary subset of the MNIST dataset.
* **`MLP_Tropical.ipynb`**: Notebook to train a custom Tropical MLP (using max-plus affine transformations and softly bounded biases) on MNIST.
* **`tropical_abstraction.py`**: Implements the zone-based tropical abstraction for standard linear layers and exports the abstraction into a custom max-plus `.lp` format.
* **`NN_to_LP.py`**: Converts a natively trained Tropical MLP checkpoint into an exact tropical LP representation.
* **`simplex_python/`**: A complete, custom implementation of the Tropical Simplex algorithm in Python, featuring phase I/II perturbation methods, tangent digraph combinatorics, and custom parsing for `.lp` files.

---

## Workflows

The repository supports two main workflows depending on the architecture of the neural network being analyzed:

### 1. Classic ReLU MLP Flow
This workflow applies a zone-based tropical polyhedral abstraction to a standard ReLU network to evaluate its bounds.

1. **Train the model**: 
   Run the `MLP_classic.ipynb` notebook. This will train a classic MLP and save the model weights as a `.pt` file (e.g., `simple_model.pt`).
2. **Generate the Tropical LP**: 
   Use the `tropical_abstraction.py` script to compute the abstraction of the trained model over a specific input region (hypercube) and export it to an `.lp` file.
   ```bash
   python tropical_abstraction.py simple_model.pt -xlb <lower_bounds_list> -xub <upper_bounds_list> -o network.lp
3. **Solve the constraints**:
   Execute the custom tropical simplex solver on the generated problem to find the optimum.
   ```bash
   python simplex_python/main.py network.lp

### 2. Tropical MLP Flow
This workflow builds, trains, and verifies a neural network built with native tropical layers.

1. **Train the model**:
   Run the `MLP_Tropical.ipynb` notebook. This trains the TropMLP architecture and saves the weights as a `.pt` file (e.g., tropical_model.pt).
2. **Convert to LP**:
   Since the network is natively tropical, use `NN_to_LP.py` to directly translate the trained checkpoint into max-plus linear programming constraints.
   ```bash
   python NN_to_LP.py tropical_model.pt -o network_tropical.lp
3. **Solve the constraints**:
   Pass the resulting `.lp` file to the tropical simplex solver.
   ```bash
   python simplex_python/main.py network_tropical.lp


To run the custom tropical simplex solver, no external math libraries are required beyond `numpy`.
