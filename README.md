# Graph Neural Network Force Field for Coarse-Grained PEO

This project implements a Graph Neural Network (GNN) to predict atomic forces for a coarse-grained Polyethylene Oxide (PEO) molecular dynamics trajectory. The model architecture is inspired by the GNNFF paper, which uses a rotationally covariant framework to predict forces directly from the system's geometry.

## Key Features
* **Direct Force Prediction:** The GNN learns to predict the 3D force vector on each coarse-grained bead directly, bypassing the need for a potential energy surface.
* **Rotationally Covariant:** The model's architecture is explicitly designed to respect the rotational physics of forces by predicting interaction magnitudes and combining them with direction vectors.
* **Automated Data Cleaning:** The pipeline automatically identifies and filters out simulation frames with unphysical, high-force outliers to ensure stable model training.
* **Modular Codebase:** The project is organized into separate modules for configuration, data preparation, model architecture, and training for clarity and maintainability.

---
## Architecture and Methodology

The project follows a pipeline that transforms raw trajectory data into a trained GNN model.

### 1. Data Preparation & Graph Construction

Each snapshot from the molecular dynamics simulation is treated as a graph $\mathcal{G}=(\mathcal{V},\mathcal{E})$.

* **Data Filtering:** An initial analysis of the dataset revealed the presence of extremely high forces, likely due to mapping artifacts. To create a stable training set, any simulation frame containing a force magnitude in the top 1% of the entire dataset is removed.
* **Normalization:** The target forces are normalized (Z-score) using the mean and standard deviation calculated **from the training set only**. This stabilizes the training process.
* **Nodes ($\mathcal{V}$):** The coarse-grained beads are the nodes of the graph. Each node is assigned a one-hot encoded feature vector corresponding to its type (`first`, `middle`, or `last` monomer).
* **Edges ($\mathcal{E}$):** Edges represent interactions between beads. They are created using a hybrid approach:
    1.  **Covalent Bonds:** Edges are explicitly added between adjacent beads in the polymer chain.
    2.  **Non-Covalent Interactions:** Edges are added between any two beads (bonded or not) within a fixed **cutoff radius**.

### 2. GNN Model Architecture

The key components of the model, inspired by GNNFF, are:

* **Initial Embeddings:**
    * **Nodes:** The one-hot feature vector for each node is passed through a linear layer to create a high-dimensional state vector, $h_i^0$.
    * **Edges:** The scalar distance $d_{ij}$ between two connected nodes is expanded into a feature vector using a **Gaussian Basis Filter**. This creates the initial edge embedding, $h_{(i,j)}^0$.
        $$h_{(i,j)}^0 = \text{exp}(-\gamma (d_{ij} - \mu_k)^2) \quad \text{for } k=1...N_{gaussians}$$

* **Message Passing Blocks:**
    The model uses a series of interaction blocks where both **node and edge embeddings are iteratively updated**. In each block, nodes aggregate information from their neighbors to update their state, and then the new node states are used to update the state of the edges connecting them. A `LayerNorm` is applied after each node update to stabilize training.

* **Covariant Force Prediction:**
    This is the core of the GNNFF method. The final, updated edge embedding $h_{(i,j)}^L$ is used by a readout network to predict a **scalar force magnitude**, $n_{ij}$. The total 3D force vector $\vec{F}_j$ on a given atom $j$ is then calculated as the vector sum of all contributions from its neighbors $i$:
    $$\vec{F}_{j}=\sum_{i\in N_{j}}n_{ij}\vec{u}_{ij}$$
    where $\vec{u}_{ij}$ is the unit vector pointing from atom $i$ to atom $j$. This construction ensures the final forces are rotationally covariant.

---
## Project Structure

The codebase is organized into the following modules:

* `config.py`: Contains all hyperparameters and configuration settings.
* `data_prepare.py`: Includes the `prepare_data` function for loading, filtering, and creating graph datasets.
* `model.py`: Defines the GNN architecture, including the `GaussianFilter`, `InteractionBlock`, and `GNNForceField` classes.
* `training.py`: Contains the `train_and_evaluate` function for the training loop and `plot_results` for visualization.
* `main.py`: The main entry point that orchestrates the entire pipeline.

---
## How to Run

1.  **Clone the Repository (Optional):**
    ```bash
    git clone [https://github.com/BBahtiri/gnn-cg-peo-forcefield.git](https://github.com/BBahtiri/gnn-cg-peo-forcefield.git)
    cd gnn-cg-peo-forcefield
    ```

2.  **Set Up Environment:** It's recommended to use a virtual environment like Conda or venv.
    ```bash
    conda create -n gnnff_env python=3.9
    conda activate gnnff_env
    ```

3.  **Install Dependencies:**
    First, install PyTorch according to your system's specifications (CPU or CUDA) from the [official website](https://pytorch.org/get-started/locally/). Then, install the required packages.
    ```bash
    # For PyTorch Geometric dependencies, replace with the command from their website
    # that matches your PyTorch/CUDA version. Example for PyTorch 2.3 + CPU:
    pip install torch_geometric torch-cluster -f [https://data.pyg.org/whl/torch-2.3.0+cpu.html](https://data.pyg.org/whl/torch-2.3.0+cpu.html)

    pip install pandas numpy scikit-learn matplotlib
    ```

4.  **Configure:**
    Open `config.py` and ensure the `FILE_PATH` points to the correct location of your `peo50_cg_trajectory.dat` file. You can also adjust model and training hyperparameters here.

5.  **Run the Pipeline:**
    Execute the main script from your terminal.
    ```bash
    python main.py
    ```

The script will print its progress, including data preparation, the training loop with loss values, the final RMSE on the test set, and will display the loss curve and parity plots upon completion.
