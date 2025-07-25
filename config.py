# config.py

# --- Data Configuration ---
FILE_PATH = r"C:\Users\betim\Documents\gnn_cg_peo\peo50_cg_trajectory.dat"
CUTOFF_RADIUS = 12.0
OUTLIER_PERCENTILE = 99.0

# --- Model Hyperparameters ---
IN_DIM = 3
HIDDEN_DIM = 128
EDGE_DIM = 64
NUM_LAYERS = 4

# --- Training Hyperparameters ---
LEARNING_RATE = 5e-4
BATCH_SIZE = 8
EPOCHS = 1000
WEIGHT_DECAY = 1e-5
PATIENCE = 50