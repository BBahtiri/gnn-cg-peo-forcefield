import torch
from torch_geometric.loader import DataLoader
import config
from data_prepare import prepare_data
from model import GNNForceField
from training import train_and_evaluate, plot_results

def main():
    # --- Prepare Data ---
    train_data, val_data, test_data = prepare_data(
        config.FILE_PATH, config.CUTOFF_RADIUS, config.OUTLIER_PERCENTILE
    )

    # --- Normalize Target Forces ---
    print("\n--- Normalizing Target Forces ---")
    all_forces_train = torch.cat([data.y for data in train_data], dim=0)
    force_mean = all_forces_train.mean(dim=0)
    force_std = all_forces_train.std(dim=0)
    
    for data_list in [train_data, val_data, test_data]:
        for graph in data_list:
            graph.y = (graph.y - force_mean) / force_std

    # --- Create DataLoaders ---
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE)
    print("\n✅ DataLoaders created.")

    # --- Initialize Model ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = GNNForceField(
        in_dim=config.IN_DIM,
        hidden_dim=config.HIDDEN_DIM,
        edge_dim=config.EDGE_DIM,
        num_layers=config.NUM_LAYERS
    ).to(device)

    # --- Train and Evaluate ---
    results = train_and_evaluate(model, train_loader, val_loader, test_loader, device)

    # --- Plot Results ---
    plot_results(results, force_mean, force_std)

if __name__ == '__main__':
    main()