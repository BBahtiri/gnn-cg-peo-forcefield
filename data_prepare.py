import pandas as pd
import numpy as np
import torch
import io
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from sklearn.model_selection import train_test_split

def prepare_data(file_path, cutoff_radius, outlier_percentile):
    with open(file_path, 'r') as f: lines = f.readlines()
    data_lines = [l for l in lines if not l.strip().startswith(('#', 'TIMESTEP', 'BOX_BOUNDS', 'N_CHAINS', 'N_BEADS', 'MONOMERS_PER_CHAIN')) and l.strip()]
    column_names = ['timestep', 'chain_id', 'monomer_id', 'bead_id', 'x', 'y', 'z', 'fx', 'fy', 'fz', 'mass', 'n_atoms', 'monomer_type']
    df_full = pd.read_csv(io.StringIO(''.join(data_lines)), sep=r'\s+', header=None, names=column_names)
    print(f"✅ Parsed {df_full['timestep'].nunique()} frames.")
    
    df_full['force_mag'] = np.linalg.norm(df_full[['fx', 'fy', 'fz']].values, axis=1)
    force_cutoff = np.percentile(df_full['force_mag'], outlier_percentile)
    print(f"Calculated {outlier_percentile}th percentile force: {force_cutoff:.2f} kcal/mol/Å.")
    outlier_timesteps = df_full[df_full['force_mag'] > force_cutoff]['timestep'].unique()
    df_filtered = df_full[~df_full['timestep'].isin(outlier_timesteps)]
    print(f"Removed {len(outlier_timesteps)} frames, leaving {df_filtered['timestep'].nunique()} frames.")
    if df_filtered.empty: raise ValueError("No frames remained after filtering.")
    
    all_graphs = []
    for _, frame_df in df_filtered.groupby('timestep'):
        frame_df = frame_df.sort_values('bead_id').reset_index(drop=True)
        monomer_types = ['first', 'middle', 'last']
        type_map = {t: i for i, t in enumerate(monomer_types)}
        node_cat = torch.tensor([type_map[t] for t in frame_df['monomer_type']], dtype=torch.long)
        node_features = torch.nn.functional.one_hot(node_cat, num_classes=len(monomer_types)).float()
        positions = torch.tensor(frame_df[['x', 'y', 'z']].values, dtype=torch.float)
        forces = torch.tensor(frame_df[['fx', 'fy', 'fz']].values, dtype=torch.float)
        bonded_edges = []
        for _, chain_df in frame_df.groupby('chain_id'):
            bead_indices = chain_df.index.tolist()
            for i in range(len(bead_indices) - 1):
                bonded_edges.extend([[bead_indices[i], bead_indices[i+1]], [bead_indices[i+1], bead_indices[i]]])
        bonded_edge_index = torch.tensor(bonded_edges, dtype=torch.long).t().contiguous()
        radius_edge_index = radius_graph(positions, r=cutoff_radius, loop=False)
        edge_index = torch.unique(torch.cat([bonded_edge_index, radius_edge_index], dim=1), dim=1)
        row, col = edge_index
        dist = torch.norm(positions[row] - positions[col], p=2, dim=-1).view(-1, 1)
        graph_data = Data(x=node_features, pos=positions, edge_index=edge_index, edge_attr=dist, y=forces)
        all_graphs.append(graph_data)
    print(f"✅ Constructed {len(all_graphs)} graph objects.")
    
    train_graphs, test_graphs = train_test_split(all_graphs, test_size=0.2, random_state=42)
    val_graphs, test_graphs = train_test_split(test_graphs, test_size=0.5, random_state=42)
    return train_graphs, val_graphs, test_graphs