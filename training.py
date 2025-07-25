import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import config

def train_and_evaluate(model, train_loader, val_loader, test_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.MSELoss()

    print("\n🚀 Starting training with Early Stopping...")
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(config.EPOCHS):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred_forces = model(batch)
            loss = criterion(pred_forces, batch.y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch.num_graphs
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred_forces = model(batch)
                loss = criterion(pred_forces, batch.y)
                total_val_loss += loss.item() * batch.num_graphs
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= config.PATIENCE:
            print(f'\nEarly stopping triggered after {epoch + 1} epochs.')
            break

    print("✅ Training complete.")
    
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    true_forces_normalized, pred_forces_normalized = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds = model(batch)
            pred_forces_normalized.append(preds.cpu())
            true_forces_normalized.append(batch.y.cpu())

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "true_forces_norm": torch.cat(true_forces_normalized, dim=0),
        "pred_forces_norm": torch.cat(pred_forces_normalized, dim=0)
    }

def plot_results(results, force_mean, force_std):
    true_forces = (results["true_forces_norm"] * force_std) + force_mean
    pred_forces = (results["pred_forces_norm"] * force_std) + force_mean
    test_rmse = torch.sqrt(torch.mean((true_forces - pred_forces)**2))
    
    print(f'\n--- Final Evaluation ---')
    print(f'Test Set Force RMSE: {test_rmse:.4f} kcal/mol/Å')

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].plot(results["train_losses"], label='Training Loss')
    axes[0].plot(results["val_losses"], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training & Validation Loss Curves')
    axes[0].legend()
    axes[0].set_yscale('log')

    fx_true = true_forces[:, 0].numpy()
    fx_pred = pred_forces[:, 0].numpy()
    axes[1].scatter(fx_true, fx_pred, alpha=0.3, s=10)
    axes[1].plot([fx_true.min(), fx_true.max()], [fx_true.min(), fx_true.max()], 'r--', label='Ideal (y=x)')
    axes[1].set_xlabel('True Fx (kcal/mol/Å)')
    axes[1].set_ylabel('Predicted Fx (kcal/mol/Å)')
    axes[1].set_title('Parity Plot for Fx Component')
    axes[1].legend()
    axes[1].axis('equal')

    plt.tight_layout()
    plt.show()