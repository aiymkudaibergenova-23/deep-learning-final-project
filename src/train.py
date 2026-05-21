import torch
import torch.nn as nn

def train_model(model, X_train_t, y_train_t,
                X_val_t, y_val_t, epochs=50):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses   = []

    for epoch in range(epochs):
        model.train()
        pred_train = model(X_train_t)
        loss_train = criterion(pred_train, y_train_t)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(X_val_t)
            loss_val = criterion(pred_val, y_val_t)

        train_losses.append(loss_train.item())
        val_losses.append(loss_val.item())

        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs} | '
                  f'Train: {loss_train.item():.4f} | '
                  f'Val: {loss_val.item():.4f}')

    return train_losses, val_losses
