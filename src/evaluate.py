import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(model, X_test_t, y_test, scaler):
    model.eval()
    with torch.no_grad():
        pred_test = model(X_test_t).numpy()

    pred_original = scaler.inverse_transform(pred_test)
    y_original    = scaler.inverse_transform(y_test)

    mae  = mean_absolute_error(y_original, pred_original)
    rmse = np.sqrt(mean_squared_error(y_original, pred_original))

    return mae, rmse

def plot_loss(train_losses, val_losses, title='Loss Curve'):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses,   label='Val Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(y_original, pred_original, title='Predictions'):
    plt.figure(figsize=(10, 4))
    plt.plot(y_original,    label='Real')
    plt.plot(pred_original, label='Predicted')
    plt.title(title)
    plt.xlabel('Day')
    plt.ylabel('Temperature (C)')
    plt.legend()
    plt.grid(True)
    plt.show()
