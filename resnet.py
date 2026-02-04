import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import os
import pickle
from scipy.signal import periodogram
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from scipy.stats import spearmanr
import subprocess

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

device = "cpu" # device to use

def parse_ppg(ppg_str):
    parts = ppg_str.split("|")
    y_values = []
    for part in parts[1:]:
        if part.strip() == "":
            continue
        _, y_val = part.split(",")
        y_values.append(float(y_val))
    return np.array(y_values)

df = pd.read_csv("data.csv")
df = df[df['p4205_i0'].notna()]
#df = df.sample(n=10000, random_state=seed)
df['y_values'] = df['p4205_i0'].apply(parse_ppg)

X = np.stack(df['y_values'].values)
X = (X - X.min(axis=1, keepdims=True)) / X.ptp(axis=1, keepdims=True)
Y = df['p21003_i0'].values

# 80/20 train/test split
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

# 70/15/15 train/val/test
#X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=seed)
#X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=seed)




class ResidualUnit(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, kernel_size=17, stride=1, dropout_rate=0.2, 
                 preactivation=True, postactivation_bn=False, activation_function='relu'):
        super(ResidualUnit, self).__init__()
        self.preactivation = preactivation
        self.postactivation_bn = postactivation_bn
        self.dropout_rate = dropout_rate

        # Activation function
        if activation_function == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            raise NotImplementedError("Activation function '{}' not implemented.".format(activation_function))

        self.bn1 = nn.BatchNorm1d(n_filters_in)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, kernel_size, stride=stride, padding=kernel_size//2 - 1, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.downsample = stride != 1 or n_filters_in != n_filters_out
        if self.downsample:
            self.conv_shortcut = nn.Conv1d(n_filters_in, n_filters_out, 1, stride=stride, bias=False)

    def forward(self, x):
        identity = x

        out = x
        if self.preactivation:
            out = self.bn1(out)
            out = self.activation(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.activation(out)
        if self.dropout_rate > 0:
            out = self.dropout(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_shortcut(identity)

        out += identity
        if not self.preactivation or self.postactivation_bn:
            out = self.bn2(out)
            out = self.activation(out)
        return out

class ResNet1DRegression(nn.Module):
    def __init__(self, input_length=100, kernel_size=16):
        super(ResNet1DRegression, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.res1 = ResidualUnit(64, 128, kernel_size=kernel_size, stride=3)
        self.res2 = ResidualUnit(128, 196, kernel_size=kernel_size, stride=3)
        self.res3 = ResidualUnit(196, 256, kernel_size=kernel_size, stride=2)
        self.res4 = ResidualUnit(256, 320, kernel_size=kernel_size, stride=2)

        self.flatten_dim = self._get_flatten_dim(input_length)
        self.fc = nn.Linear(self.flatten_dim, 1)

    def _get_flatten_dim(self, input_length):
        with torch.no_grad():
            x = torch.zeros(1, 1, input_length)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.res1(x)
            x = self.res2(x)
            x = self.res3(x)
            x = self.res4(x)
            flatten_dim = x.view(1, -1).size(1)
        return flatten_dim

    def forward(self, x):
        # x shape: [batch_size, 1, input_length]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def get_latent(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        latent = x.view(x.size(0), -1)
        
        return latent

def train_age_resnet(device, X, Y, lr=0.001, batch_size=256, num_epoch=16, end_factor=0.1, use_tqdm=True, input_length=100):
    model = ResNet1DRegression(input_length=input_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=end_factor, total_iters=(num_epoch * len(X)) // batch_size
    )
    criterion = nn.MSELoss()  # MSE loss for regression
    
    epoch_iterator = tqdm(range(num_epoch), desc="Training Epochs") if use_tqdm else range(num_epoch)
    for epoch in epoch_iterator:
        # Shuffle data at each epoch
        permutation = np.random.permutation(len(X))
        X = X[permutation]
        Y = Y[permutation]
        
        for batch_idx in range(0, len(X), batch_size):
            batch_data = X[batch_idx:batch_idx+batch_size]
            batch_target = Y[batch_idx:batch_idx+batch_size]
            
            # Prepare tensors
            batch_data = torch.tensor(batch_data, dtype=torch.float32).unsqueeze(1).to(device)
            batch_target = torch.tensor(batch_target, dtype=torch.float32).view(-1, 1).to(device)
            
            optimizer.zero_grad()
            pred = model(batch_data)
            loss = criterion(pred, batch_target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_iterator.set_description(f"Epoch {epoch+1}, loss: {loss.item():.5f}")
    
    return model

# Testing function for the ResNet regression model
def test_age_resnet(model, device, X):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(len(X)):
            data = torch.tensor(X[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            output = model(data)
            predictions.append(output.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    return predictions

def run_dx_upload_minimal(filename):
    command = ["dx", "upload", filename]
    try:
        subprocess.run(command, capture_output=True)
    except FileNotFoundError:
         print("Error: 'dx' command not found. Make sure dx-toolkit is installed and in your PATH.")


kf = KFold(n_splits=5, shuffle=True, random_state=seed)
start_fold = 1 

print(f"\nStarting 5-Fold Cross-Validation from Fold {start_fold}...")

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
    fold = fold_idx + 1 

    if fold < start_fold:
        print(f"\n--- Manually skipping Fold {fold} ---")
        continue

    print(f"\n=== Processing Fold {fold} ===")
    X_tr, X_te = X[train_idx], X[test_idx]
    Y_tr, Y_te = Y[train_idx], Y[test_idx]
    print(f"Fold {fold}: Train size={len(X_tr)}, Test size={len(X_te)}")

    model = train_age_resnet(
        device, X_tr, Y_tr,
        lr=0.0001,
        batch_size=1024,
        num_epoch=40, 
        end_factor=0.1,
        use_tqdm=True,
        input_length=100 
    )

    print(f"Fold {fold}: Testing model...")
    preds = test_age_resnet(model, device, X_te)
    preds = preds.flatten() 
    Y_te = Y_te.flatten()   

    valid_indices = ~np.isnan(preds) & ~np.isnan(Y_te) & ~np.isinf(preds) & ~np.isinf(Y_te)
    if np.sum(valid_indices) < 2:
        mse, mae, spearman_corr, spearman_p = np.nan, np.nan, np.nan, np.nan
        print(f"Warning: Not enough valid data points ({np.sum(valid_indices)}) for metrics in Fold {fold}.")
    else:
        mse = mean_squared_error(Y_te[valid_indices], preds[valid_indices])
        mae = mean_absolute_error(Y_te[valid_indices], preds[valid_indices])
        spearman_corr, spearman_p = spearmanr(preds[valid_indices], Y_te[valid_indices])

    print(f"Fold {fold} — MSE: {mse:.5f}, MAE: {mae:.5f}, "
          f"Spearman ρ: {spearman_corr:.5f} (p={spearman_p:.2e})")

    results = pd.DataFrame({
        "squared_error":  [mse],
        "absolute_error": [mae],
        "spearman_corr":  [spearman_corr],
        "spearman_p":     [spearman_p]
    })
    results_filename = f"resnet_lr0001_model_state_fold_{fold}_results.csv" 
    results.to_csv(results_filename, index=False)
    print(f"Saved results to {results_filename}")

    
    model_filename = f"resnet_lr0001_model_state_fold{fold}.pth"
    torch.save(
        model.state_dict(),
        model_filename
    )
    print(f"Saved model state to {model_filename}")


print("\nCross-validation finished.")