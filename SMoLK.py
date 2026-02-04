import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import periodogram
from scipy.stats import spearmanr
import subprocess
import numpy as np
import random
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error


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

class LearnedFiltersAge(nn.Module):
    def __init__(self, num_kernels=24):
        super(LearnedFiltersAge, self).__init__()
        # 100 point signals so chose 25, 15, 10 instead of 192, 96, 64
        self.conv1 = nn.Conv1d(1, num_kernels, kernel_size=25, stride=1, bias=True)
        self.conv2 = nn.Conv1d(1, num_kernels, kernel_size=15, stride=1, bias=True)
        self.conv3 = nn.Conv1d(1, num_kernels, kernel_size=10, stride=1, bias=True)
        
        self.linear = nn.Linear(num_kernels*3 + 51, 1)  # 51 instead of 321 for size of power spectrum
    
    def forward(self, x, powerspectrum):
        c1 = F.leaky_relu(self.conv1(x)).mean(dim=-1) # shape of x is [B, 1, 100]
        c2 = F.leaky_relu(self.conv2(x)).mean(dim=-1)
        c3 = F.leaky_relu(self.conv3(x)).mean(dim=-1)
        
        aggregate = torch.cat([c1, c2, c3, powerspectrum], dim=1) # shape of powerspectrum is [B, 51]
        aggregate = self.linear(aggregate)
        
        return aggregate
    
    def get_latent(self, x, powerspectrum):
        c1 = F.leaky_relu(self.conv1(x)).mean(dim=-1)
        c2 = F.leaky_relu(self.conv2(x)).mean(dim=-1)
        c3 = F.leaky_relu(self.conv3(x)).mean(dim=-1)
        
        # Concatenate to form latent representation
        latent = torch.cat([c1, c2, c3, powerspectrum], dim=1)
        return latent

def train_age(device, X, Y, num_kernels=24, lr=0.001, batch_size=256, num_epoch=16, end_factor=0.1, use_tqdm=True):
    # compute power spectra for X
    PowerSpectra = []
    for i in tqdm(range(0, len(X)), desc="Computing Power Spectra"):
        PowerSpectra.append(periodogram(X[i], fs=100)[1])
    PowerSpectra = np.float32(PowerSpectra)
    
    model = LearnedFiltersAge(num_kernels=num_kernels).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=end_factor, total_iters=(num_epoch * len(X)) // batch_size
    )
    criterion = nn.MSELoss() # MSE Loss instead of CrossEntropy
    
    epoch_losses = [] 
    epoch_iterator = tqdm(range(num_epoch), desc="Training Epochs") if use_tqdm else range(num_epoch)
    for epoch in epoch_iterator:
        epoch_loss = 0.0 
        n_batches  = 0
        # Shuffle data at each epoch
        permutation = np.random.permutation(len(X))
        X = X[permutation]
        Y = Y[permutation]
        PowerSpectra = PowerSpectra[permutation]
        
        for batch_idx in range(0, len(X), batch_size):
            batch_data = X[batch_idx:batch_idx+batch_size]
            batch_power = PowerSpectra[batch_idx:batch_idx+batch_size]
            batch_target = Y[batch_idx:batch_idx+batch_size]
            
            # Prepare tensors
            batch_data = torch.tensor(batch_data, dtype=torch.float32).unsqueeze(1).to(device) 
            batch_power = torch.tensor(batch_power, dtype=torch.float32).to(device)
            batch_target = torch.tensor(batch_target, dtype=torch.float32).view(-1, 1).to(device)
            
            optimizer.zero_grad()
            pred = model(batch_data, batch_power)
            loss = criterion(pred, batch_target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item() 
            n_batches  += 1
            
            epoch_iterator.set_description(f"Epoch {epoch+1}, loss: {loss.item():.5f}")
        avg_loss = epoch_loss / n_batches
        epoch_losses.append(avg_loss)
        if use_tqdm:
            epoch_iterator.set_postfix(avg_loss=f"{avg_loss:.5f}")
    
    return model, epoch_losses

def test_age(model, device, X):
    # compute power spectra for X
    PowerSpectra = []
    for i in range(len(X)):
        PowerSpectra.append(periodogram(X[i], fs=100)[1])
    PowerSpectra = np.stack(PowerSpectra)
    
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(X)):
            data = torch.tensor(X[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) 
            powerspectrum = torch.tensor(PowerSpectra[i], dtype=torch.float32).unsqueeze(0).to(device) 
            output = model(data, powerspectrum)
            predictions.append(output.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    return predictions

def run_dx_upload(filename):
    command = ["dx", "upload", filename]
    try:
        subprocess.run(command, capture_output=True)
    except FileNotFoundError:
         print("Error")

            
# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=seed)
start_fold = 1
for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
    fold = fold_idx + 1
    if fold < start_fold:
        print(f"\n--- Manually skipping Fold {fold} ---")
        continue 
    
    print(f"\n=== Fold {fold} ===")
    X_tr, X_te = X[train_idx], X[test_idx]
    Y_tr, Y_te = Y[train_idx], Y[test_idx]
    
    model, epoch_losses = train_age(
        device, X_tr, Y_tr,
        num_kernels=384, lr=0.0001,
        batch_size=1024, num_epoch=200,
        end_factor=0.1, use_tqdm=True
    )

    # save per-epoch loss for this fold
    loss_df = pd.DataFrame({
        "epoch": np.arange(1, len(epoch_losses) + 1),
        "train_loss": epoch_losses
    })
    loss_file = f"smolk_model_lr0001_384_200_fold_{fold}_epoch_loss.csv"
    loss_df.to_csv(loss_file, index=False)
    run_dx_upload(loss_file)
    
    preds = test_age(model, device, X_te)
    preds = preds.flatten()
    Y_te = Y_te.flatten()
    mse = mean_squared_error(Y_te, preds)
    mae = mean_absolute_error(Y_te, preds)
    spearman_corr, spearman_p = spearmanr(preds, Y_te)
    print(f"Fold {fold} — MSE: {mse:.5f}, MAE: {mae:.5f}, "
          f"Spearman ρ: {spearman_corr:.5f} (p={spearman_p:.2e})")
    
    results = pd.DataFrame({
        "squared_error":  [mse],
        "absolute_error": [mae],
        "spearman_corr":  [spearman_corr],
        "spearman_p":     [spearman_p]
    })
    results_filename = f"smolk_model_lr0001_384_200_fold_{fold}_results.csv"
    results.to_csv(results_filename, index=False)
    run_dx_upload(results_filename) 
    
    model_filename = f"smolk_model_lr0001_384_200_fold{fold}.pth" 
    torch.save(
        model.state_dict(),
        model_filename
    )
    run_dx_upload(model_filename)