"""
QUANTITATIVE FIXED INCOME: END-TO-END CIR / CIR++ CALIBRATION PIPELINE
"""

import os
from typing import Dict, List, Union
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns



# PART 1: DATA ENGINEERING AND PREPROCESSING


print("\n[INFO] Loading and preprocessing datasets...")
train = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')

# Clean headers by trimming trailing/leading whitespaces
train.columns = train.columns.str.strip()
test.columns = test.columns.str.strip()

# Handle missing values via forward fill
train = train.ffill()
test = test.ffill()

# Combine data rows chronologically to evaluate continuous rolling lookback windows
full_data = pd.concat([train, test], axis=0).reset_index(drop=True)
test_start_idx = len(train)

# Calibration and tracking hyperparameters
w_size: int = 60              # 60-day historical calibration lookback window
decay_beta: float = 0.94      # RiskMetrics lambda decay factor for time-varying volatility
cpp_momentum: float = 0.98    # Autoregressive memory coefficient for CIR++ shift layer

# Define the cross-sectional term structure target maturities
tenors_map: Dict[str, float] = {
    'ZC050YR': 0.5, 
    'ZC075YR': 0.75, 
    'ZC100YR': 1.0, 
    'ZC200YR': 2.0
}

#Pre-compute normalized EWMA weights (weights sum to 1)
ewma_weights = np.array([decay_beta**(w_size - 1 - t) for t in range(w_size)])
ewma_weights /= np.sum(ewma_weights)



# PART 2: BASE CIR MODEL IMPLEMENTATION & CALIBRATION

def cir_yield(r: Union[float, np.ndarray], tau: float, k: float, th: float, sig: float) -> Union[float, np.ndarray]:
    """Computes exact closed-form zero-coupon yields for a single-factor CIR model."""
    
    # Floor inputs to prevent divide-by-zero or math errors
    k = max(k, 1e-6)
    th = max(th, 1e-6)
    sig = max(sig, 1e-6)
    
    # CIR analytical components
    h = np.sqrt(k**2 + 2 * sig**2)
    exp_h = np.exp(h * tau)
    den = 2 * h + (k + h) * (exp_h - 1)
    
    # Calculate analytical affine pricing multipliers A(tau) and B(tau)
    A_num = 2 * h * np.exp((k + h) * tau / 2)
    A = (A_num / den)**(2 * k * th / sig**2)
    B = 2 * (exp_h - 1) / den
    
    # Project yield while shielding from log-of-zero mathematical errors
    return (B * r - np.log(np.maximum(A, 1e-12))) / tau


print("[INFO] Running unified rolling out-of-sample calibration loop...")

# Storage structures for historical logging and parameter tracking
all_actuals = []
all_base_preds = []
all_cpp_preds = []

kappa_history = []
theta_history = []
sigma_history = []

last_step_residuals = {col: 0.0 for col in tenors_map.keys()}
current_params = [0.15, 0.03, 0.05]     # Initial parameter guess [kappa, theta, sigma]

# Out-of-sample rolling calibration loop
for i in range(test_start_idx, len(full_data)):
    window = full_data.iloc[i - w_size:i]
    r_short_hist = window['ZC025YR'].values
    
    # Loss function: EWMA-weighted SSE across cross-sectional tenors
    def objective(p):
        k, th, sig = p
        if k <= 0 or th <= 0 or sig <= 0: return 1e10
        err = 0.0
        for col, tau in tenors_map.items():
            preds = cir_yield(r_short_hist, tau, k, th, sig)
            err += np.sum(ewma_weights * ((window[col].values - preds)**2))
        return float(err)

    # Minimize cross-sectional residuals using the bounded L-BFGS-B method
    res = minimize(objective, current_params, bounds=[(0.001, 2.0), (0.001, 0.20), (0.001, 0.20)], method='L-BFGS-B')
    if res.success:
        current_params = list(res.x)
        
    k_opt, th_opt, sig_opt = current_params
    kappa_history.append(k_opt)
    theta_history.append(th_opt)
    sigma_history.append(sig_opt)
    
    

    k_opt, th_opt, sig_opt = current_params
    kappa_history.append(k_opt)
    theta_history.append(th_opt)
    sigma_history.append(sig_opt)
    
    # PART 3: THE PREDICTION CHALLENGE: YIELD CURVE CONSTRUCTION
    # Generate predictions for the current out-of-sample day
    current_row = full_data.iloc[i]
    r_today = current_row['ZC025YR']
    
    day_actuals = []
    day_base_preds = []
    day_cpp_preds = []
    new_residuals = {}
    
    for col, tau in tenors_map.items():
        actual_yield = current_row[col]
        
        # Construct the baseline theoretical CIR model forecast
        base_pred = cir_yield(r_today, tau, k_opt, th_opt, sig_opt)

        # PART 4: MODEL IMPROVEMENT & EXTENSIONS
        # Apply error-correction using the saved momentum parameter (0.98 shift)
        cpp_pred = base_pred + (cpp_momentum * last_step_residuals[col])
        
        day_actuals.append(actual_yield)
        day_base_preds.append(base_pred)
        day_cpp_preds.append(cpp_pred)
        
        # Save baseline model error to pass forward to the next day
        new_residuals[col] = actual_yield - base_pred
        
    all_actuals.append(day_actuals)
    all_base_preds.append(day_base_preds)
    all_cpp_preds.append(day_cpp_preds)
    
    last_step_residuals = new_residuals

# Flatten arrays for global evaluation metrics
y_true = np.array(all_actuals).flatten()
y_base = np.array(all_base_preds).flatten()
y_cpp = np.array(all_cpp_preds).flatten()

r2_base_60 = float(r2_score(y_true, y_base))
r2_cpp_60 = float(r2_score(y_true, y_cpp))


print("              FINAL MODEL PERFORMANCE              ")

print(f"Global Baseline CIR Model R2 Score : {r2_base_60:.6f}")
print(f"Global Advanced CIR++ Model R2 Score: {r2_cpp_60:.6f}")


# Plotting Section
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
last_actual = all_actuals[-1]
last_base = all_base_preds[-1]
last_cpp = all_cpp_preds[-1]
tenor_years = list(tenors_map.values())

# Chart A: Cross-Sectional Yield Curve Snapshot
axes[0].plot(tenor_years, last_actual, 'ko-', label='Market Actuals', linewidth=2.5)
axes[0].plot(tenor_years, last_base, 'r--', label=f'CIR Baseline (R²: {r2_base_60:.4f})', alpha=0.9)
axes[0].plot(tenor_years, last_cpp, 'b-.', label=f'CIR++ Model (R²: {r2_cpp_60:.4f})', alpha=0.9)
axes[0].set_title("Yield Curve Alignment (Final Test Snapshot)", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Maturity Tenor (Years)")
axes[0].set_ylabel("Yield (%)")
axes[0].legend(frameon=True, loc='best')

# Chart B: Out-of-Sample Residual Error Probability Densities
base_residuals = y_true - y_base
cpp_residuals = y_true - y_cpp
sns.kdeplot(base_residuals, ax=axes[1], color='red', fill=True, label='Baseline Residuals', alpha=0.3)
sns.kdeplot(cpp_residuals, ax=axes[1], color='blue', fill=True, label='CIR++ Residuals', alpha=0.3)
axes[1].axvline(0, color='black', linestyle=':', alpha=0.7)
axes[1].set_title("Error Density Profile (60-Day EWMA)", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Prediction Error (Actual - Predicted)")
axes[1].set_ylabel("Density")
axes[1].legend(frameon=True)
plt.tight_layout()
plt.show()


# PART 5: CRITICAL ANALYSIS

# 1. ANALYSIS OF THE SINGLE-FACTOR COMPRESSION CONSTRAINT (Maturity Breakdown)

print("       CRITICAL ANALYSIS: TENOR BREAKDOWN         ")

actuals_df = pd.DataFrame(all_actuals, columns=list(tenors_map.keys()))
base_df = pd.DataFrame(all_base_preds, columns=list(tenors_map.keys()))
cpp_df = pd.DataFrame(all_cpp_preds, columns=list(tenors_map.keys()))

for col in tenors_map.keys():
    r2_b = r2_score(actuals_df[col], base_df[col])
    r2_c = r2_score(actuals_df[col], cpp_df[col])
    print(f"Tenor {col} ({tenors_map[col]:.2f}Y) -> Baseline R²: {r2_b:8.5f} | CIR++ R²: {r2_c:8.5f}")

# 2. ANALYSIS OF FELLER CONDITION REGIME BREAKDOWNS (Stochastic Integrity Check)
# Mathematically, the Feller condition requires 2 * kappa * theta >= sigma^2.
# If violated, the probability density at zero is positive, allowing anomalous drops.
k_arr = np.array(kappa_history)
th_arr = np.array(theta_history)
sig_arr = np.array(sigma_history)

feller_metric = (2 * k_arr * th_arr) - (sig_arr ** 2)
violations = np.sum(feller_metric < 0)
violation_rate = (violations / len(feller_metric)) * 100


print("     CRITICAL ANALYSIS: FELLER REGIME SANITY       ")

print(f"Total Out-of-Sample Tracking Horizons : {len(feller_metric)} Days")
print(f"Empirical Feller Violations Logged   : {violations} Days ({violation_rate:.2f}%)")

if violation_rate > 0:
    print("\n[DIAGNOSTIC CONCLUSION]:")
    print(f"The structural breakdown rate of {violation_rate:.2f}% confirms that single-factor")
    print("equilibrium restrictions collapse under time-varying historical volatility.")
    print("This explicitly validates why the deterministic error-correction layer (CIR++)")
    print("is mathematically necessary to adapt to actual market shapes.")

    
    
