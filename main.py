import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import seed_everything, DEVICE
from data_utils import *  
from methods import * 
# (run_split, run_plcp, run_gaussian_scoring, run_cqr, run_rcp, run_rcp_multi_head, run_rcp_density_improved)

# Suppress warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")

def rcp_protocol_split(X, Y, cal_size=0.2, seed=42):
    """Splits data into Training, Calibration, and Test sets."""
    # Convert cal_size fraction to int if needed, here simplified
    n_cal = int(len(X) * cal_size)    
    X_rem, X_cal, Y_rem, Y_cal = train_test_split(X, Y, test_size=n_cal, random_state=seed)
    X_tr, X_te, Y_tr, Y_te = train_test_split(X_rem, Y_rem, test_size=0.3, random_state=seed)
    return X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te

def run_benchmark_suite():
    seed_everything(42)
    print(f"Using Device: {DEVICE}")

    # Dataset Registry
    dataset_loaders = {    
        "naval": load_naval,    
        "gas_turbine":load_gas_turbine,
        "diamond": load_diamonds,            
        "superconduct": load_superconductivity,
        "bike": load_bike,
        "SGEMM": load_sgemm_product,                               
        "Transcoding": load_transcoding, 
        "WEC": load_wec,                  
    }
    
    alpha = 0.1
    n_seeds = 20
    
    # Method Registry
    methods = [
        ('Split', run_split),
        ('PLCP-Pin-K20', lambda *a: run_plcp(*a, n_groups=20, score_type='pinball')),
        ('PLCP-Pin-K50', lambda *a: run_plcp(*a, n_groups=50, score_type='pinball')),
        ('Gaussian-Scoring', run_gaussian_scoring),
        ('CQR-Pinball', lambda *a: run_cqr(*a, 'pinball')),
        ('CQR-ALD', lambda *a: run_cqr(*a, 'ald')),        
        ('RCP-Pinball', run_rcp),        
        ('RCP-ALD', lambda *a: run_rcp(*a, 'ald')),
        ('RCP-MultiHead', run_rcp_multi_head),                
        
        # Colorful Pinball Variants (CPCP)
        ('CPCP-Split-0.02', lambda *a, **k: run_rcp_density_improved(*a, epsilon=0.02, mode='vanilla', **k)),                
        ('CPCP-Clip-0.02', lambda *a, **k: run_rcp_density_improved(*a, epsilon=0.02, mode='clip', clip_max=5.0, **k)),            
        ('CPCP-Mix-0.02', lambda *a, **k: run_rcp_density_improved(*a, epsilon=0.02, mode='mix', mix_ratio=0.5, **k)),
        ('CPCP-Clip+Mix', lambda *a, **k: run_rcp_density_improved(*a, epsilon=0.02, mode='clip', clip_max=5.0, mix_ratio=0.5, **k)),
    ]
    
    for ds_name, loader in dataset_loaders.items():
        print(f"\n>>>>>> Running {ds_name} <<<<<<")
        try: 
            X, Y = loader()
            print(f"Data Shape: {X.shape}, {Y.shape}")
        except Exception as e: 
            print(f"Error loading {ds_name}: {e}")
            continue
            
        results = {m[0]: [] for m in methods}
        
        for seed in range(n_seeds):
            print(f"Seed {seed}...", end="", flush=True)
            X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te = rcp_protocol_split(X, Y, seed=42+seed)
            
            for name, func in methods:
                try:
                    # Pass extra args for CPCP methods
                    if 'CPCP' in name or 'RCP-Density' in name:
                        res = func(X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te, alpha, dataset_name=ds_name, seed=seed)
                    else:
                        res = func(X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te, alpha)
                    results[name].append(res)
                except Exception as e:
                    print(f" Err({name}:{e})", end="")
            print(" Done")
            
        # Summary & Save
        summary_rows = []
        for name, mets in results.items():
            if not mets: continue
            row = {'Method': name}
            for k in ['Cov', 'Size', 'WSC', 'CCE']:
                vals = [m[k] for m in mets]
                row[k] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
            summary_rows.append(row)
        
        print("\nSummary:")
        df_res = pd.DataFrame(summary_rows)
        print(df_res)
        
        if not os.path.exists("./results"): os.makedirs("./results")
        df_res.to_csv(f"./results/{ds_name}_results.csv")

if __name__ == "__main__":
    run_benchmark_suite()