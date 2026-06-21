#!/usr/bin/env python3
import argparse
import os
import pickle
import json
import numpy as np
from factorRepresentation_dayWeighted_penalizedEigenBasis import FPCA_penalized

def main():
    parser = argparse.ArgumentParser(description="Tune FPCA smoothness penalties using Cross-Validation.")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="data/DJX_data/DJX_lists.pkl",
                        help="Path to the pickle file containing the IVS lists.")
    parser.add_argument("--n_days", type=int, default=None,
                        help="Limit the dataset to the first N days (useful for faster tuning).")
    
    # CV Hyperparameters
    parser.add_argument("--omega_m_grid", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3],
                        help="Grid of candidate values for omega_m (moneyness penalty).")
    parser.add_argument("--omega_t_grid", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3],
                        help="Grid of candidate values for omega_t (tau/maturity penalty).")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="Number of cross-validation splits/folds.")
    parser.add_argument("--cv_type", type=str, choices=["day", "observation"], default="day",
                        help="Cross-validation splitting strategy ('day' or 'observation').")
    
    # Component tuning arguments
    parser.add_argument("--fpc_index", type=int, default=0,
                        help="Index of the FPC to tune (0 for the first FPC, >0 for subsequent FPCs).")
    parser.add_argument("--previous_blist_path", type=str, default=None,
                        help="Path to a pickle file containing previously fitted B matrices (required if fpc_index > 0).")
    
    # Optimization parameters
    parser.add_argument("--maxit", type=int, default=10,
                        help="Maximum number of iterations for the FPC fitting algorithm.")
    parser.add_argument("--threshold", type=float, default=1e-4,
                        help="Convergence threshold for the FPC fitting algorithm.")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for fold splitting reproducibility.")
    
    # Output arguments
    parser.add_argument("--output_prefix", type=str, default="cv_tuning",
                        help="Prefix for the saved results and parameters JSON files.")
    
    args = parser.parse_args()
    
    print("="*60)
    print("FPCA Smoothness Penalty Cross-Validation Tuner")
    print("="*60)
    
    # 1. Load Data
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found at: {args.data_path}")
        
    print(f"Loading data from {args.data_path}...")
    with open(args.data_path, "rb") as f:
        uniqueDates = pickle.load(f)
        tau = pickle.load(f)
        moneyness = pickle.load(f)
        iv = pickle.load(f)
        S = pickle.load(f)
        rfRate = pickle.load(f)
        dividendRate = pickle.load(f)
        
    # Apply day limit if requested
    if args.n_days is not None:
        print(f"Limiting dataset to the first {args.n_days} days...")
        tau = tau[:args.n_days]
        moneyness = moneyness[:args.n_days]
        iv = iv[:args.n_days]
        S = S[:args.n_days] if S is not None else None
        rfRate = rfRate[:args.n_days] if rfRate is not None else None
        dividendRate = dividendRate[:args.n_days] if dividendRate is not None else None
        
    logMoneyness = [np.log(m) for m in moneyness]
    
    maxLogMoney = np.ceil(np.max([np.max(m) for m in logMoneyness])*100)/100
    minLogMoney = np.floor(np.min([np.min(m) for m in logMoneyness])*100)/100
    maxTau = np.ceil(np.max([np.max(t) for t in tau])*10)/10
    minTau = np.floor(np.min([np.min(t) for t in tau])*10)/10
    
    # 2. Handle previous components
    previous_BList = None
    if args.fpc_index > 0:
        if args.previous_blist_path is None:
            raise ValueError("For subsequent FPCs (fpc_index > 0), you must provide a --previous_blist_path.")
        if not os.path.exists(args.previous_blist_path):
            raise FileNotFoundError(f"Previous BList pickle file not found at: {args.previous_blist_path}")
            
        print(f"Loading previous BList from {args.previous_blist_path}...")
        with open(args.previous_blist_path, "rb") as f:
            # We assume the pickle file contains BList or list of B matrices
            previous_BList = pickle.load(f)
            # If the pickle has scoreMat and BList together (like SPX_FPCA_ApproxPenal_logM_tau_iv.pkl), retrieve only the BList
            if not isinstance(previous_BList, list):
                previous_BList = pickle.load(f)
            
        # Slice to the correct length corresponding to fpc_index
        previous_BList = previous_BList[:args.fpc_index]
        print(f"Loaded {len(previous_BList)} previous FPC(s).")
        
    # 3. Instantiate model
    print("Initializing FPCA model instance...")
    fpca = FPCA_penalized(
        logMoneyness, tau, iv,
        nb_spline_moneyness=30, nb_spline_tau=36,
        order_moneyness=4, order_tau=4,
        S=S, r=rfRate, q=dividendRate,
        range_moneyness = [minLogMoney, maxLogMoney],
        range_tau = [minTau, maxTau]
    )
    
    # 4. Run Cross-Validation
    print(f"\nStarting Grid Search CV ({args.cv_type}-based split, {args.n_splits} splits):")
    print(f"  omega_m grid: {args.omega_m_grid}")
    print(f"  omega_t grid: {args.omega_t_grid}")
    print(f"  Tuning FPC index: {args.fpc_index}")
    
    best_m, best_t, results = fpca.cross_validate_penalties(
        omega_m_grid=args.omega_m_grid,
        omega_t_grid=args.omega_t_grid,
        n_splits=args.n_splits,
        cv_type=args.cv_type,
        fpc_index=args.fpc_index,
        previous_BList=previous_BList,
        threshold=args.threshold,
        maxit=args.maxit,
        random_state=args.random_state
    )
    
    print("\n" + "="*50)
    print("TUNING COMPLETED")
    print("="*50)
    print(f"Optimal omega_m (moneyness): {best_m}")
    print(f"Optimal omega_t (tau/maturity): {best_t}")
    
    # 5. Save results
    best_params_file = f"{args.output_prefix}_best_params.json"
    results_file = f"{args.output_prefix}_results.json"
    
    best_params_data = {
        "best_omega_m": best_m,
        "best_omega_t": best_t,
        "fpc_index": args.fpc_index,
        "cv_type": args.cv_type,
        "n_splits": args.n_splits
    }
    
    print(f"\nSaving best parameters to: {best_params_file}")
    with open(best_params_file, "w") as f:
        json.dump(best_params_data, f, indent=4)
        
    print(f"Saving detailed grid results to: {results_file}")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nDone! Best parameters and results successfully saved.")

if __name__ == "__main__":
    main()
