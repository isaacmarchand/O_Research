#!/usr/bin/env python3
import argparse
import os
import pickle
import numpy as np
from factorRepresentation_dayWeighted_penalizedEigenBasis import FPCA_penalized

def main():
    parser = argparse.ArgumentParser(description="Estimate a specific FPC, loading previous components if needed, and saving the scores and eigenfunctions.")
    
    # Paths & Component Index
    parser.add_argument("--data_path", type=str, default="data/DJX_data/DJX_lists_traded.pkl",
                        help="Path to the pickle file containing the IVS lists.")
    parser.add_argument("--fpc_index", type=int, default=0,
                        help="Index of the FPC to estimate (0 for the first FPC, 1 for second, etc.).")
    parser.add_argument("--previous_path", type=str, default=None,
                        help="Path to the pickle file containing previously saved scoreMat and BList (required if fpc_index > 0).")
    parser.add_argument("--output_path", type=str, default="data/DJX_data/estimated_FPCA.pkl",
                        help="Path to save the resulting scoreMat and BList.")
    parser.add_argument("--n_days", type=int, default=None,
                        help="Limit the dataset to the first N days.")
    
    # Model basis parameters
    parser.add_argument("--nb_spline_moneyness", type=int, default=30,
                        help="Number of spline basis functions for moneyness.")
    parser.add_argument("--nb_spline_tau", type=int, default=36,
                        help="Number of spline basis functions for maturity (tau).")
    parser.add_argument("--order_moneyness", type=int, default=4,
                        help="Spline order for moneyness.")
    parser.add_argument("--order_tau", type=int, default=4,
                        help="Spline order for tau.")
    
    # Optimization/Fitting parameters
    parser.add_argument("--maxit", type=int, default=30,
                        help="Maximum number of iterations for the fitting algorithm.")
    parser.add_argument("--threshold", type=float, default=1e-4,
                        help="Convergence threshold for the fitting algorithm.")
    
    # Hyperparameters for fitting (if not running CV)
    parser.add_argument("--omega_m", type=float, default=0.05,
                        help="Smoothness penalty for moneyness.")
    parser.add_argument("--omega_m2", type=float, default=0.0,
                        help="Second smoothness penalty for moneyness.")
    parser.add_argument("--omega_t", type=float, default=0.05,
                        help="Smoothness penalty for tau.")
    parser.add_argument("--d_m", type=int, default=2,
                        help="Order of difference penalty for moneyness.")
    parser.add_argument("--d_m2", type=int, default=3,
                        help="Order of second difference penalty for moneyness.")
    parser.add_argument("--d_t", type=int, default=2,
                        help="Order of difference penalty for tau.")
    
    # CV Option & parameters
    parser.add_argument("--run_cv", action="store_true",
                        help="Run cross-validation grid search to find optimal hyperparameters before fitting.")
    parser.add_argument("--omega_m_grid", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3],
                        help="Grid of candidate values for omega_m.")
    parser.add_argument("--omega_m2_grid", type=float, nargs="+", default=[0.0],
                        help="Grid of candidate values for omega_m2.")
    parser.add_argument("--omega_t_grid", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3],
                        help="Grid of candidate values for omega_t.")
    parser.add_argument("--cv_type", type=str, choices=["rolling", "day", "observation"], default="rolling",
                        help="Cross-validation splitting strategy.")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="Number of splits/folds for cross-validation.")
    parser.add_argument("--train_window_size", type=int, default=None,
                        help="CV size of training window.")
    parser.add_argument("--test_window_size", type=int, default=None,
                        help="CV size of validation window.")
    parser.add_argument("--theta_cal", type=float, default=0.0,
                        help="Weight of calendar spread arbitrage penalty in CV.")
    parser.add_argument("--theta_but", type=float, default=0.0,
                        help="Weight of butterfly spread arbitrage penalty in CV.")
    parser.add_argument("--friction_tol", type=float, default=1e-4,
                        help="Friction tolerance threshold for arbitrage check in CV.")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for fold splits.")

    args = parser.parse_args()
    
    print("="*60)
    print("FPCA Component Estimator and Pickler")
    print("="*60)

    # 1. Validation and Loading of previous components
    previous_BList = None
    previous_scoreMat = None
    if args.fpc_index > 0:
        if args.previous_path is None:
            raise ValueError(f"For fpc_index={args.fpc_index} (> 0), a path to the previously saved components must be provided via --previous_path.")
        if not os.path.exists(args.previous_path):
            raise FileNotFoundError(f"Previous components file not found at: {args.previous_path}")
        print(f"Loading previous components from {args.previous_path}...")
        with open(args.previous_path, "rb") as f:
            previous_scoreMat = pickle.load(f)
            previous_BList = pickle.load(f)
        
        # Verify sizes
        if not isinstance(previous_BList, list) or len(previous_BList) < args.fpc_index:
            raise ValueError(f"The loaded BList contains {len(previous_BList) if isinstance(previous_BList, list) else 'non-list'} FPCs, but fpc_index={args.fpc_index} requires at least {args.fpc_index} components.")
        if previous_scoreMat.shape[1] < args.fpc_index:
            raise ValueError(f"The loaded scoreMat has {previous_scoreMat.shape[1]} columns, but fpc_index={args.fpc_index} requires at least {args.fpc_index} columns.")
        
        # Slicing the components to index of the FPC
        previous_BList = previous_BList[:args.fpc_index]
        previous_scoreMat = previous_scoreMat[:, :args.fpc_index]
        
        if args.n_days is not None:
            previous_scoreMat = previous_scoreMat[:args.n_days, :]
            
        print(f"Loaded {len(previous_BList)} previous components successfully.")

    # 2. Load and process dataset
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found at: {args.data_path}")
        
    print(f"Loading dataset from {args.data_path}...")
    with open(args.data_path, "rb") as f:
        uniqueDates = pickle.load(f)
        tau = pickle.load(f)
        moneyness = pickle.load(f)
        iv = pickle.load(f)
        S = pickle.load(f)
        rfRate = pickle.load(f)
        dividendRate = pickle.load(f)
        
    if args.n_days is not None:
        print(f"Limiting dataset to the first {args.n_days} days...")
        tau = tau[:args.n_days]
        moneyness = moneyness[:args.n_days]
        iv = iv[:args.n_days]
        S = S[:args.n_days] if S is not None else None
        rfRate = rfRate[:args.n_days] if rfRate is not None else None
        dividendRate = dividendRate[:args.n_days] if dividendRate is not None else None
        
    logMoneyness = [np.log(m) for m in moneyness]
    
    # Calculate limits dynamically to ensure splines are properly bounded
    maxLogMoney = np.ceil(np.max([np.max(m) for m in logMoneyness])*100)/100
    minLogMoney = np.floor(np.min([np.min(m) for m in logMoneyness])*100)/100
    maxTau = np.ceil(np.max([np.max(t) for t in tau])*10)/10
    minTau = np.floor(np.min([np.min(t) for t in tau])*10)/10
    
    print("Initializing FPCA model instance...")
    fpca = FPCA_penalized(
        logMoneyness, tau, iv,
        nb_spline_moneyness=args.nb_spline_moneyness,
        nb_spline_tau=args.nb_spline_tau,
        order_moneyness=args.order_moneyness,
        order_tau=args.order_tau,
        S=S, r=rfRate, q=dividendRate,
        range_moneyness=[minLogMoney, maxLogMoney],
        range_tau=[minTau, maxTau]
    )

    # 3. Cross-Validation Option
    omega_m = args.omega_m
    omega_m2 = args.omega_m2
    omega_t = args.omega_t
    
    if args.run_cv:
        print(f"\nRunning Cross Validation to find optimal parameters for FPC {args.fpc_index + 1}...")
        omega_m, omega_m2, omega_t, cv_results = fpca.cross_validate_penalties(
            omega_m_grid=args.omega_m_grid,
            omega_m2_grid=args.omega_m2_grid,
            omega_t_grid=args.omega_t_grid,
            n_splits=args.n_splits,
            cv_type=args.cv_type,
            fpc_index=args.fpc_index,
            previous_BList=previous_BList,
            threshold=args.threshold,
            maxit=args.maxit,
            d_m=args.d_m,
            d_m2=args.d_m2,
            d_t=args.d_t,
            random_state=args.random_state,
            train_window_size=args.train_window_size,
            test_window_size=args.test_window_size,
            theta_cal=args.theta_cal,
            theta_but=args.theta_but,
            friction_tol=args.friction_tol
        )
        print("Cross-Validation completed.")
        print(f"Optimal parameters selected: omega_m={omega_m}, omega_m2={omega_m2}, omega_t={omega_t}")

    # Set prior state of FPCA instance with loaded components before fitting the target FPC
    if args.fpc_index > 0:
        fpca.scoreMat = previous_scoreMat
        fpca.BList = list(previous_BList)

    # 4. Perform FPC estimation
    print(f"\nEstimating FPC {args.fpc_index + 1} using: omega_m={omega_m}, omega_m2={omega_m2}, omega_t={omega_t}...")
    if args.fpc_index == 0:
        alpha, B = fpca.first_FPC_fit(
            threshold=args.threshold,
            maxit=args.maxit,
            omega_m=omega_m,
            omega_m2=omega_m2,
            omega_t=omega_t,
            d_m=args.d_m,
            d_m2=args.d_m2,
            d_t=args.d_t
        )
    else:
        alpha, B = fpca.subsequent_FPC_fit(
            threshold=args.threshold,
            maxit=args.maxit,
            omega_m=omega_m,
            omega_m2=omega_m2,
            omega_t=omega_t,
            d_m=args.d_m,
            d_m2=args.d_m2,
            d_t=args.d_t
        )
        
    print(f"FPC {args.fpc_index + 1} estimation complete.")
    
    # 5. Save components using pickle in the requested format (lines 1632-1634)
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    print(f"Saving score matrix and BList to: {args.output_path}...")
    with open(args.output_path, "wb") as f:
        pickle.dump(fpca.scoreMat, f)
        pickle.dump(fpca.BList, f)
        
    print("Estimation script completed successfully!")

if __name__ == '__main__':
    main()
