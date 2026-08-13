#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.special import roots_legendre
from sklearn.linear_model import LinearRegression

import pickle as pickle


#%%

class FPCA_penalized_totVar:
    def __init__(self, moneyness, tau, totVar, nb_spline_moneyness = 10, nb_spline_tau = 10, order_moneyness = 4, order_tau = 4, S=None, r=None, q=None, range_moneyness = [-.15, .15], range_tau = [0, 1], dailyWeights = None):
        self.S = S
        self.r = r
        self.q = q
        
        if dailyWeights is None:
            self.dailyWeights = np.repeat(1, len(totVar))
        elif len(dailyWeights) != len(totVar):
            raise ValueError("the length of the daily weight vector has to be the same as the number of days observed len(totVar)")
        else:
            self.dailyWeights = dailyWeights
        
        # Clean data once to avoid repeated computation in iterations
        self.cleaned_data = []
        for i in range(len(totVar)):
            m_f = np.asarray(moneyness[i]).flatten()
            t_f = np.asarray(tau[i]).flatten()
            totVar_f = np.asarray(totVar[i]).flatten()
            mask = ~np.isnan(totVar_f) & ~np.isnan(m_f) & ~np.isnan(t_f)
            self.cleaned_data.append((m_f[mask], t_f[mask], totVar_f[mask]))
        
        # Range of moneyness and time to expiry (tau) for which the model will be trained on
        self.range_moneyness = range_moneyness # range set for logMoneyness (can be changed depending on type of moneyness)
        self.range_tau = range_tau
        
        # Initiate B-splines tensor product that will be used to estimate eigenfunctions
        self.nb_spline_moneyness = nb_spline_moneyness
        self.nb_spline_tau = nb_spline_tau
        self.order_moneyness = order_moneyness      # order of splines
        self.order_tau = order_tau
        
        self.BSplines_2d(self.range_moneyness, self.range_tau, 
                         self.nb_spline_moneyness, self.nb_spline_tau,
                         self.order_moneyness, self.order_tau)
        
        # Set starting weight B
        self.B0 = np.zeros((self.nb_spline_moneyness, self.nb_spline_tau)) + 0.1
        
        # Create list of Bs containing all the matrix B fitted so far for the FPCs 
        self.BList = []
       
    def first_FPC_fit(self, threshold = 1e-4, maxit = 10, omega_m = 1.0, omega_m2 = 0.0, omega_t = 1.0, d_m = 2, d_m2 = 3, d_t = 2, monotone = True):
        """
        Estimate the first FPC and its scores.

        Parameters:
        - threshold: indicate what change in MSE do we consider as reaching convergence
        - maxit: maximum number of iterations
        - omega_m: penalty weight for the moneyness dimension
        - omega_m2: penalty weight for the second moneyness dimension penalty
        - omega_t: penalty weight for the tau dimension
        - d_m: order of the difference penalty for moneyness
        - d_m2: order of the second difference penalty for moneyness
        - d_t: order of the difference penalty for tau

        Returns:
        - scores: List of array of estimated scores for the first FPC
        - B: 2D array of dimension (nb_spline_moneyness X nb_spline_tau) specifying the first FPC
        """

        B = self.B0
        
        # Rescale the B s.t. 2-norm of eigen function is 1
        norm = self.compute_norm(B)
        B = B / norm
        
        old_mse = 1e10 # Large initial value
        old_B = B
        maxBChange = 1e10
        j = 0
        
        # Create score Matrix nb_days X 1 -> will add columns as nb of FPC increases
        self.scoreMat = np.zeros((len(self.cleaned_data), 1))
        
        # --- Precompute Penalty Matrix P ---
        S_m = self.nb_spline_moneyness
        S_tau = self.nb_spline_tau
        D_diff_m = np.diff(np.eye(S_m), n=d_m, axis=0)
        D_diff_m2 = np.diff(np.eye(S_m), n=d_m2, axis=0)
        D_diff_t = np.diff(np.eye(S_tau), n=d_t, axis=0)
        
        # Note: Row-major (C-style) Kronecker order used here
        P_m = np.kron(D_diff_m.T @ D_diff_m, np.diag(1/(fpca.t_t[4:]**2)))
        P_m2 = np.kron(D_diff_m2.T @ D_diff_m2, np.diag(1/(fpca.t_t[4:]**2)))
        P_t = np.kron(np.eye(S_m), D_diff_t.T @ D_diff_t)
        P = omega_m * P_m + omega_m2 * P_m2 + omega_t * P_t
        
        #check monotonicity condition
        identityWithZeros = np.eye(S_tau)
        identityWithZeros[0:3,:] = 0
        D_diff_t_monotone = np.diff(identityWithZeros, n=1, axis=0)
        np.fill_diagonal(D_diff_t_monotone,np.concatenate(([0,0],-(self.t_t[4:-3]/self.t_t[5:-2]))))
        P_1 = np.kron(np.eye(S_m), D_diff_t_monotone)
        
        omegaMonotone = 10**6
        delta = np.zeros((S_tau-1)*S_m) + .001
        
        while j < maxit:
            scores = []
            mse_list = []
            for i in range(len(self.cleaned_data)):
                m_clean, tau_clean, totVar_clean = self.cleaned_data[i]
                
                if len(totVar_clean) == 0:
                    scores.append(0.0)
                    continue

                # Build the design matrix X (features)
                # Evaluate every basis function on the discrete (moneyness, tau) points
                X = self.evaluator(B, np.column_stack([m_clean, tau_clean])).reshape(-1, 1)
                # weights_tau = 1/(tau_clean**2)
                
                # Fit the linear regression model for alpha_i
                reg = LinearRegression(fit_intercept=False)
                reg.fit(X, totVar_clean)

                alpha_i = reg.coef_[0]
                scores.append(alpha_i)
                mse_list.append(np.mean((totVar_clean - reg.predict(X))**2))
            
            avg_mse = np.mean(mse_list)
            print(f'FPC 1, Iteration {j+1} : MSE change = {(avg_mse - old_mse) / old_mse}, Max B Change = {maxBChange}')
            if np.abs(avg_mse - old_mse) / old_mse < threshold:
                break
            
            if maxBChange < threshold:
                break
            
            old_mse = avg_mse
            
            # Minimize loss for beta
            # (sum_XWX + P) @ Beta = sum_XWy
            n_Beta = self.nb_spline_moneyness * self.nb_spline_tau
            sum_XWX = np.zeros((n_Beta, n_Beta))
            sum_XWy = np.zeros(n_Beta)
            
            for i in range(len(self.cleaned_data)):
                m_clean, tau_clean, totVar_clean = self.cleaned_data[i]
                if len(totVar_clean) == 0: continue
                
                # Create weights such that every day has a standardized weight in the regression no matter the number of observations of the surface on that day
                I_i = len(totVar_clean)
                W_i = self.dailyWeights[i] / I_i 
                W_tau = np.diag(1/(tau_clean**2))
                
                # Basis matrices
                D_m = BSpline.design_matrix(m_clean, self.t_m, self.order_moneyness - 1).toarray()
                D_t = BSpline.design_matrix(tau_clean, self.t_t, self.order_tau - 1).toarray()
                
                # Row-wise Kronecker product using einsum
                X_i = np.einsum('ja,jb->jab', D_m, D_t).reshape(len(totVar_clean), -1)
                
                alpha_i = scores[i]
                sum_XWX += W_i * (alpha_i**2) * (X_i.T @ W_tau @ X_i)
                sum_XWy += W_i * alpha_i * (X_i.T @ W_tau @ totVar_clean)
            
            
            if monotone:
                isNotMonotone = np.zeros((S_tau-1)*S_m)
                for it in range(30):
                    weightMonotone = np.diag(isNotMonotone * omegaMonotone)
                    P_monotone = P_1.T @ weightMonotone @ P_1
                    Beta = np.linalg.solve(sum_XWX + P + P_monotone, sum_XWy + P_1.T @ weightMonotone @ delta)
                    isNotMonotone_new = (P_1 @ Beta <= delta) * 1
                    if(np.sum(isNotMonotone != isNotMonotone_new) == 0): break
                    isNotMonotone = isNotMonotone_new
                    
            else:
                # Solve penalized normal equations
                Beta = np.linalg.solve(sum_XWX + P, sum_XWy)
            
            # Row-major reshaping used to recover matrix form
            B = Beta.reshape(self.nb_spline_moneyness, self.nb_spline_tau)
            
            # Rescale the B s.t. 2-norm of eigen function is 1
            norm = self.compute_norm(B)
            B = B / norm
            j += 1
            
            maxBChange = np.max(np.abs(B - old_B))
            old_B = B
        
        self.scoreMat[:, 0] = scores
        self.BList.append(B)
        return scores, B
        
    def subsequent_FPC_fit(self, threshold = 1e-4, maxit = 10, omega_m = 1.0, omega_m2 = 0.0, omega_t = 1.0, d_m = 2, d_m2 = 3, d_t = 2):
        """
        Estimate subsequent FPCs and their scores conditional on the FPC being orthogonal to all previous FPCs.

        Parameters:
        - threshold: indicate what change in MSE do we consider as reaching convergence
        - maxit: maximum number of iterations
        - omega_m: penalty weight for the moneyness dimension
        - omega_m2: penalty weight for the second moneyness dimension penalty
        - omega_t: penalty weight for the tau dimension
        - d_m: order of the difference penalty for moneyness
        - d_m2: order of the second difference penalty for moneyness
        - d_t: order of the difference penalty for tau

        Returns:
        - scores: List of array of estimated scores for the current FPC
        - B: 2D array of dimension (nb_spline_moneyness X nb_spline_tau) specifying the fitted FPC
        """
        
        B = self.B0
        
        # Rescale the B s.t. 2-norm of eigen function is 1
        norm = self.compute_norm(B)
        B = B / norm
        
        old_mse = 1e10 # Large initial value
        old_B = B
        maxBChange = 1e10
        j = 0
        
        # Number of current FPC being fitted
        curr_fpc_idx = len(self.BList) + 1
        
        # Pre-compute fixed residual surfaces from previously fitted FPCs and their fixed scores
        cleaned_residuals = []
        for i in range(len(self.cleaned_data)):
            m_clean, tau_clean, totVar_clean = self.cleaned_data[i]
            if len(totVar_clean) == 0:
                cleaned_residuals.append(np.array([]))
            else:
                prev_fit = np.zeros(len(totVar_clean))
                for k in range(len(self.BList)):
                    prev_fit += self.scoreMat[i, k] * self.evaluator(self.BList[k], np.column_stack([m_clean, tau_clean]))
                cleaned_residuals.append(totVar_clean - prev_fit)

        # Add a column for the new FPC scores
        self.scoreMat = np.column_stack((self.scoreMat, np.zeros((len(self.cleaned_data), 1))))
        
        # --- Precompute Penalty Matrix P ---
        K = self.nb_spline_moneyness
        L = self.nb_spline_tau
        D_diff_m = np.diff(np.eye(K), n=d_m, axis=0)
        D_diff_m2 = np.diff(np.eye(K), n=d_m2, axis=0)
        D_diff_t = np.diff(np.eye(L), n=d_t, axis=0)
        
        # Note: Row-major (C-style) Kronecker order used here
        P_m = np.kron(D_diff_m.T @ D_diff_m, np.diag(1/(fpca.t_t[4:]**2)))
        P_m2 = np.kron(D_diff_m2.T @ D_diff_m2, np.diag(1/(fpca.t_t[4:]**2)))
        P_t = np.kron(np.eye(K), D_diff_t.T @ D_diff_t)
        P = omega_m * P_m + omega_m2 * P_m2 + omega_t * P_t
        
        while j < maxit:
            mse_list = []  
            for i in range(len(self.cleaned_data)):
                m_clean, tau_clean, totVar_clean = self.cleaned_data[i]
                resid_i = cleaned_residuals[i]
                
                if len(totVar_clean) == 0:
                    self.scoreMat[i, -1] = 0.0
                    continue

                # Evaluate current FPC candidate on the discrete (moneyness, tau) points
                psi_k = self.evaluator(B, np.column_stack([m_clean, tau_clean]))
                X = psi_k.reshape(-1, 1)

                # Fit univariate linear regression model for ONLY the current FPC score
                reg = LinearRegression(fit_intercept=False)
                reg.fit(X, resid_i)

                alpha_k_i = reg.coef_[0]
                self.scoreMat[i, -1] = alpha_k_i
                mse_list.append(np.mean((resid_i - reg.predict(X))**2))
            
            avg_mse = np.mean(mse_list)
            print(f'FPC {curr_fpc_idx}, Iteration {j+1} : MSE change = {(avg_mse - old_mse) / old_mse}, Max B Change = {maxBChange}')
            if np.abs(avg_mse - old_mse) / old_mse < threshold:
                break
            
            if maxBChange < threshold:
                break
            
            old_mse = avg_mse
            
            # Minimize loss for Beta subject to orthogonality constraints
            n_Beta = self.nb_spline_moneyness * self.nb_spline_tau
            sum_XWX = np.zeros((n_Beta, n_Beta))
            sum_XWy = np.zeros(n_Beta)
            
            for i in range(len(self.cleaned_data)):
                m_clean, tau_clean, totVar_clean = self.cleaned_data[i]
                resid_i = cleaned_residuals[i]
                if len(totVar_clean) == 0: continue
                
                # Create weights such that every day has a standardized weight in the regression no matter the number of observations of the surface on that day
                I_i = len(totVar_clean)
                W_i = self.dailyWeights[i] / I_i 
                W_tau = np.diag(1/(tau_clean**2))
                
                # Basis matrices
                D_m = BSpline.design_matrix(m_clean, self.t_m, self.order_moneyness - 1).toarray()
                D_t = BSpline.design_matrix(tau_clean, self.t_t, self.order_tau - 1).toarray()
                
                # Row-wise Kronecker product using einsum to build basis design matrix
                X_i = np.einsum('ja,jb->jab', D_m, D_t).reshape(len(totVar_clean), -1)
                
                curr_alpha = self.scoreMat[i, -1]
                
                sum_XWX += W_i * (curr_alpha**2) * (X_i.T @ W_tau @ X_i)
                sum_XWy += W_i * curr_alpha * (X_i.T @ W_tau @ resid_i)
            
            # Constraints matrix A: A @ Beta = 0 for each previous FPC
            A_list = []
            for prev_B in self.BList:
                A_list.append((self.W_m @ prev_B @ self.W_t).flatten())
            A = np.array(A_list)
            
            # Solve normal equations with linear equality constraints using KKT system:
            KKT_A = np.block([
                [sum_XWX + P, A.T],
                [A, np.zeros((len(self.BList), len(self.BList)))]
            ])
            KKT_b = np.concatenate([sum_XWy, np.zeros(len(self.BList))])
            
            try:
                sol = np.linalg.solve(KKT_A, KKT_b)
                Beta = sol[:n_Beta]
            except np.linalg.LinAlgError:
                print('error in KKT system')
                # Fallback to least squares if KKT matrix is singular
                Beta, _, _, _ = np.linalg.lstsq(KKT_A, KKT_b, rcond=None)
                Beta = Beta[:n_Beta]
            
            # Row-major reshaping used to recover matrix form
            B = Beta.reshape(self.nb_spline_moneyness, self.nb_spline_tau)
            
            # Rescale the B s.t. 2-norm of eigen function is 1
            norm = self.compute_norm(B)
            B = B / norm
            j += 1
            
            maxBChange = np.max(np.abs(B - old_B))
            old_B = B
            
        self.BList.append(B)
        return self.scoreMat[:, -1].tolist(), B

    def reconstruct_surface(self, coords, BList, scores):
        """
        Reconstructs the implied variance surface at given coordinates using provided Bs and scores.
        
        Parameters:
        - coords: 2D array of shape (n, 2)
        - BList: List of 2D B matrices
        - scores: 1D array of scores corresponding to each B
        
        Returns:
        - reconstructed_totVar: 1D array of length n
        """
        reconstructed_totVar = np.zeros(len(coords))
        for k in range(len(BList)):
            reconstructed_totVar += scores[k] * self.evaluator(BList[k], coords)
        return reconstructed_totVar

    def project_to_scores(self, coords, totVar, BList):
        """
        Projects the given observations (coords and totVar) onto the list of eigenfunctions (BList)
        to estimate the scores using linear regression without intercept.
        
        Parameters:
        - coords: 2D array of shape (n, 2)
        - totVar: 1D array of implied variances corresponding to coords
        - BList: List of 2D B matrices representing the eigenfunctions
        
        Returns:
        - scores: 1D array of length len(BList)
        """
        if len(totVar) == 0 or len(BList) == 0:
            return np.zeros(len(BList))
            
        X = np.zeros((len(totVar), len(BList)))
        for k in range(len(BList)):
            X[:, k] = self.evaluator(BList[k], coords)
            
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, totVar)
        return reg.coef_

    def cross_validate_penalties(self, omega_m_grid, omega_t_grid, omega_m2_grid=None, n_splits=5, 
                                 fpc_index=0, previous_BList=None, threshold=1e-4, maxit=10, 
                                 d_m=2, d_m2=3, d_t=2,
                                 train_window_size=None, test_window_size=None,
                                 theta_cal=0.0, theta_but=0.0, friction_tol=1e-4):
        """
        Runs grid search cross-validation to select optimal smoothness hyperparameters (omega_m, omega_m2, omega_t).
        
        Parameters:
        - omega_m_grid: list/array of candidate values for omega_m
        - omega_t_grid: list/array of candidate values for omega_t
        - omega_m2_grid: list/array of candidate values for omega_m2. If None, defaults to [0.0].
        - n_splits: number of CV folds
        - fpc_index: index of the FPC being fitted (0 for first, >0 for subsequent)
        - previous_BList: list of B matrices for already fitted FPCs (required if fpc_index > 0)
        - threshold: convergence threshold for FPC fitting
        - maxit: maximum iterations for FPC fitting
        - d_m: order of difference penalty for moneyness
        - d_m2: order of second difference penalty for moneyness
        - d_t: order of difference penalty for tau
        - train_window_size: size of the rolling training window (int). If None, uses an expanding window (all past data).
        - test_window_size: size of the future testing/validation window (int). If None, defaults to n_days // (n_splits + 1).
        - theta_cal: weight of calendar spread arbitrage violation magnitude in CV score
        - theta_but: weight of butterfly spread arbitrage violation magnitude in CV score
        - friction_tol: tolerance threshold below which arbitrage violations are ignored (representing transaction costs)
        
        Returns:
        - best_omega_m: float
        - best_omega_m2: float
        - best_omega_t: float
        - results: list of dicts containing details for each grid point
        """
            
        if fpc_index > 0 and previous_BList is None:
            raise ValueError("previous_BList must be provided when fpc_index > 0")
            
        if previous_BList is None:
            previous_BList = []
            
        if omega_m2_grid is None:
            omega_m2_grid = [0.0]
            
        folds = self._split_folds(n_splits, train_window_size, test_window_size)
        
        best_score = np.inf
        best_omega_m = None
        best_omega_m2 = None
        best_omega_t = None
        results = []
        
        for omega_m in omega_m_grid:
            for omega_m2 in omega_m2_grid:
                for omega_t in omega_t_grid:
                    # Run CV for this grid point
                    mean_val_score, mean_val_mse, mean_cal_viol, mean_but_viol = self._evaluate_grid_point(
                        omega_m, omega_m2, omega_t, folds, fpc_index, previous_BList,
                        threshold, maxit, d_m, d_m2, d_t,
                        theta_cal=theta_cal, theta_but=theta_but, friction_tol=friction_tol
                    )
                    
                    results.append({
                        'omega_m': omega_m,
                        'omega_m2': omega_m2,
                        'omega_t': omega_t,
                        'mean_val_score': mean_val_score,
                        'mean_val_mse': mean_val_mse,
                        'mean_cal_viol': mean_cal_viol,
                        'mean_but_viol': mean_but_viol
                    })
                    
                    print(f"CV Grid: omega_m={omega_m:.4f}, omega_m2={omega_m2:.4f}, omega_t={omega_t:.4f} -> Mean Val Score = {mean_val_score:.8f} (MSE={mean_val_mse:.8f}, CalViol={mean_cal_viol:.8f}, ButViol={mean_but_viol:.8f})")
                    
                    if mean_val_score < best_score:
                        best_score = mean_val_score
                        best_omega_m = omega_m
                        best_omega_m2 = omega_m2
                        best_omega_t = omega_t
                        
        return best_omega_m, best_omega_m2, best_omega_t, results

    def _split_folds(self, n_splits, train_window_size=None, test_window_size=None):
        """
        Generates fold indices/splits for cross-validation.
        """
        n_days = len(self.cleaned_data)
        test_size = test_window_size if test_window_size is not None else n_days // (n_splits + 1)
        test_size = max(1, test_size)
        
        folds = []
        for k in range(n_splits):
            val_start = n_days - (n_splits - k) * test_size
            val_end = val_start + test_size
            
            if val_start <= 0:
                raise ValueError(
                    f"Not enough days ({n_days}) for {n_splits} splits with test_window_size={test_size}. "
                    f"At split index {k}, validation start index is {val_start} which is <= 0."
                )
            
            if train_window_size is None:
                train_start = 0
            else:
                train_start = max(0, val_start - train_window_size)
                if train_start >= val_start:
                    raise ValueError(
                        f"Training window starts at or after validation start index at split {k} "
                        f"(train_start={train_start}, val_start={val_start})."
                    )
            
            train_idx = np.arange(train_start, val_start)
            val_idx = np.arange(val_start, val_end)
            folds.append((train_idx, val_idx))
        return folds
        
    def _evaluate_grid_point(self, omega_m, omega_m2, omega_t, folds, fpc_index, previous_BList,
                             threshold, maxit, d_m, d_m2, d_t,
                             theta_cal=0.0, theta_but=0.0, friction_tol=1e-4):
        """
        Evaluates a single grid point (omega_m, omega_m2, omega_t) over all folds.
        """
        n_splits = len(folds)
            
        print(n_splits)
            
        val_scores = []
        val_mses = []
        val_cal_viols = []
        val_but_viols = []
        
        for fold_idx in range(n_splits):
            train_idx, val_idx = folds[fold_idx]
            fold_res = self._evaluate_fold_day(
                train_idx, val_idx, omega_m, omega_m2, omega_t, fpc_index, previous_BList,
                threshold, maxit, d_m, d_m2, d_t,
                theta_cal=theta_cal, theta_but=theta_but, friction_tol=friction_tol
            )
                
            if fold_res is not None:
                val_scores.append(fold_res['score'])
                val_mses.append(fold_res['mse'])
                val_cal_viols.append(fold_res['cal_viol'])
                val_but_viols.append(fold_res['but_viol'])
                
        if not val_scores:
            return np.inf, np.inf, np.inf, np.inf
            
        return np.mean(val_scores), np.mean(val_mses), np.mean(val_cal_viols), np.mean(val_but_viols)

    def _evaluate_fold_day(self, train_idx, val_idx, omega_m, omega_m2, omega_t, fpc_index, previous_BList,
                           threshold, maxit, d_m, d_m2, d_t,
                           theta_cal=0.0, theta_but=0.0, friction_tol=1e-4):
        """
        Fits the model on train days and evaluates on validation days.
        """
        # Create a new FPCA_penalized_totVar instance for training on the subset of days
        fpca_train = FPCA_penalized_totVar(
            moneyness=[], tau=[], totVar=[],
            nb_spline_moneyness=self.nb_spline_moneyness,
            nb_spline_tau=self.nb_spline_tau,
            order_moneyness=self.order_moneyness,
            order_tau=self.order_tau,
            S=[self.S[idx] for idx in train_idx] if self.S is not None else None,
            r=[self.r[idx] for idx in train_idx] if self.r is not None else None,
            q=[self.q[idx] for idx in train_idx] if self.q is not None else None,
            range_moneyness=self.range_moneyness,
            range_tau=self.range_tau,
            dailyWeights=None
        )
        # Assign train day data directly
        fpca_train.cleaned_data = [self.cleaned_data[i] for i in train_idx]
        fpca_train.dailyWeights = np.asarray(self.dailyWeights)[train_idx]
        fpca_train.BList = list(previous_BList)
        
        # Fit the candidate FPC
        if fpc_index == 0:
            fpca_train.scoreMat = np.zeros((len(train_idx), 1))
            _, B_new = fpca_train.first_FPC_fit(
                threshold=threshold, maxit=maxit, omega_m=omega_m, omega_m2=omega_m2, omega_t=omega_t, d_m=d_m, d_m2=d_m2, d_t=d_t
            )
        else:
            fpca_train.scoreMat = np.zeros((len(train_idx), len(previous_BList)))
            for local_i, global_i in enumerate(train_idx):
                m_clean, tau_clean, totVar_clean = self.cleaned_data[global_i]
                if len(totVar_clean) > 0:
                    coords = np.column_stack([m_clean, tau_clean])
                    fpca_train.scoreMat[local_i, :] = fpca_train.project_to_scores(coords, totVar_clean, previous_BList)
            _, B_new = fpca_train.subsequent_FPC_fit(
                threshold=threshold, maxit=maxit, omega_m=omega_m, omega_m2=omega_m2, omega_t=omega_t, d_m=d_m, d_m2=d_m2, d_t=d_t
            )
            
        BList_val = previous_BList + [B_new]
        
        # Evaluate on validation days
        val_fold_scores = []
        val_mses = []
        val_cal_viols = []
        val_but_viols = []
        for i in val_idx:
            m_val, tau_val, totVar_val = self.cleaned_data[i]
            if len(totVar_val) == 0:
                continue
            coords_val = np.column_stack([m_val, tau_val])
            scores_val = self.project_to_scores(coords_val, totVar_val, BList_val)
            y_hat_val = self.reconstruct_surface(coords_val, BList_val, scores_val)
            mse_val = np.mean((totVar_val - y_hat_val)**2)
            
            if theta_cal > 0.0 or theta_but > 0.0:
                m_grid = np.linspace(self.range_moneyness[0], self.range_moneyness[1], 15)
                tau_grid = np.linspace(max(1e-4, self.range_tau[0]), self.range_tau[1], 15)
                
                original_BList = self.BList
                self.BList = BList_val
                cal_mat, but_mat = self.compute_arbitrage_metrics(scores_val, m_grid=m_grid, tau_grid=tau_grid)
                self.BList = original_BList
                
                cal_viol_mag = np.mean(np.maximum(0.0, -cal_mat - friction_tol))
                but_viol_mag = np.mean(np.maximum(0.0, -but_mat - friction_tol))
                
                day_score = mse_val + theta_cal * cal_viol_mag + theta_but * but_viol_mag
            else:
                cal_viol_mag = 0.0
                but_viol_mag = 0.0
                day_score = mse_val
                
            val_fold_scores.append(day_score)
            val_mses.append(mse_val)
            val_cal_viols.append(cal_viol_mag)
            val_but_viols.append(but_viol_mag)
            
        if not val_fold_scores:
            return None
            
        return {
            'score': np.mean(val_fold_scores),
            'mse': np.mean(val_mses),
            'cal_viol': np.mean(val_cal_viols),
            'but_viol': np.mean(val_but_viols)
        }

    def _evaluate_fold_observation(self, fold_obs_splits, fold_idx, omega_m, omega_m2, omega_t, fpc_index, previous_BList,
                                   threshold, maxit, d_m, d_m2, d_t,
                                   theta_cal=0.0, theta_but=0.0, friction_tol=1e-4):
        """
        Fits the model on train observations and evaluates on validation observations.
        """
        n_days = len(self.cleaned_data)
        
        train_cleaned_data = []
        val_data = []
        
        for i in range(n_days):
            m_clean, tau_clean, totVar_clean = self.cleaned_data[i]
            day_folds = fold_obs_splits[i]
            
            val_idx = day_folds[fold_idx]
            train_idx = np.concatenate([day_folds[j] for j in range(len(day_folds)) if j != fold_idx])
            
            if len(train_idx) > 0:
                train_cleaned_data.append((m_clean[train_idx], tau_clean[train_idx], totVar_clean[train_idx]))
            else:
                train_cleaned_data.append((np.array([]), np.array([]), np.array([])))
                
            if len(val_idx) > 0:
                val_data.append((np.column_stack([m_clean[val_idx], tau_clean[val_idx]]), totVar_clean[val_idx], train_idx))
            else:
                val_data.append(None)
                
        fpca_train = FPCA_penalized_totVar(
            moneyness=[], tau=[], totVar=[],
            nb_spline_moneyness=self.nb_spline_moneyness,
            nb_spline_tau=self.nb_spline_tau,
            order_moneyness=self.order_moneyness,
            order_tau=self.order_tau,
            S=self.S,
            r=self.r,
            q=self.q,
            range_moneyness=self.range_moneyness,
            range_tau=self.range_tau,
            dailyWeights=None
        )
        fpca_train.cleaned_data = train_cleaned_data
        fpca_train.dailyWeights = self.dailyWeights
        fpca_train.BList = list(previous_BList)
        
        if fpc_index == 0:
            fpca_train.scoreMat = np.zeros((n_days, 1))
            _, B_new = fpca_train.first_FPC_fit(
                threshold=threshold, maxit=maxit, omega_m=omega_m, omega_m2=omega_m2, omega_t=omega_t, d_m=d_m, d_m2=d_m2, d_t=d_t
            )
        else:
            fpca_train.scoreMat = np.zeros((n_days, len(previous_BList)))
            for i in range(n_days):
                m_clean, tau_clean, totVar_clean = train_cleaned_data[i]
                if len(totVar_clean) > 0:
                    coords = np.column_stack([m_clean, tau_clean])
                    fpca_train.scoreMat[i, :] = fpca_train.project_to_scores(coords, totVar_clean, previous_BList)
            _, B_new = fpca_train.subsequent_FPC_fit(
                threshold=threshold, maxit=maxit, omega_m=omega_m, omega_m2=omega_m2, omega_t=omega_t, d_m=d_m, d_m2=d_m2, d_t=d_t
            )
            
        BList_val = previous_BList + [B_new]
        
        val_fold_scores = []
        val_mses = []
        val_cal_viols = []
        val_but_viols = []
        for i in range(n_days):
            val_info = val_data[i]
            if val_info is None:
                continue
            coords_val, totVar_val, train_idx = val_info
            
            m_clean, tau_clean, totVar_clean = self.cleaned_data[i]
            coords_train = np.column_stack([m_clean[train_idx], tau_clean[train_idx]])
            totVar_train = totVar_clean[train_idx]
            
            if len(totVar_train) == 0:
                continue
                
            scores_train_i = self.project_to_scores(coords_train, totVar_train, BList_val)
            
            y_hat_val = self.reconstruct_surface(coords_val, BList_val, scores_train_i)
            mse_val = np.mean((totVar_val - y_hat_val)**2)
            
            if theta_cal > 0.0 or theta_but > 0.0:
                m_grid = np.linspace(self.range_moneyness[0], self.range_moneyness[1], 15)
                tau_grid = np.linspace(max(1e-4, self.range_tau[0]), self.range_tau[1], 15)
                
                original_BList = self.BList
                self.BList = BList_val
                cal_mat, but_mat = self.compute_arbitrage_metrics(scores_train_i, m_grid=m_grid, tau_grid=tau_grid)
                self.BList = original_BList
                
                cal_viol_mag = np.mean(np.maximum(0.0, -cal_mat - friction_tol))
                but_viol_mag = np.mean(np.maximum(0.0, -but_mat - friction_tol))
                
                day_score = mse_val + theta_cal * cal_viol_mag + theta_but * but_viol_mag
            else:
                cal_viol_mag = 0.0
                but_viol_mag = 0.0
                day_score = mse_val
                
            val_fold_scores.append(day_score)
            val_mses.append(mse_val)
            val_cal_viols.append(cal_viol_mag)
            val_but_viols.append(but_viol_mag)
            
        if not val_fold_scores:
            return None
            
        return {
            'score': np.mean(val_fold_scores),
            'mse': np.mean(val_mses),
            'cal_viol': np.mean(val_cal_viols),
            'but_viol': np.mean(val_but_viols)
        }

    def totVar_to_price_surface(self, coords, totVar, day_index=0):
        """
        Computes the Black-Scholes call option price surface from the implied variance surface.
        
        Parameters:
        - coords: 2D array of shape (n, 2) where column 0 is log-moneyness and column 1 is time to expiry (tau).
        - totVar: 1D array of implied variances corresponding to coords.
        - day_index: Index of the day to retrieve the underlying stock price (S), risk-free rate (r), and dividend rate (q).
        
        Returns:
        - call_prices: 1D array of Call option prices.
        """
        from scipy.stats import norm
        
        log_m = np.asarray(coords[:, 0])
        tau = np.asarray(coords[:, 1])
        
        S_val = self.S[day_index] if self.S is not None and day_index < len(self.S) else 1.0
        r_val = self.r[day_index] if self.r is not None and day_index < len(self.r) else 0.0
        q_val = self.q[day_index] if self.q is not None and day_index < len(self.q) else 0.0
        
        K = S_val * np.exp(log_m)
        
        tau_safe = np.maximum(tau, 1e-10)
        totVar_safe = np.maximum(totVar, 1e-10)
        iv_safe = np.sqrt(totVar_safe/tau_safe)
        
        d1 = (-log_m + (r_val - q_val + 0.5 * totVar_safe) * tau_safe) / (iv_safe * np.sqrt(tau_safe))
        d2 = d1 - iv_safe * np.sqrt(tau_safe)
        
        call_prices = S_val * np.exp(-q_val * tau_safe) * norm.cdf(d1) - K * np.exp(-r_val * tau_safe) * norm.cdf(d2)
        
        # Exact intrinsic value for expiration (tau=0)
        call_prices = np.where(tau <= 0, np.maximum(S_val - K, 0), call_prices)
        
        return call_prices

    def compute_explained_variance(self):
        """
        Computes the proportion of variance explained by each FPC using reconstruction.
        
        Returns:
        - prop_var_marginal: List of marginal proportion of variance explained by each FPC.
        """
        num_fpcs = len(self.BList)
        total_tss = 0
        cumulative_rss = np.zeros(num_fpcs)
        
        for i in range(len(self.cleaned_data)):
            m_clean, tau_clean, totVar_clean = self.cleaned_data[i]
            if len(totVar_clean) == 0: continue
            
            coords = np.column_stack([m_clean, tau_clean])
            total_tss += np.sum(totVar_clean**2)
            
            for k in range(1, num_fpcs + 1):
                y_hat = self.reconstruct_surface(coords, self.BList[:k], self.scoreMat[i, :k])
                cumulative_rss[k-1] += np.sum((totVar_clean - y_hat)**2)
                
        # Cumulative proportion of variance explained
        prop_var_cum = 1 - (cumulative_rss / total_tss)
        
        # Marginal proportion of variance explained
        prop_var_marginal = np.zeros(num_fpcs)
        prop_var_marginal[0] = prop_var_cum[0]
        if num_fpcs > 1:
            prop_var_marginal[1:] = np.diff(prop_var_cum)
            
        return prop_var_marginal.tolist()
    
    def _compute_gram_matrix(self, t, order, range_t):
        """
        Computes the Gram matrix (inner products) of B-spline basis functions.
        """
        n_basis = len(t) - order
        W = np.zeros((n_basis, n_basis))
        
        # Quadrature points for degree 2k
        quad_points, quad_weights = roots_legendre(order)
        
        # Unique knots
        uknots = np.unique(t[(t>=range_t[0]) & (t<=range_t[1])])
        
        for i in range(len(uknots) - 1):
            a, b = uknots[i], uknots[i+1]
            if a == b: continue
            
            # Map quadrature points to [a, b]
            x = 0.5 * (b - a) * quad_points + 0.5 * (b + a)
            w = 0.5 * (b - a) * quad_weights
            
            # Evaluate all basis functions at x
            B = BSpline.design_matrix(x, t, order-1).toarray()
            
            # Integration on this interval: B.T @ diag(w) @ B
            W += (B.T * w) @ B
            
        return W

    def compute_inner_product(self, B1, B2):
        """
        Computes the L2 inner product between two 2D eigenfunctions defined by coefficients B1 and B2.
        <psi1, psi2> = Tr(B1.T @ W_m @ B2 @ W_t)
        """
        inner_prod = np.trace(B1.T @ self.W_m @ B2 @ self.W_t)
        return inner_prod

    def compute_norm(self, B):
        """
        Computes the L2 norm of the 2D eigenfunction defined by coefficients B.
        ||psi||^2 = <psi, psi>
        """
        return np.sqrt(self.compute_inner_product(B, B))
    
    def BSplines_2d(self, range_moneyness=[-.15, .1], range_tau=[0, 1],
                    nb_spline_moneyness=10, nb_spline_tau=10,
                    order_moneyness=4, order_tau=4):
        """
        order_moneyness, order_tau are the order of the spline used in both dimensions respectively.
        """
        k_m = order_moneyness - 1
        k_t = order_tau - 1
        
        # Generate clamped knot vectors
        t_breakPts_m = np.linspace(range_moneyness[0], range_moneyness[1], nb_spline_moneyness - k_m + 1)
        self.t_m = np.concatenate(([range_moneyness[0]] * k_m, t_breakPts_m, [range_moneyness[1]] * k_m))
        
        t_breakPts_t = np.linspace(range_tau[0], range_tau[1], nb_spline_tau - k_t + 1)**2
        self.t_t = np.concatenate(([range_tau[0]] * k_t, t_breakPts_t, [range_tau[1]] * k_t))
        
        # Pre-compute Gram matrices for normalization
        self.W_m = self._compute_gram_matrix(self.t_m, order_moneyness, range_moneyness)
        self.W_t = self._compute_gram_matrix(self.t_t, order_tau, range_tau)
        
        def evaluator(B, coords):
            """
            Evaluates the tensor product B-spline surface.
            
            Parameters:
            - B: 2D array of dimension (nb_spline_moneyness X nb_spline_tau)
            - coords: 2D array of shape (n, 2), where column 0 is moneyness and column 1 is tau.
            
            Returns:
            - 1D array of length n containing the evaluated implied variances.
            """
            x_m = np.clip(np.asarray(coords[:, 0]), range_moneyness[0], range_moneyness[1])
            x_t = np.clip(np.asarray(coords[:, 1]), range_tau[0], range_tau[1])
            
            D_m = BSpline.design_matrix(x_m, self.t_m, order_moneyness - 1)
            D_t = BSpline.design_matrix(x_t, self.t_t, order_tau - 1)
            
            res = np.sum((D_m @ B) * D_t.toarray(), axis=1)
            
            return res
            
        self.evaluator = evaluator
        
    def plot_eigen_functions(self, B, num_points=30, figAngle=-70):
        """
        Plots the B-spline surface given a coefficient matrix B.
        
        Parameters:
        - B: 2D array of dimension (nb_spline_moneyness X nb_spline_tau)
        - num_points: Number of points along each dimension for the grid.
        - figAngle: Viewing angle (azimuth).
        """
        m_vals = np.linspace(self.range_moneyness[0], self.range_moneyness[1], num_points)
        tau_vals = np.linspace(self.range_tau[0], self.range_tau[1], num_points)
        M, Tau = np.meshgrid(m_vals, tau_vals)
        
        coords = np.column_stack([M.flatten(), Tau.flatten()])
        
        Z_flat = self.evaluator(B, coords)
        Z = Z_flat.reshape(M.shape)
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot_surface(M, Tau, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        ax.set_title("B-Spline Evaluated Surface")
        ax.set_xlabel("Moneyness (m)")
        ax.set_ylabel("Time to Expiry (tau)")
        
        ax.view_init(elev=15, azim=figAngle)
        plt.show()

    def plot_reconstruction_ivs(self, index, num_points=30, figAngle=-70):
        """
        Plots the reconstructed IVS (converted to implied volatility) for a given day and overlaps the actual observations.
        
        Parameters:
        - index: Index of the day to plot.
        - num_points: Number of points along each dimension for the grid.
        - figAngle: Viewing angle (azimuth).
        """
        m_clean, tau_clean, totVar_clean = self.cleaned_data[index]
        scores = self.scoreMat[index, :]
        
        # Grid for surface
        m_vals = np.linspace(self.range_moneyness[0], self.range_moneyness[1], num_points)
        tau_vals = np.linspace(self.range_tau[0], self.range_tau[1], num_points)
        M, Tau = np.meshgrid(m_vals, tau_vals)
        coords_grid = np.column_stack([M.flatten(), Tau.flatten()])
        
        # Reconstruct on grid (returns implied variance v)
        Z_flat = self.reconstruct_surface(coords_grid, self.BList, scores)
        Z = Z_flat.reshape(M.shape)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot reconstructed surface converted to IV
        ax.plot_surface(M, Tau, np.sqrt(np.maximum(0, Z)/Tau), cmap='viridis', edgecolor='none', alpha=0.5)
        
        # Plot actual observations converted to IV
        ax.scatter(m_clean, tau_clean, np.sqrt(np.maximum(0, totVar_clean)/tau_clean), color='red', s=10, label='Actual Observations')
        
        ax.set_title(f"Reconstructed IVS vs Actual Observations (Day {index})")
        ax.set_xlabel("Moneyness (m)")
        ax.set_ylabel("Time to Expiry (tau)")
        ax.set_zlabel("Implied Volatility")
        
        ax.view_init(elev=15, azim=figAngle)
        plt.legend()
        plt.show()

    def plot_reconstruction_totVar(self, index, num_points=30, figAngle=-70):
        """
        Plots the reconstructed Implied Variance Surface for a given day and overlaps the actual observations.
        
        Parameters:
        - index: Index of the day to plot.
        - num_points: Number of points along each dimension for the grid.
        - figAngle: Viewing angle (azimuth).
        """
        m_clean, tau_clean, totVar_clean = self.cleaned_data[index]
        scores = self.scoreMat[index, :]
        
        # Grid for surface
        m_vals = np.linspace(self.range_moneyness[0], self.range_moneyness[1], num_points)
        tau_vals = np.linspace(self.range_tau[0], self.range_tau[1], num_points)
        M, Tau = np.meshgrid(m_vals, tau_vals)
        coords_grid = np.column_stack([M.flatten(), Tau.flatten()])
        
        # Reconstruct on grid
        Z_flat = self.reconstruct_surface(coords_grid, self.BList, scores)
        Z = Z_flat.reshape(M.shape)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot reconstructed surface
        ax.plot_surface(M, Tau, Z, cmap='viridis', edgecolor='none', alpha=0.5)
        
        # Plot actual observations
        ax.scatter(m_clean, tau_clean, totVar_clean, color='red', s=10, label='Actual Observations')
        
        ax.set_title(f"Reconstructed Implied Variance Surface vs Actual Observations (Day {index})")
        ax.set_xlabel("Moneyness (m)")
        ax.set_ylabel("Time to Expiry (tau)")
        ax.set_zlabel("Implied Variance")
        
        ax.view_init(elev=15, azim=figAngle)
        plt.legend()
        plt.show()

    def compute_arbitrage_metrics(self, scores, m_grid=None, tau_grid=None):
        """
        Computes calendar and butterfly arbitrage metrics on a grid using implied variance.
        
        Parameters:
        - scores: Scores for the reconstruction (1D array).
        - m_grid: Grid of log-moneyness. If None, uses 50 points in range_moneyness.
        - tau_grid: Grid of maturities. If None, uses 50 points in range_tau.
        
        Returns:
        - calendar_metrics: 2D array of shape (len(tau_grid), len(m_grid)) representing d_tau w
        - butterfly_metrics: 2D array of shape (len(tau_grid), len(m_grid)) representing g(m)
        """
        if m_grid is None:
            m_grid = np.linspace(self.range_moneyness[0], self.range_moneyness[1], 50)
        if tau_grid is None:
            tau_min = max(1e-4, self.range_tau[0])
            tau_grid = np.linspace(tau_min, self.range_tau[1], 50)
            
        M, T = np.meshgrid(m_grid, tau_grid)
        coords = np.column_stack([M.flatten(), T.flatten()])
        
        # totVar is w(m, tau) = tau * sigma(m, tau)^2
        totVar_flat = self.reconstruct_surface(coords, self.BList, scores)
        totVar = totVar_flat.reshape(M.shape)
        
        # Total variance w(m, tau) = v(m, tau) * tau = totVar
        w = totVar
        
        # 1. Calendar Spread Metric: d_tau w
        calendar_metrics = np.zeros_like(w)
        for j in range(len(m_grid)):
            w_slice = w[:, j]
            # Central difference
            calendar_metrics[1:-1, j] = (w_slice[2:] - w_slice[:-2]) / (tau_grid[2:] - tau_grid[:-2])
            # One-sided differences at boundaries
            calendar_metrics[0, j] = (w_slice[1] - w_slice[0]) / (tau_grid[1] - tau_grid[0])
            calendar_metrics[-1, j] = (w_slice[-1] - w_slice[-2]) / (tau_grid[-1] - tau_grid[-2])
            
        # 2. Butterfly Metric: g(m)
        butterfly_metrics = np.zeros_like(w)
        for i in range(len(tau_grid)):
            w_slice = w[i, :]
            m_slice = m_grid
            
            # d_m w (central difference)
            w_m = np.zeros_like(w_slice)
            w_m[1:-1] = (w_slice[2:] - w_slice[:-2]) / (m_slice[2:] - m_slice[:-2])
            w_m[0] = (w_slice[1] - w_slice[0]) / (m_slice[1] - m_slice[0])
            w_m[-1] = (w_slice[-1] - w_slice[-2]) / (m_slice[-1] - m_slice[-2])
            
            # d_mm w (central difference for non-uniform grid)
            w_mm = np.zeros_like(w_slice)
            h1 = m_slice[1:-1] - m_slice[:-2]
            h2 = m_slice[2:] - m_slice[1:-1]
            w_mm[1:-1] = 2 * ((w_slice[2:] - w_slice[1:-1])/h2 - (w_slice[1:-1] - w_slice[:-2])/h1) / (h1 + h2)
            # Boundary handling for second derivative: set to nearest calculated value
            w_mm[0] = w_mm[1]
            w_mm[-1] = w_mm[-2]
            
            # g(m) = (1 - m*w_m/(2w))^2 - (w_m^2/4)*(1/w + 1/4) + w_mm/2
            term1 = (1 - m_slice * w_m / (2 * w_slice))**2
            term2 = (w_m**2 / 4) * (1 / w_slice + 0.25)
            term3 = w_mm / 2
            butterfly_metrics[i, :] = term1 - term2 + term3
            
        return calendar_metrics, butterfly_metrics

    def compute_price_arbitrage_metrics(self, scores, m_grid=None, tau_grid=None, day_index=0):
        """
        Computes calendar, call spread, and butterfly spread arbitrage penalty matrices 
        on a grid using relative call prices (c = Call / S), according to Cont and Vuletic (2023).
        
        Parameters:
        - scores: Scores for the reconstruction (1D array).
        - m_grid: Grid of log-moneyness. If None, uses 50 points in range_moneyness.
        - tau_grid: Grid of maturities. If None, uses 50 points in range_tau.
        - day_index: Index of the day to retrieve the underlying stock price (S).
        
        Returns:
        - P1: 2D array representing calendar spread arbitrage penalties.
        - P2: 2D array representing call spread arbitrage penalties.
        - P3: 2D array representing butterfly spread arbitrage penalties.
        """
        if m_grid is None:
            m_grid = np.linspace(self.range_moneyness[0], self.range_moneyness[1], 50)
        if tau_grid is None:
            tau_min = max(1e-4, self.range_tau[0])
            tau_grid = np.linspace(tau_min, self.range_tau[1], 50)
            
        M, T = np.meshgrid(m_grid, tau_grid)
        coords = np.column_stack([M.flatten(), T.flatten()])
        
        totVar_flat = self.reconstruct_surface(coords, self.BList, scores)
        
        call_prices_flat = self.totVar_to_price_surface(coords, totVar_flat, day_index=day_index)
        
        S_val = self.S[day_index] if self.S is not None and day_index < len(self.S) else 1.0
        c_flat = call_prices_flat / S_val
        c = c_flat.reshape(M.shape)
        
        # Absolute moneyness m_i = exp(log-moneyness) = K/S
        m_abs = np.exp(m_grid)
        
        P1 = np.zeros_like(c)
        P2 = np.zeros_like(c)
        P3 = np.zeros_like(c)
        
        # 1. Calendar spread arbitrage (P1)
        for j in range(len(tau_grid) - 1):
            for i in range(len(m_grid)):
                val = tau_grid[j] * (c[j, i] - c[j+1, i]) / (tau_grid[j+1] - tau_grid[j])
                P1[j, i] = max(0, val)
                
        # 2. Call spread arbitrage (P2)
        for j in range(len(tau_grid)):
            for i in range(len(m_grid) - 1):
                val = (c[j, i+1] - c[j, i]) / (m_abs[i+1] - m_abs[i])
                P2[j, i] = max(0, val)
                
        # 3. Butterfly spread arbitrage (P3)
        for j in range(len(tau_grid)):
            for i in range(1, len(m_grid) - 1):
                left_diff = (c[j, i] - c[j, i-1]) / (m_abs[i] - m_abs[i-1])
                right_diff = (c[j, i+1] - c[j, i]) / (m_abs[i+1] - m_abs[i])
                val = left_diff - right_diff
                P3[j, i] = max(0, val)
                
        return P1, P2, P3

    def plot_arbitrage_violations(self, index, num_points=50, figAngle=-70):
        """
        Plots where calendar and butterfly arbitrage violations occur on the reconstructed surface.
        
        Parameters:
        - index: Index of the day to plot.
        - num_points: Number of points along each dimension for the grid.
        - figAngle: Viewing angle (azimuth).
        """
        scores = self.scoreMat[index, :]
        m_vals = np.linspace(self.range_moneyness[0], self.range_moneyness[1], num_points)
        tau_min = max(1e-4, self.range_tau[0])
        tau_vals = np.linspace(tau_min, self.range_tau[1], num_points)
        
        calendar_metrics, butterfly_metrics = self.compute_arbitrage_metrics(scores, m_grid=m_vals, tau_grid=tau_vals)
        
        calendar_metrics = np.minimum(0, calendar_metrics)
        butterfly_metrics = np.minimum(0, butterfly_metrics)
        
        M, T = np.meshgrid(m_vals, tau_vals)
        
        fig = plt.figure(figsize=(16, 7))
        
        # Plot Calendar Spread violations
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(M, T, calendar_metrics, cmap='RdYlGn', edgecolor='none', alpha=0.8)
        ax1.plot_surface(M, T, np.zeros_like(M), color='black', alpha=0.2)
        ax1.set_title(f"Calendar Spread Metric (d_tau w) - Day {index}\n(Violation if < 0)")
        ax1.set_xlabel("Moneyness (m)")
        ax1.set_ylabel("Time to Expiry (tau)")
        ax1.view_init(elev=15, azim=figAngle)
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

        # Plot Butterfly violations
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(M, T, butterfly_metrics, cmap='RdYlGn', edgecolor='none', alpha=0.8)
        ax2.plot_surface(M, T, np.zeros_like(M), color='black', alpha=0.2)
        ax2.set_title(f"Butterfly Metric (g(m)) - Day {index}\n(Violation if < 0)")
        ax2.set_xlabel("Moneyness (m)")
        ax2.set_ylabel("Time to Expiry (tau)")
        ax2.view_init(elev=15, azim=figAngle)
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
        
        plt.tight_layout()
        plt.show()

    def measure_raw_arbitrage(self, day_index, tolerance=1e-8):
        """
        Measures static arbitrage in the raw data for a given day.
        """
        m_clean, tau_clean, totVar_clean = self.cleaned_data[day_index]
        if len(totVar_clean) == 0:
            return {
                'calendar_violations': 0,
                'calendar_violation_sum': 0.0,
                'vertical_violations': 0,
                'vertical_violation_sum': 0.0,
                'butterfly_violations': 0,
                'butterfly_violation_sum': 0.0,
                'total_violations': 0,
                'total_nb_observations': 0
            }
            
        coords = np.column_stack([m_clean, tau_clean])
        call_prices = self.totVar_to_price_surface(coords, totVar_clean, day_index=day_index)
        
        S_val = self.S[day_index] if self.S is not None and day_index < len(self.S) else 1.0
        r_val = self.r[day_index] if self.r is not None and day_index < len(self.r) else 0.0
        
        strikes = S_val * np.exp(m_clean)
        maturities = np.round(tau_clean, 6)
        
        vert_violations, vert_sum, vert_returns, butt_violations, butt_sum, butt_returns = self._check_vertical_and_butterfly_arbitrage(
            call_prices, strikes, maturities, r_val, tolerance
        )
        
        cal_violations, cal_sum, cal_returns = self._check_calendar_arbitrage(
            call_prices, strikes, maturities, tolerance
        )
        
        total_violations = vert_violations + butt_violations + cal_violations
        
        return {
            'calendar_violations': cal_violations,
            'calendar_violation_sum': cal_sum,
            'vertical_violations': vert_violations,
            'vertical_violation_sum': vert_sum,
            'butterfly_violations': butt_violations,
            'butterfly_violation_sum': butt_sum,
            'total_violations': total_violations,
            'total_nb_observations': len(call_prices)
        }

    def _check_vertical_and_butterfly_arbitrage(self, call_prices, strikes, maturities, r_val, tolerance):
        """
        Checks vertical (call spread) and butterfly spread violations for each maturity slice.
        """
        vert_violations = 0
        vert_sum = 0.0
        vertical_returns = []
        butt_violations = 0
        butt_sum = 0.0
        butterfly_returns = []
        
        unique_mats = np.unique(maturities)
        for mat in unique_mats:
            mask = (maturities == mat)
            mat_calls = call_prices[mask]
            mat_strikes = strikes[mask]
            
            sort_idx = np.argsort(mat_strikes)
            c = mat_calls[sort_idx]
            k = mat_strikes[sort_idx]
            
            n = len(c)
            if n < 2:
                continue
                
            if hasattr(r_val, "__len__") and not isinstance(r_val, (str, bytes)):
                r_arr = np.asarray(r_val)
                mat_r = r_arr[mask]
                mat_r_sorted = mat_r[sort_idx]
                discount = np.exp(-mat_r_sorted * mat)
            else:
                discount = np.repeat(np.exp(-r_val * mat), n)
                
            for j in range(n - 1):
                if c[j] < c[j+1] - tolerance:
                    vert_violations += 1
                    violation_mag = c[j+1] - c[j]
                    vert_sum += violation_mag
                    vertical_returns.append(violation_mag / max(c[j], 1e-10))
                
                max_diff = discount[j] * (k[j+1] - k[j])
                if (c[j] - c[j+1]) > max_diff + tolerance:
                    vert_violations += 1
                    violation_mag = (c[j] - c[j+1]) - max_diff
                    vert_sum += violation_mag
                    vertical_returns.append(violation_mag / max(c[j+1] + max_diff, 1e-10))
            
            for j in range(1, n - 1):
                width = k[j+1] - k[j-1]
                if width <= 1e-10:
                    continue
                lambd = (k[j+1] - k[j]) / width
                c_convex = lambd * c[j-1] + (1.0 - lambd) * c[j+1]
                if c[j] > c_convex + tolerance:
                    butt_violations += 1
                    violation_mag = c[j] - c_convex
                    butt_sum += violation_mag
                    butterfly_returns.append(violation_mag / max(c_convex, 1e-10))
                    
        return vert_violations, vert_sum, vertical_returns, butt_violations, butt_sum, butterfly_returns

    def _check_calendar_arbitrage(self, call_prices, strikes, maturities, tolerance):
        """
        Checks calendar spread violations between consecutive maturity slices.
        """
        cal_violations = 0
        cal_sum = 0.0
        calendar_returns = []
        
        unique_mats = np.sort(np.unique(maturities))
        if len(unique_mats) < 2:
            return cal_violations, cal_sum, calendar_returns
            
        for i in range(len(unique_mats) - 1):
            mat_a = unique_mats[i]
            mat_b = unique_mats[i+1]
            
            mask_a = (maturities == mat_a)
            mask_b = (maturities == mat_b)
            
            calls_a = call_prices[mask_a]
            strikes_a = strikes[mask_a]
            
            calls_b = call_prices[mask_b]
            strikes_b = strikes[mask_b]
            
            rounded_a = np.round(strikes_a, 2)
            rounded_b = np.round(strikes_b, 2)
            
            dict_a = {rounded_a[j]: calls_a[j] for j in range(len(rounded_a))}
            
            for j in range(len(rounded_b)):
                strk_b = rounded_b[j]
                if strk_b in dict_a:
                    call_a = dict_a[strk_b]
                    call_b = calls_b[j]
                    
                    if call_b < call_a - tolerance:
                        cal_violations += 1
                        violation_mag = call_a - call_b
                        cal_sum += violation_mag
                        calendar_returns.append(violation_mag / max(call_b, 1e-10))
                        
        return cal_violations, cal_sum, calendar_returns

    def measure_all_raw_arbitrage(self, tolerance=1e-8):
        """
        Measures static arbitrage in the raw data for all days.
        """
        records = []
        for i in range(len(self.cleaned_data)):
            summary = self.measure_raw_arbitrage(i, tolerance=tolerance)
            summary['day_index'] = i
            records.append(summary)
        return pd.DataFrame(records)

#%%
if __name__ == '__main__':
    #%% USING DJX DATA **TRADED** (sparse data)
    ###################################################################################################
    with open("/Users/macbook/Documents/global_O_Research/O_Research/data/DJX_data/DJX_lists_traded_training.pkl", "rb") as f:
        uniqueDates = pickle.load(f)
        tau = pickle.load(f)
        moneyness = pickle.load(f)
        iv = pickle.load(f)
        S = pickle.load(f)
        rfRate = pickle.load(f)
        dividendRate = pickle.load(f)
    
    logMoneyness = [np.log(m) for m in moneyness]
    sqrtTau = [np.sqrt(t) for t in tau]
    
    # Implied variance is iv^2
    totVar_data = [t*(v**2) for t,v in zip(tau,iv)]
    
    with open("/Users/macbook/Documents/global_O_Research/O_Research/data/DJX_data/DJX_lists_traded_testing.pkl", "rb") as f:
        uniqueDates_test = pickle.load(f)
        tau_test = pickle.load(f)
        moneyness_test = pickle.load(f)
        iv_test = pickle.load(f)
        S_test = pickle.load(f)
        rfRate_test = pickle.load(f)
        dividendRate_test = pickle.load(f)
    
    logMoneyness_test = [np.log(m) for m in moneyness_test]
    sqrtTau_test = [np.sqrt(t) for t in tau_test]
    
    totVar_data_test = [t*(v**2) for t,v in zip(tau_test, iv_test)]
    
    #%% Estimate the First few FPCs
    fpca = FPCA_penalized_totVar(logMoneyness, tau, totVar_data, nb_spline_moneyness = 30, nb_spline_tau = 36, order_moneyness = 4, order_tau = 4, range_moneyness=[-.15,.15])
    alpha1, B1 = fpca.first_FPC_fit(maxit=20, omega_m=0.02**2, omega_m2= 0.05**2, omega_t=0.1**2, monotone=False)
    fpca.plot_eigen_functions(B1, num_points=50, figAngle=-70)
    
    alpha2, B2 = fpca.subsequent_FPC_fit(maxit=30, omega_m=0.025**2, omega_m2= 0.0**2, omega_t=0.025**2)
    fpca.plot_eigen_functions(B2, num_points=50, figAngle=-70)
    
    alpha3, B3 = fpca.subsequent_FPC_fit(maxit=30, omega_m=0.2**2, omega_m2= 0.1**2, omega_t=0.025**2)
    fpca.plot_eigen_functions(B3, num_points=50, figAngle=-70)
    
    print(fpca.compute_explained_variance())
    
    with open("/Users/macbook/Documents/global_O_Research/O_Research/data/DJX_data/DJX_traded_FPCA_totVariance.pkl", "wb") as f:
        pickle.dump(fpca.scoreMat, f)
        pickle.dump(fpca.BList, f)
    
    #%% Load Basis representation fit
    fpca = FPCA_penalized_totVar(logMoneyness, tau, totVar_data, nb_spline_moneyness = 30, nb_spline_tau = 36, order_moneyness = 4, order_tau = 4, range_moneyness=[-.15,.15])
    with open("/Users/macbook/Documents/global_O_Research/O_Research/data/DJX_data/DJX_traded_FPCA_totVariance.pkl", "rb") as f:
        fpca.scoreMat = pickle.load(f)
        fpca.BList = pickle.load(f)
        
    #%% Testing Statistics
    val_mses_test = []
    val_scores_test = []
    
    for i in range(len(uniqueDates_test)):
        coords_val = np.column_stack([logMoneyness_test[i], tau_test[i]])
        scores_val = fpca.project_to_scores(coords_val, totVar_data_test[i], fpca.BList)
        y_hat_val = fpca.reconstruct_surface(coords_val, fpca.BList, scores_val)
        val_mses_test.append(np.mean((totVar_data_test[i] - y_hat_val)**2))
        val_scores_test.append(scores_val)
        
    print(np.mean(np.sqrt(val_mses_test)))

    #%% Training Data Statistics
    val_mses = []
    val_scores = []
    
    for i in range(len(uniqueDates)):
        coords_val = np.column_stack([logMoneyness[i], tau[i]])
        scores_val = fpca.project_to_scores(coords_val, totVar_data[i], fpca.BList)
        y_hat_val = fpca.reconstruct_surface(coords_val, fpca.BList, scores_val)
        val_mses.append(np.mean((totVar_data[i] - y_hat_val)**2))
        val_scores.append(scores_val)
            
    print(np.mean(np.sqrt(val_mses)))
    
    #%% Measure Static Arbitrage
    nbDays = len(fpca.cleaned_data)
    nbCalendar = np.zeros(nbDays)
    nbButterfly = np.zeros(nbDays)
    for i in range(nbDays):
        day_scores = fpca.scoreMat[i, :]
        calendar_metrics, butterfly_metrics = fpca.compute_arbitrage_metrics(day_scores)
        nbCalendar[i] = np.sum(calendar_metrics < 0)
        nbButterfly[i] = np.sum(butterfly_metrics < 0)
    
    uniqueDates_dates = pd.to_datetime(uniqueDates)
    plt.plot(uniqueDates_dates, nbButterfly/(50*50), label="But. arb.")
    plt.plot(uniqueDates_dates, nbCalendar/(50*50), label="Cal. arb.")
    plt.legend() 
    
    # Plot violations for the first day
    fpca.plot_arbitrage_violations(0)
    
    
    
    #%% USING SPX DATA (dense data)
    ###################################################################################################
    with open("/Users/macbook/Documents/global_O_Research/O_Research/data/SPX_data/SPX_lists_training.pkl", "rb") as f:
        uniqueDates = pickle.load(f)
        tau = pickle.load(f)
        moneyness = pickle.load(f)
        iv = pickle.load(f)
        S = pickle.load(f)
        rfRate = pickle.load(f)
        dividendRate = pickle.load(f)
    
    logMoneyness = [np.log(m) for m in moneyness]
    sqrtTau = [np.sqrt(t) for t in tau]
    
    # Implied variance is iv^2
    totVar_data = [(v**2) for v in iv]
    
    with open("/Users/macbook/Documents/global_O_Research/O_Research/data/SPX_data/SPX_lists_testing.pkl", "rb") as f:
        uniqueDates_test = pickle.load(f)
        tau_test = pickle.load(f)
        moneyness_test = pickle.load(f)
        iv_test = pickle.load(f)
        S_test = pickle.load(f)
        rfRate_test = pickle.load(f)
        dividendRate_test = pickle.load(f)
    
    logMoneyness_test = [np.log(m) for m in moneyness_test]
    sqrtTau_test = [np.sqrt(t) for t in tau_test]
    
    totVar_data_test = [(v**2) for v in iv_test]
    
    #%% Estimate the First few FPCs
    fpca = FPCA_penalized_totVar(logMoneyness, tau, totVar_data, nb_spline_moneyness = 30, nb_spline_tau = 36, order_moneyness = 4, order_tau = 4, range_moneyness=[-.15,.15])
    alpha1, B1 = fpca.first_FPC_fit(maxit=20, omega_m=0.05**2, omega_m2= 0.05**2, omega_t=0.05**2, monotone=False)
    fpca.plot_eigen_functions(B1, num_points=50, figAngle=-70)
    
    alpha2, B2 = fpca.subsequent_FPC_fit(maxit=30, omega_m=0.025**2, omega_m2= 0.0**2, omega_t=0.025**2)
    fpca.plot_eigen_functions(B2, num_points=50, figAngle=-70)
    
    alpha3, B3 = fpca.subsequent_FPC_fit(maxit=30, omega_m=0.2**2, omega_m2= 0.1**2, omega_t=0.025**2)
    fpca.plot_eigen_functions(B3, num_points=50, figAngle=-70)
    
    print(fpca.compute_explained_variance())
    
    with open("/Users/macbook/Documents/global_O_Research/O_Research/data/SPX_data/SPX_FPCA_totVariance.pkl", "wb") as f:
        pickle.dump(fpca.scoreMat, f)
        pickle.dump(fpca.BList, f)
    
    #%% Load Basis representation fit
    fpca = FPCA_penalized_totVar(logMoneyness, tau, totVar_data, nb_spline_moneyness = 30, nb_spline_tau = 36, order_moneyness = 4, order_tau = 4, range_moneyness=[-.15,.15])
    with open("/Users/macbook/Documents/global_O_Research/O_Research/data/SPX_data/SPX_FPCA_totVariance.pkl", "rb") as f:
        fpca.scoreMat = pickle.load(f)
        fpca.BList = pickle.load(f)
        
    #%% Testing Statistics
    val_mses_test = []
    val_mses_test_ivs = []
    
    for i in range(len(uniqueDates_test)):
        coords_val = np.column_stack([logMoneyness_test[i], tau_test[i]])
        scores_val = fpca.project_to_scores(coords_val, totVar_data_test[i], fpca.BList)
        y_hat_val = fpca.reconstruct_surface(coords_val, fpca.BList, scores_val)
        val_mses_test.append(np.mean((totVar_data_test[i] - y_hat_val)**2))
        y_hat_val[y_hat_val<0] = 0
        val_mses_test_ivs.append(np.mean((np.sqrt(totVar_data_test[i]) - np.sqrt(y_hat_val))**2))
        
    print(np.mean(np.sqrt(val_mses_test)))
    print(np.mean(np.sqrt(val_mses_test_ivs)))

    #%% Training Data Statistics
    val_mses = []
    val_mses_ivs = []
    
    for i in range(len(uniqueDates)):
        coords_val = np.column_stack([logMoneyness[i], tau[i]])
        scores_val = fpca.project_to_scores(coords_val, totVar_data[i], fpca.BList)
        y_hat_val = fpca.reconstruct_surface(coords_val, fpca.BList, scores_val)
        val_mses.append(np.mean((totVar_data[i] - y_hat_val)**2))
        y_hat_val[y_hat_val<0] = 0
        val_mses_ivs.append(np.mean((np.sqrt(totVar_data[i]) - np.sqrt(y_hat_val))**2))
            
    print(np.mean(np.sqrt(val_mses)))
    print(np.mean(np.sqrt(val_mses_ivs)))
    
    #%% Measure Static Arbitrage
    nbDays = len(fpca.cleaned_data)
    nbCalendar = np.zeros(nbDays)
    nbButterfly = np.zeros(nbDays)
    for i in range(nbDays):
        day_scores = fpca.scoreMat[i, :]
        calendar_metrics, butterfly_metrics = fpca.compute_arbitrage_metrics(day_scores, m_grid= np.linspace(-.15, .15, 50), tau_grid= np.linspace(.001, 1, 50))
        nbCalendar[i] = np.sum(calendar_metrics < 0)
        nbButterfly[i] = np.sum(butterfly_metrics < 0)
    
    uniqueDates_dates = pd.to_datetime(uniqueDates)
    plt.plot(uniqueDates_dates, nbButterfly/(50*50), label="But. arb.")
    plt.plot(uniqueDates_dates, nbCalendar/(50*50), label="Cal. arb.")
    plt.legend() 
    
    # Plot violations for the first day
    fpca.plot_arbitrage_violations(0)
# %%
