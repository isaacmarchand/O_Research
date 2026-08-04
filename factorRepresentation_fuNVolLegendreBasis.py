#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pickle
from scipy.special import eval_legendre as Legendre
from scipy.linalg import fractional_matrix_power


class FuNVolLegendreFPCA:
    """
    FuNVol Implied Volatility Surface (IVS) Dimensionality Reduction Model (Choudhary et al., 2024).
    
    This approach:
    1. Projects irregular daily IVS data onto a set of 2D Legendre polynomial basis functions.
    2. Performs Functional Principal Component Analysis (FPCA) on the fitted Legendre coefficient surfaces 
       using a continuous L2 Gram integration weight matrix over [-1, 1] x [-1, 1].
    3. Reconstructs surfaces using the top K FPC eigenfunctions and daily scores.
    """

    def __init__(self, order=4, range_moneyness=[-0.15, 0.15], range_tau=[0, 1], S=None, r=None, q=None):
        """
        Initialize FuNVol Legendre Basis functions and continuous integration weight matrix.
        
        Parameters:
        - order: Max degree of Legendre polynomial tensor products (default: 4).
        - range_moneyness: Tuple/list [min_m, max_m] for normalization to [-1, 1].
        - range_tau: Tuple/list [min_tau, max_tau] for normalization to [-1, 1].
        - S: Optional list/array of spot prices per day.
        - r: Optional list/array of risk-free rates per day.
        - q: Optional list/array of dividend yields per day.
        """
        self.order_val = order
        self.range_moneyness = range_moneyness
        self.range_tau = range_tau
        self.S = S
        self.r = r
        self.q = q

        # Initialize basis functions and order pairings
        self._init_legendre_basis()

        # Compute continuous Gram integration weight matrix W over [-1, 1] x [-1, 1]
        self.W = self._init_weight_matrix()

        # Placeholders for FPCA parameters fitted from data
        self.a_mean = None
        self.kappa = None
        self.b_transform = None
        self.psi_eigenfunctions = None

    def _init_legendre_basis(self):
        """
        Constructs scaled 2D Legendre polynomial basis functions phi_k(x, y).
        """
        self.phi = []
        self.order_pairs = []

        for i in range(1, self.order_val + 1):
            for j in range(i + 1):
                # Scaled 2D Legendre polynomial tensor product
                temp = lambda x, y, j=j, i=i: (
                    Legendre(j, x) / np.sqrt(2 / (2 * j + 1)) *
                    Legendre(i - j, y) / np.sqrt(2 / (2 * (i - j) + 1))
                )
                self.phi.append(temp)
                self.order_pairs.append([j, i - j])

        self.num_basis = len(self.phi)
        self.num_features = 1 + self.num_basis  # 1 for intercept a0 + 14 basis functions = 15 features for order 4

    def _init_weight_matrix(self, num_pts=101):
        """
        Computes the Gram weight matrix W for continuous L2 inner products over [-1, 1] x [-1, 1].
        """
        grid_x = np.linspace(-1, 1, num_pts)
        grid_y = np.linspace(-1, 1, num_pts)
        dx = grid_x[1] - grid_x[0]
        dy = grid_y[1] - grid_y[0]
        X, Y = np.meshgrid(grid_x, grid_y)

        W = np.zeros((self.num_features, self.num_features))

        # Intercept weight (integral of 1 * 1 over [-1,1]^2 is 4)
        W[0, 0] = np.sum(np.ones_like(X) * dx * dy)

        for i in range(self.num_basis):
            fi = self.phi[i](X, Y)
            W[i + 1, 0] = np.sum(fi * dx * dy)
            W[0, i + 1] = W[i + 1, 0]

            for j in range(i + 1, self.num_basis):
                fj = self.phi[j](X, Y)
                W[i + 1, j + 1] = np.sum(fi * fj * dx * dy)
                W[j + 1, i + 1] = W[i + 1, j + 1]

            W[i + 1, i + 1] = np.sum(fi * fi * dx * dy)

        return W

    def _normalize_coords(self, m, tau):
        """
        Normalizes (moneyness, tau) coordinates to [-1, 1] x [-1, 1].
        """
        m_arr = np.asarray(m)
        tau_arr = np.asarray(tau)

        x_norm = 2.0 * (m_arr - self.range_moneyness[0]) / (self.range_moneyness[1] - self.range_moneyness[0]) - 1.0
        y_norm = 2.0 * (tau_arr - self.range_tau[0]) / (self.range_tau[1] - self.range_tau[0]) - 1.0

        return x_norm, y_norm

    def _clean_data_for_day(self, m_i, tau_i, iv_i):
        """
        Cleans NaNs and flattens input vectors for a single day.
        """
        m_flat = np.asarray(m_i).flatten()
        tau_flat = np.asarray(tau_i).flatten()
        iv_flat = np.asarray(iv_i).flatten()

        valid_mask = ~np.isnan(iv_flat) & ~np.isnan(m_flat) & ~np.isnan(tau_flat)
        return m_flat[valid_mask], tau_flat[valid_mask], iv_flat[valid_mask]

    def _evaluate_legendre_matrix(self, x_norm, y_norm):
        """
        Evaluates the full feature matrix (intercept + basis functions) on normalized coordinates.
        Returns matrix of shape (N_obs, num_features).
        """
        if len(x_norm) == 0:
            return np.zeros((0, self.num_features))

        cols = [np.ones_like(x_norm)]
        for func in self.phi:
            cols.append(func(x_norm, y_norm))

        return np.column_stack(cols)

    def fit_legendre_coefficients_single_day(self, m_i, tau_i, iv_i):
        """
        Projects a single day's IVS observations onto Legendre basis functions via OLS regression.
        
        Returns:
        - a_coeff: 1D array of length num_features containing [a0, a1, ..., a_K].
        """
        m_clean, tau_clean, iv_clean = self._clean_data_for_day(m_i, tau_i, iv_i)
        if len(iv_clean) < self.num_features:
            return np.full(self.num_features, np.nan)

        x_norm, y_norm = self._normalize_coords(m_clean, tau_clean)
        Phi_mat = self._evaluate_legendre_matrix(x_norm, y_norm)

        a_coeff, _, _, _ = np.linalg.lstsq(Phi_mat, iv_clean, rcond=None)
        return a_coeff

    def fit_legendre_coefficients(self, moneyness, tau, iv):
        """
        Fits Legendre basis coefficients across all daily surfaces.
        
        Returns:
        - a_matrix: 2D array of shape (N_days, num_features).
        """
        a_list = []
        for i in range(len(iv)):
            a_i = self.fit_legendre_coefficients_single_day(moneyness[i], tau[i], iv[i])
            a_list.append(a_i)

        return np.array(a_list)

    def fit_fpca(self, a_matrix):
        """
        Performs FPCA on the fitted Legendre coefficient matrix.
        
        Parameters:
        - a_matrix: 2D array of shape (N_days, num_features).
        """
        valid_mask = ~np.isnan(a_matrix).any(axis=1)
        a_clean = a_matrix[valid_mask]

        if len(a_clean) == 0:
            raise ValueError("No valid coefficient rows to fit FPCA.")

        # Compute mean coefficient vector
        self.a_mean = np.mean(a_clean, axis=0)

        # Center coefficients
        c = a_clean - self.a_mean

        # Sample covariance matrix of coefficients
        A = (c.T @ c)

        # Compute continuous L2 inner product transformation matrix
        sqrt_W = fractional_matrix_power(self.W, 0.5)
        B = (sqrt_W @ A @ sqrt_W) / len(a_clean)

        # Eigen-decomposition
        eigenvals, u = np.linalg.eig(B)

        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvals)[::-1]
        self.kappa = np.real(eigenvals[idx])
        u_sorted = np.real(u[:, idx])

        # FPC expansion transformation matrix
        inv_sqrt_W = np.linalg.inv(sqrt_W)
        self.b_transform = inv_sqrt_W @ u_sorted

        # Define FPC eigenfunctions psi_k(x, y)
        self.psi_eigenfunctions = []
        for k in range(len(self.kappa)):
            b_k = self.b_transform[:, k]
            psi_k = lambda x, y, b_k=b_k: b_k[0] + np.sum(
                [b_k[j + 1] * self.phi[j](x, y) for j in range(self.num_basis)], axis=0
            )
            self.psi_eigenfunctions.append(psi_k)

        # Define mean surface evaluator
        self.mean_surface = lambda x, y: self.a_mean[0] + np.sum(
            [self.a_mean[j + 1] * self.phi[j](x, y) for j in range(self.num_basis)], axis=0
        )

    def compute_scores_single_day(self, m_i, tau_i, iv_i, K=5):
        """
        Projects a single day's IVS observations onto the first K FPC eigenfunctions.
        
        Parameters:
        - m_i, tau_i, iv_i: Daily observation vectors.
        - K: Number of FPC components to use.
        
        Returns:
        - scores_i: 1D array of length K.
        """
        if self.psi_eigenfunctions is None:
            raise ValueError("FPCA model has not been fitted. Call fit_fpca() first.")

        m_clean, tau_clean, iv_clean = self._clean_data_for_day(m_i, tau_i, iv_i)
        if len(iv_clean) == 0:
            return np.full(K, np.nan)

        x_norm, y_norm = self._normalize_coords(m_clean, tau_clean)
        mean_iv = self.mean_surface(x_norm, y_norm)
        zero_mean_iv = iv_clean - mean_iv

        # Build FPC evaluation design matrix Psi of shape (N_obs, K)
        Psi_mat = np.zeros((len(x_norm), K))
        for k in range(K):
            Psi_mat[:, k] = self.psi_eigenfunctions[k](x_norm, y_norm)

        scores_i, _, _, _ = np.linalg.lstsq(Psi_mat, zero_mean_iv, rcond=None)
        return scores_i

    def compute_scores(self, moneyness, tau, iv, K=5):
        """
        Computes FPC scores for all daily surfaces.
        """
        scores_list = []
        for i in range(len(iv)):
            s_i = self.compute_scores_single_day(moneyness[i], tau[i], iv[i], K=K)
            scores_list.append(s_i)
        return np.array(scores_list)

    def predict_iv(self, m, tau, scores, K=None):
        """
        Predicts implied volatility for given coordinates using fitted FPC scores.
        
        Parameters:
        - m, tau: Moneyness and time-to-expiry arrays.
        - scores: 1D array of length K containing FPC scores for the day.
        - K: Number of FPC components (defaults to len(scores)).
        
        Returns:
        - iv_pred: Array of predicted implied volatilities.
        """
        if K is None:
            K = len(scores)

        x_norm, y_norm = self._normalize_coords(m, tau)
        iv_pred = np.copy(self.mean_surface(x_norm, y_norm))

        for k in range(K):
            iv_pred += scores[k] * self.psi_eigenfunctions[k](x_norm, y_norm)

        return iv_pred

    def reconstruct_surface(self, scores, K=None, m_bounds=None, tau_bounds=None, num_points=30):
        """
        Generates a 2D grid of predicted implied volatility values from FPC scores.
        """
        if m_bounds is None:
            m_bounds = self.range_moneyness
        if tau_bounds is None:
            tau_bounds = self.range_tau
            
        m_vals = np.linspace(m_bounds[0], m_bounds[1], num_points)
        tau_vals = np.linspace(tau_bounds[0], tau_bounds[1], num_points)
        M, Tau = np.meshgrid(m_vals, tau_vals)

        Z = self.predict_iv(M, Tau, scores, K=K)
        return M, Tau, Z

    def plot_reconstruction(self, scores=None, m_i=None, tau_i=None, iv_i=None, index=None, scores_matrix=None, moneyness=None, tau=None, iv=None, K=None, num_points=30, figAngle=-70, day_label=""):
        """
        Plots the reconstructed IVS for a given day and overlaps the actual observations.
        
        Parameters:
        - scores: 1D array of FPC scores for the day. (Optional if index, scores_matrix are provided)
        - m_i: 1D array of moneyness observations. (Optional if index, moneyness are provided)
        - tau_i: 1D array of time to expiry observations. (Optional if index, tau are provided)
        - iv_i: 1D array of implied volatility observations. (Optional if index, iv are provided)
        - index: Day index to select from scores_matrix, moneyness, tau, iv datasets.
        - scores_matrix: 2D array of daily FPC scores.
        - moneyness: List of daily moneyness vectors.
        - tau: List of daily time to expiry vectors.
        - iv: List of daily implied volatility vectors.
        - K: Number of FPC components to use in reconstruction.
        - num_points: Number of grid points along each dimension.
        - figAngle: Azimuth viewing angle.
        - day_label: Label string for title.
        """
        if index is not None:
            if scores_matrix is not None:
                scores = scores_matrix[index]
            if moneyness is not None and tau is not None and iv is not None:
                m_i = moneyness[index]
                tau_i = tau[index]
                iv_i = iv[index]
            if not day_label:
                day_label = f" (Day {index})"
                
        if scores is None or m_i is None or tau_i is None or iv_i is None:
            raise ValueError("Must provide either (scores, m_i, tau_i, iv_i) or (index, scores_matrix, moneyness, tau, iv).")
            
        m_clean, tau_clean, iv_clean = self._clean_data_for_day(m_i, tau_i, iv_i)
        
        M, Tau, Z = self.reconstruct_surface(scores, K=K, m_bounds=self.range_moneyness, tau_bounds=self.range_tau, num_points=num_points)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot reconstructed surface
        ax.plot_surface(M, Tau, Z, cmap='viridis', edgecolor='none', alpha=0.5)
        
        # Plot actual observations
        ax.scatter(m_clean, tau_clean, iv_clean, color='red', s=10, label='Actual Observations')
        
        ax.set_title(f"Reconstructed IVS vs Actual Observations{day_label}")
        ax.set_xlabel("Moneyness (m)")
        ax.set_ylabel("Time to Expiry (tau)")
        ax.set_zlabel("Implied Volatility")
        
        ax.view_init(elev=15, azim=figAngle)
        plt.legend()
        plt.show()

    def plot_eigen_functions(self, K=5, num_points=30, figAngle=-70):
        """
        Plots the top K Functional Principal Component (FPC) eigenfunctions psi_k(m, tau).
        
        Parameters:
        - K: Number of top FPC eigenfunctions to plot.
        - num_points: Number of grid points along each dimension.
        - figAngle: Azimuth viewing angle for 3D subplots.
        """
        if self.psi_eigenfunctions is None:
            raise ValueError("FPCA model has not been fitted. Call fit_fpca() first.")
            
        K = min(K, len(self.psi_eigenfunctions))
        
        m_vals = np.linspace(self.range_moneyness[0], self.range_moneyness[1], num_points)
        tau_vals = np.linspace(self.range_tau[0], self.range_tau[1], num_points)
        M, Tau = np.meshgrid(m_vals, tau_vals)
        
        x_norm, y_norm = self._normalize_coords(M, Tau)
        
        n_cols = min(K, 3)
        n_rows = int(np.ceil(K / n_cols))
        
        fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
        fig.suptitle(f"Top {K} FuNVol Functional Principal Component (FPC) Eigenfunctions", fontsize=16)
        
        tot_kappa = np.sum(self.kappa) if self.kappa is not None and np.sum(self.kappa) > 0 else 1.0
        
        for k in range(K):
            psi_k_val = self.psi_eigenfunctions[k](x_norm, y_norm)
            var_explained_pct = (self.kappa[k] / tot_kappa) * 100 if self.kappa is not None else 0.0
            
            ax = fig.add_subplot(n_rows, n_cols, k + 1, projection='3d')
            surf = ax.plot_surface(M, Tau, psi_k_val, cmap='viridis', edgecolor='none', alpha=0.85)
            
            ax.set_title(f"FPC {k + 1} ({var_explained_pct:.2f}% Variance)", fontsize=12)
            ax.set_xlabel("Moneyness (m)")
            ax.set_ylabel("Time to Expiry (tau)")
            ax.set_zlabel("FPC Value")
            ax.view_init(elev=15, azim=figAngle)
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
            
        plt.tight_layout()
        plt.show()

    def _compute_day_ss_res_tot(self, m_clean, tau_clean, iv_clean, scores_i, K=None):
        """
        Computes residual sum of squares (SS_res) and total sum of squares (SS_tot) for one day.
        """
        if len(iv_clean) == 0 or np.isnan(scores_i).any():
            return np.nan, np.nan

        iv_pred = self.predict_iv(m_clean, tau_clean, scores_i, K=K)
        ss_res = np.sum((iv_clean - iv_pred) ** 2)
        ss_tot = np.sum((iv_clean - np.mean(iv_clean)) ** 2)
        return ss_res, ss_tot

    def compute_variance_explained(self, moneyness, tau, iv, scores, K=None):
        """
        Computes R^2 (proportion of variance explained) for each day and overall across all days.
        """
        daily_r2 = []
        ss_res_list = []
        ss_tot_list = []

        for i in range(len(iv)):
            m_clean, tau_clean, iv_clean = self._clean_data_for_day(moneyness[i], tau[i], iv[i])
            scores_i = scores[i]

            ss_res, ss_tot = self._compute_day_ss_res_tot(m_clean, tau_clean, iv_clean, scores_i, K=K)

            if np.isnan(ss_res) or np.isnan(ss_tot):
                daily_r2.append(np.nan)
                continue

            if ss_tot == 0:
                r2 = 1.0 if ss_res == 0 else 0.0
            else:
                r2 = 1.0 - (ss_res / ss_tot)

            daily_r2.append(r2)
            ss_res_list.append(ss_res)
            ss_tot_list.append(ss_tot)

        valid_ss_res = np.sum([r for r in ss_res_list if not np.isnan(r)])
        valid_ss_tot = np.sum([t for t in ss_tot_list if not np.isnan(t)])

        total_r2 = 1.0 - (valid_ss_res / valid_ss_tot) if valid_ss_tot > 0 else 0.0
        return daily_r2, total_r2

    def compute_arbitrage_metrics(self, scores, K=None, m_grid=None, tau_grid=None):
        """
        Computes calendar and butterfly arbitrage metrics on a grid according to
        Gatheral & Jacquier (2014) and Chundary (2024) / FuNVol paper Section 4.
        
        Parameters:
        - scores: 1D array of FPC scores for the day.
        - K: Number of FPC components (defaults to len(scores)).
        - m_grid: Grid of log-moneyness. If None, uses 50 points in range_moneyness.
        - tau_grid: Grid of maturities. If None, uses 50 points in range_tau.
        
        Returns:
        - calendar_metrics: 2D array of shape (len(tau_grid), len(m_grid)) representing d_tau w
        - butterfly_metrics: 2D array of shape (len(tau_grid), len(m_grid)) representing g_tau(m)
        """
        if m_grid is None:
            m_grid = np.linspace(self.range_moneyness[0], self.range_moneyness[1], 50)
        if tau_grid is None:
            tau_min = max(1e-4, self.range_tau[0])
            tau_grid = np.linspace(tau_min, self.range_tau[1], 50)
            
        M, T = np.meshgrid(m_grid, tau_grid)
        
        iv = self.predict_iv(M, T, scores, K=K)
        w = (iv ** 2) * T
        
        calendar_metrics = self._compute_calendar_spread_metric(w, tau_grid, m_grid)
        butterfly_metrics = self._compute_butterfly_spread_metric(w, tau_grid, m_grid)
        
        return calendar_metrics, butterfly_metrics

    def _compute_calendar_spread_metric(self, w, tau_grid, m_grid):
        """
        Computes d_tau w using central differences (one-sided at boundaries).
        """
        calendar_metrics = np.zeros_like(w)
        for j in range(len(m_grid)):
            w_slice = w[:, j]
            calendar_metrics[1:-1, j] = (w_slice[2:] - w_slice[:-2]) / (tau_grid[2:] - tau_grid[:-2])
            calendar_metrics[0, j] = (w_slice[1] - w_slice[0]) / (tau_grid[1] - tau_grid[0])
            calendar_metrics[-1, j] = (w_slice[-1] - w_slice[-2]) / (tau_grid[-1] - tau_grid[-2])
        return calendar_metrics

    def _compute_butterfly_spread_metric(self, w, tau_grid, m_grid):
        """
        Computes g_tau(m) for each slice in tau_grid.
        g_tau(m) = (1 - m * w_m / (2 * w))^2 - (w_m^2 / 4) * (1 / w + 1 / 4) + w_mm / 2
        """
        butterfly_metrics = np.zeros_like(w)
        for i in range(len(tau_grid)):
            w_slice = w[i, :]
            w_safe = np.maximum(w_slice, 1e-10)
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
            w_mm[0] = w_mm[1]
            w_mm[-1] = w_mm[-2]
            
            term1 = (1 - m_slice * w_m / (2 * w_safe))**2
            term2 = (w_m**2 / 4) * (1 / w_safe + 0.25)
            term3 = w_mm / 2
            butterfly_metrics[i, :] = term1 - term2 + term3
            
        return butterfly_metrics

    def ivs_to_price_surface(self, coords, iv, day_index=0, S_val=None, r_val=None, q_val=None):
        """
        Computes Black-Scholes call option prices from implied volatility surface coordinates.
        
        Parameters:
        - coords: 2D array of shape (n, 2) where col 0 is log-moneyness and col 1 is tau.
        - iv: 1D array of implied volatilities.
        - day_index: Index of the day (used to extract S, r, q if set in self).
        - S_val, r_val, q_val: Optional explicit asset price, risk-free rate, and dividend rate.
        
        Returns:
        - call_prices: 1D array of Black-Scholes call prices.
        """
        from scipy.stats import norm
        
        log_m = np.asarray(coords[:, 0])
        tau = np.asarray(coords[:, 1])
        
        if S_val is None:
            S_val = self.S[day_index] if self.S is not None and day_index < len(self.S) else 1.0
        if r_val is None:
            r_val = self.r[day_index] if self.r is not None and day_index < len(self.r) else 0.0
        if q_val is None:
            q_val = self.q[day_index] if self.q is not None and day_index < len(self.q) else 0.0
            
        K = S_val * np.exp(log_m)
        tau_safe = np.maximum(tau, 1e-10)
        iv_safe = np.maximum(iv, 1e-10)
        
        d1 = (-log_m + (r_val - q_val + 0.5 * iv_safe**2) * tau_safe) / (iv_safe * np.sqrt(tau_safe))
        d2 = d1 - iv_safe * np.sqrt(tau_safe)
        
        call_prices = S_val * np.exp(-q_val * tau_safe) * norm.cdf(d1) - K * np.exp(-r_val * tau_safe) * norm.cdf(d2)
        call_prices = np.where(tau <= 0, np.maximum(S_val - K, 0), call_prices)
        
        return call_prices

    def compute_price_arbitrage_metrics(self, scores, K=None, m_grid=None, tau_grid=None, day_index=0, S_val=None, r_val=None, q_val=None):
        """
        Computes calendar, call spread, and butterfly spread arbitrage penalty matrices 
        on a grid using relative call prices (c = Call / S), according to Cont & Vuletic (2023).
        """
        if m_grid is None:
            m_grid = np.linspace(self.range_moneyness[0], self.range_moneyness[1], 50)
        if tau_grid is None:
            tau_min = max(1e-4, self.range_tau[0])
            tau_grid = np.linspace(tau_min, self.range_tau[1], 50)
            
        M, T = np.meshgrid(m_grid, tau_grid)
        coords = np.column_stack([M.flatten(), T.flatten()])
        
        iv_flat = self.predict_iv(M, T, scores, K=K).flatten()
        call_prices_flat = self.ivs_to_price_surface(coords, iv_flat, day_index=day_index, S_val=S_val, r_val=r_val, q_val=q_val)
        
        if S_val is None:
            S_val = self.S[day_index] if self.S is not None and day_index < len(self.S) else 1.0
            
        c_flat = call_prices_flat / S_val
        c = c_flat.reshape(M.shape)
        m_abs = np.exp(m_grid)
        
        P1 = np.zeros_like(c)
        P2 = np.zeros_like(c)
        P3 = np.zeros_like(c)
        
        for j in range(len(tau_grid) - 1):
            for i in range(len(m_grid)):
                val = tau_grid[j] * (c[j, i] - c[j+1, i]) / (tau_grid[j+1] - tau_grid[j])
                P1[j, i] = max(0, val)
                
        for j in range(len(tau_grid)):
            for i in range(len(m_grid) - 1):
                val = (c[j, i+1] - c[j, i]) / (m_abs[i+1] - m_abs[i])
                P2[j, i] = max(0, val)
                
        for j in range(len(tau_grid)):
            for i in range(1, len(m_grid) - 1):
                left_diff = (c[j, i] - c[j, i-1]) / (m_abs[i] - m_abs[i-1])
                right_diff = (c[j, i+1] - c[j, i]) / (m_abs[i+1] - m_abs[i])
                val = left_diff - right_diff
                P3[j, i] = max(0, val)
                
        return P1, P2, P3

    def plot_arbitrage_violations(self, scores, K=None, m_grid=None, tau_grid=None, num_points=50, figAngle=-70, day_label=""):
        """
        Plots where calendar and butterfly arbitrage violations occur on the reconstructed surface.
        """
        if m_grid is None:
            m_grid = np.linspace(self.range_moneyness[0], self.range_moneyness[1], num_points)
        if tau_grid is None:
            tau_min = max(1e-4, self.range_tau[0])
            tau_grid = np.linspace(tau_min, self.range_tau[1], num_points)
            
        calendar_metrics, butterfly_metrics = self.compute_arbitrage_metrics(scores, K=K, m_grid=m_grid, tau_grid=tau_grid)
        
        calendar_violations = np.minimum(0, calendar_metrics)
        butterfly_violations = np.minimum(0, butterfly_metrics)
        
        M, T = np.meshgrid(m_grid, tau_grid)
        
        fig = plt.figure(figsize=(16, 7))
        
        # Plot Calendar Spread violations
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(M, T, calendar_violations, cmap='RdYlGn', edgecolor='none', alpha=0.8)
        ax1.plot_surface(M, T, np.zeros_like(M), color='black', alpha=0.2)
        ax1.set_title(f"Calendar Spread Metric (d_tau w) {day_label}\n(Violation if < 0)")
        ax1.set_xlabel("Moneyness (m)")
        ax1.set_ylabel("Time to Expiry (tau)")
        ax1.view_init(elev=15, azim=figAngle)
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

        # Plot Butterfly violations
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(M, T, butterfly_violations, cmap='RdYlGn', edgecolor='none', alpha=0.8)
        ax2.plot_surface(M, T, np.zeros_like(M), color='black', alpha=0.2)
        ax2.set_title(f"Butterfly Metric (g(m)) {day_label}\n(Violation if < 0)")
        ax2.set_xlabel("Moneyness (m)")
        ax2.set_ylabel("Time to Expiry (tau)")
        ax2.view_init(elev=15, azim=figAngle)
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
        
        plt.tight_layout()
        plt.show()

    def measure_surface_arbitrage(self, scores, K=None, m_grid=None, tau_grid=None, tolerance=0.0):
        """
        Measures calendar and butterfly arbitrage violations on a continuous grid for a single surface.
        """
        calendar_metrics, butterfly_metrics = self.compute_arbitrage_metrics(scores, K=K, m_grid=m_grid, tau_grid=tau_grid)
        
        cal_mask = calendar_metrics < -tolerance
        cal_violations = int(np.sum(cal_mask))
        cal_sum = float(np.sum(-calendar_metrics[cal_mask]))
        
        butt_mask = butterfly_metrics < -tolerance
        butt_violations = int(np.sum(butt_mask))
        butt_sum = float(np.sum(-butterfly_metrics[butt_mask]))
        
        total_grid_points = calendar_metrics.size
        
        return {
            'calendar_violations': cal_violations,
            'calendar_violation_sum': cal_sum,
            'calendar_violation_pct': cal_violations / total_grid_points,
            'butterfly_violations': butt_violations,
            'butterfly_violation_sum': butt_sum,
            'butterfly_violation_pct': butt_violations / total_grid_points,
            'total_violations': cal_violations + butt_violations,
            'total_grid_points': total_grid_points
        }

    def measure_all_surface_arbitrage(self, scores_matrix, K=None, m_grid=None, tau_grid=None, tolerance=0.0):
        """
        Measures surface arbitrage across a sequence of daily FPC score vectors.
        """
        records = []
        for i, s_i in enumerate(scores_matrix):
            summary = self.measure_surface_arbitrage(s_i, K=K, m_grid=m_grid, tau_grid=tau_grid, tolerance=tolerance)
            summary['day_index'] = i
            records.append(summary)
        return pd.DataFrame(records)

    def measure_raw_arbitrage(self, m_i, tau_i, iv_i, day_index=0, S_val=None, r_val=None, q_val=None, tolerance=1e-8):
        """
        Measures static arbitrage in discrete option data for a single day.
        """
        m_clean, tau_clean, iv_clean = self._clean_data_for_day(m_i, tau_i, iv_i)
        if len(iv_clean) == 0:
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
        call_prices = self.ivs_to_price_surface(coords, iv_clean, day_index=day_index, S_val=S_val, r_val=r_val, q_val=q_val)
        
        if S_val is None:
            S_val = self.S[day_index] if self.S is not None and day_index < len(self.S) else 1.0
        if r_val is None:
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

    def measure_all_raw_arbitrage(self, moneyness, tau, iv, S=None, r=None, q=None, tolerance=1e-8):
        """
        Measures static arbitrage in raw discrete option data across all days.
        """
        records = []
        for i in range(len(iv)):
            S_val = S[i] if S is not None and i < len(S) else None
            r_val = r[i] if r is not None and i < len(r) else None
            q_val = q[i] if q is not None and i < len(q) else None
            
            summary = self.measure_raw_arbitrage(
                moneyness[i], tau[i], iv[i], day_index=i, S_val=S_val, r_val=r_val, q_val=q_val, tolerance=tolerance
            )
            summary['day_index'] = i
            records.append(summary)
        return pd.DataFrame(records)



#%%
if __name__ == "__main__":
    #%% USING SPX DATA (dense data)
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

    flattenIV = [v for vDay in iv for v in vDay]
    meanIV = np.mean(flattenIV)
    ivCentered = [v - meanIV for v in iv]

    ivLog = [np.log(v) for v in iv]

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

    flattenIV_test = [v for vDay in iv_test for v in vDay]
    meanIV_test = np.mean(flattenIV)
    ivCentered_test = [v - meanIV for v in iv_test]

    ivLog_test = [np.log(v) for v in iv_test]

    #%% Estimate FuNVol Legendre Basis & FPCA for Training Data
    # Min/Max bounds across training and testing data for domain normalization
    all_log_m = np.concatenate([np.concatenate(logMoneyness), np.concatenate(logMoneyness_test)])
    all_tau = np.concatenate([np.concatenate(tau), np.concatenate(tau_test)])
    range_m = [np.min(all_log_m), np.max(all_log_m)]
    range_t = [np.min(all_tau), np.max(all_tau)]

    funvol_model = FuNVolLegendreFPCA(order=4, range_moneyness=range_m, range_tau=range_t)

    # 1. Project training surfaces onto Legendre polynomials
    print("Projecting training surfaces onto 2D Legendre polynomial basis...")
    a_matrix_train = funvol_model.fit_legendre_coefficients(logMoneyness, tau, iv)

    # 2. Fit FPCA on training Legendre coefficients
    print("Fitting FPCA on Legendre coefficient surfaces...")
    funvol_model.fit_fpca(a_matrix_train)

    # Use K=3 components (or K=5) for evaluation
    K_comp = 3
    scores_train = funvol_model.compute_scores(logMoneyness, tau, iv, K=K_comp)
    daily_r2_train, total_r2_train = funvol_model.compute_variance_explained(logMoneyness, tau, iv, scores_train, K=K_comp)

    #%% Training Data Statistics
    val_mses = []
    for i in range(len(uniqueDates)):
        m_clean, tau_clean, iv_clean = funvol_model._clean_data_for_day(logMoneyness[i], tau[i], iv[i])
        if len(iv_clean) == 0:
            continue
        y_hat_val = funvol_model.predict_iv(m_clean, tau_clean, scores_train[i], K=K_comp)
        val_mses.append(np.mean((iv_clean - y_hat_val) ** 2))

    val_rmse = np.sqrt(val_mses)

    #%% Testing Data Statistics
    scores_test = funvol_model.compute_scores(logMoneyness_test, tau_test, iv_test, K=K_comp)
    daily_r2_test, total_r2_test = funvol_model.compute_variance_explained(logMoneyness_test, tau_test, iv_test, scores_test, K=K_comp)

    val_mses_test = []
    for i in range(len(uniqueDates_test)):
        m_clean, tau_clean, iv_clean = funvol_model._clean_data_for_day(logMoneyness_test[i], tau_test[i], iv_test[i])
        if len(iv_clean) == 0:
            continue
        y_hat_val = funvol_model.predict_iv(m_clean, tau_clean, scores_test[i], K=K_comp)
        val_mses_test.append(np.mean((iv_clean - y_hat_val) ** 2))

    val_rmse_test = np.sqrt(val_mses_test)

    #%% Display Goodness of Fit Summary
    print(f"\n=== FuNVol (Legendre FPCA, K={K_comp}) Model - Goodness of Fit Summary ===")
    print(f"Training Set  - Mean MSE: {np.mean(val_mses):.6e} | Mean RMSE: {np.mean(val_rmse):.6f} | Total R^2: {total_r2_train:.4%}")
    print(f"Testing Set   - Mean MSE: {np.mean(val_mses_test):.6e} | Mean RMSE: {np.mean(val_rmse_test):.6f} | Total R^2: {total_r2_test:.4%}")

    #%% Arbitrage Analysis (Gatheral & Jacquier / FuNVol Section 4)
    print(f"\n=== FuNVol (Legendre FPCA, K={K_comp}) Model - Arbitrage Analysis ===")
    
    funvol_model.S = S
    funvol_model.r = rfRate
    funvol_model.q = dividendRate
    
    # 1. Continuous Surface Arbitrage on fitted model surfaces
    arb_surface_train = funvol_model.measure_all_surface_arbitrage(scores_train, K=K_comp)
    arb_surface_test = funvol_model.measure_all_surface_arbitrage(scores_test, K=K_comp)
    
    print("Fitted Surface Arbitrage (Grid-based: d_tau w < 0 or g(m) < 0):")
    print(f"  [Train] Mean Calendar Violations: {arb_surface_train['calendar_violations'].mean():.2f} / {arb_surface_train['total_grid_points'].iloc[0]} ({arb_surface_train['calendar_violation_pct'].mean():.2%}) | Sum: {arb_surface_train['calendar_violation_sum'].mean():.4f}")
    print(f"  [Train] Mean Butterfly Violations: {arb_surface_train['butterfly_violations'].mean():.2f} / {arb_surface_train['total_grid_points'].iloc[0]} ({arb_surface_train['butterfly_violation_pct'].mean():.2%}) | Sum: {arb_surface_train['butterfly_violation_sum'].mean():.4f}")
    print(f"  [Test]  Mean Calendar Violations: {arb_surface_test['calendar_violations'].mean():.2f} / {arb_surface_test['total_grid_points'].iloc[0]} ({arb_surface_test['calendar_violation_pct'].mean():.2%}) | Sum: {arb_surface_test['calendar_violation_sum'].mean():.4f}")
    print(f"  [Test]  Mean Butterfly Violations: {arb_surface_test['butterfly_violations'].mean():.2f} / {arb_surface_test['total_grid_points'].iloc[0]} ({arb_surface_test['butterfly_violation_pct'].mean():.2%}) | Sum: {arb_surface_test['butterfly_violation_sum'].mean():.4f}")
    
    # 2. Raw Discrete Data Arbitrage
    arb_raw_train = funvol_model.measure_all_raw_arbitrage(logMoneyness, tau, iv, S=S, r=rfRate, q=dividendRate)
    print("\nRaw Data Discrete Price Arbitrage:")
    print(f"  [Train] Mean Vertical Violations: {arb_raw_train['vertical_violations'].mean():.2f} | Mean Butterfly: {arb_raw_train['butterfly_violations'].mean():.2f} | Mean Calendar: {arb_raw_train['calendar_violations'].mean():.2f}")
    print(f"  [Train] Mean Total Violations: {arb_raw_train['total_violations'].mean():.2f} out of {arb_raw_train['total_nb_observations'].mean():.1f} options/day")

    #%% Save fitted FuNVol scores and transformation matrix
    with open("/Users/macbook/Documents/global_O_Research/O_Research/data/SPX_data/SPX_FuNVol_scores.pkl", "wb") as f:
        pickle.dump(scores_train, f)
        pickle.dump(scores_test, f)
        pickle.dump(funvol_model.b_transform, f)
        pickle.dump(funvol_model.kappa, f)

