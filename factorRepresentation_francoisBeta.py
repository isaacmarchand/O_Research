#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pickle


class FrancoisBetaExtractor:
    """
    Extensible Implied Volatility Surface (IVS) model proposed by François et al. (2022).
    
    This class projects discrete implied volatility observations onto 5 financially meaningful
    basis functions (Long-term ATM level, Time to maturity slope, Moneyness slope, Smile attenuation, Smirk)
    to compute the corresponding 5 Beta parameters per observation date (surface).
    """

    def __init__(self, T_max=5.0, T_conv=0.25, S=None, r=None, q=None, range_moneyness=(-0.15, 0.10), range_tau=(1/365, 1.0)):
        """
        Initialize the 5 François basis functions with specified hyper-parameters.
        
        Parameters:
        - T_max: Maximum time to maturity scaling parameter (default: 5.0).
        - T_conv: Convergence horizon scaling parameter for short-term ATM slope (default: 0.25).
        - S: Optional list/array of spot prices per day.
        - r: Optional list/array of risk-free rates per day.
        - q: Optional list/array of dividend yields per day.
        - range_moneyness: Default moneyness bounds (min, max).
        - range_tau: Default tau bounds (min, max).
        """
        self.T_max = T_max
        self.T_conv = T_conv
        self.S = S
        self.r = r
        self.q = q
        self.range_moneyness = range_moneyness
        self.range_tau = range_tau

        # 5 Basis Functions defined in François et al. (2022)
        self.basis_functions = [
            lambda m, tau: np.ones_like(m),                                      # Basis 1: Long term ATM level
            lambda m, tau: np.exp(-np.sqrt(tau / self.T_conv)),                  # Basis 2: Time to maturity slope
            lambda m, tau: m * (m <= 0) + (np.exp(2 * m) - 1) / (np.exp(2 * m) + 1) * (m > 0), # Basis 3: Moneyness slope
            lambda m, tau: (1 - np.exp(-(m**2))) * np.log(self.T_max / tau),     # Basis 4: Smile attenuation
            lambda m, tau: (1 - np.exp((3 * m)**3)) * np.log(tau / self.T_max) * (m < 0)       # Basis 5: Smirk
        ]
        
        self.basis_names = [
            "Long term ATM level",
            "Time to maturity slope",
            "Moneyness slope",
            "Smile attenuation",
            "Smirk"
        ]

    def _clean_data_for_day(self, m_i, tau_i, iv_i, *extra_arrays):
        """
        Cleans NaNs, filters observations within self.range_moneyness and self.range_tau, and flattens arrays for a single day.
        Optionally filters extra_arrays using the same valid_mask.
        """
        m_flat = np.asarray(m_i).flatten()
        tau_flat = np.asarray(tau_i).flatten()
        iv_flat = np.asarray(iv_i).flatten()

        valid_mask = ~np.isnan(iv_flat) & ~np.isnan(m_flat) & ~np.isnan(tau_flat)
        if self.range_moneyness is not None:
            valid_mask &= (m_flat >= self.range_moneyness[0]) & (m_flat <= self.range_moneyness[1])
        if self.range_tau is not None:
            valid_mask &= (tau_flat >= self.range_tau[0]) & (tau_flat <= self.range_tau[1])

        cleaned_m = m_flat[valid_mask]
        cleaned_tau = tau_flat[valid_mask]
        cleaned_iv = iv_flat[valid_mask]

        if len(extra_arrays) > 0:
            cleaned_extras = []
            for arr in extra_arrays:
                if arr is not None and hasattr(arr, "__len__") and not isinstance(arr, (str, bytes)) and len(arr) == len(m_flat):
                    arr_flat = np.asarray(arr).flatten()
                    cleaned_extras.append(arr_flat[valid_mask])
                else:
                    cleaned_extras.append(arr)
            return (cleaned_m, cleaned_tau, cleaned_iv, *cleaned_extras)

        return cleaned_m, cleaned_tau, cleaned_iv

    def _evaluate_basis_matrix(self, m_clean, tau_clean):
        """
        Evaluates the 5 basis functions on discrete (moneyness, tau) observation points.
        Returns design matrix X of shape (N_obs, 5).
        """
        if len(m_clean) == 0:
            return np.zeros((0, len(self.basis_functions)))
        return np.column_stack([func(m_clean, tau_clean) for func in self.basis_functions])

    def _fit_beta_single_day(self, X, iv_clean):
        """
        Fits OLS regression (without intercept) of implied volatilities on the basis matrix X.
        Returns a 1D array of 5 beta coefficients.
        """
        if len(iv_clean) == 0 or X.shape[0] < len(self.basis_functions):
            return np.full(len(self.basis_functions), np.nan)

        beta_i, _, _, _ = np.linalg.lstsq(X, iv_clean, rcond=None)
        return beta_i

    def compute_beta_single_day(self, m_i, tau_i, iv_i):
        """
        Computes the 5 Beta parameters for a single day's IVS observation.
        
        Parameters:
        - m_i: 1D array-like of moneyness values for the day.
        - tau_i: 1D array-like of time to expiry values for the day.
        - iv_i: 1D array-like of implied volatility values for the day.
        
        Returns:
        - beta_i: 1D numpy array of 5 estimated beta coefficients.
        """
        m_clean, tau_clean, iv_clean = self._clean_data_for_day(m_i, tau_i, iv_i)
        X = self._evaluate_basis_matrix(m_clean, tau_clean)
        beta_i = self._fit_beta_single_day(X, iv_clean)
        return beta_i

    def compute_betas(self, moneyness, tau, iv, return_as_array=True):
        """
        Computes the Beta parameters for a sequence of daily implied volatility surfaces.
        
        Parameters:
        - moneyness: List of 1D array-like moneyness values for each day.
        - tau: List of 1D array-like time to expiry values for each day.
        - iv: List of 1D array-like implied volatility values for each day.
        - return_as_array: If True, returns a 2D numpy array of shape (N_days, 5).
                           If False, returns a list of 1D numpy arrays.
        
        Returns:
        - betas: 2D numpy array of shape (N_days, 5) or list of 1D numpy arrays.
        """
        if len(moneyness) != len(tau) or len(tau) != len(iv):
            raise ValueError("Lengths of moneyness, tau, and iv lists must all be equal.")

        betas_list = []
        for i in range(len(iv)):
            beta_i = self.compute_beta_single_day(moneyness[i], tau[i], iv[i])
            betas_list.append(beta_i)

        if return_as_array:
            return np.array(betas_list)
        return betas_list

    def predict_iv(self, beta, m, tau):
        """
        Predicts implied volatility at specified (m, tau) coordinates using given Beta parameters.
        
        Parameters:
        - beta: 1D array of 5 Beta coefficients.
        - m: Array-like of moneyness coordinates.
        - tau: Array-like of time to expiry coordinates.
        
        Returns:
        - iv_pred: Array of predicted implied volatilities matching shape of m/tau.
        """
        m_arr = np.asarray(m)
        tau_arr = np.asarray(tau)
        beta_arr = np.asarray(beta)

        if len(beta_arr) != len(self.basis_functions):
            raise ValueError(f"Beta must contain exactly {len(self.basis_functions)} coefficients.")

        iv_pred = np.zeros_like(m_arr, dtype=float)
        for coeff, func in zip(beta_arr, self.basis_functions):
            iv_pred += coeff * func(m_arr, tau_arr)

        return iv_pred

    def reconstruct_surface(self, beta, m_bounds=None, tau_bounds=None, num_points=30):
        """
        Generates a 2D grid of predicted implied volatility values from a set of Beta parameters.
        
        Parameters:
        - beta: 1D array of 5 Beta coefficients.
        - m_bounds: Tuple (min_m, max_m) for moneyness grid. If None, uses self.range_moneyness.
        - tau_bounds: Tuple (min_tau, max_tau) for time to expiry grid. If None, uses self.range_tau.
        - num_points: Number of grid points along each dimension.
        
        Returns:
        - M: 2D meshgrid of moneyness values.
        - Tau: 2D meshgrid of time to expiry values.
        - Z: 2D array of predicted implied volatilities on the meshgrid.
        """
        if m_bounds is None:
            m_bounds = self.range_moneyness
        if tau_bounds is None:
            tau_bounds = self.range_tau

        m_vals = np.linspace(m_bounds[0], m_bounds[1], num_points)
        tau_vals = np.linspace(tau_bounds[0], tau_bounds[1], num_points)
        M, Tau = np.meshgrid(m_vals, tau_vals)

        Z = self.predict_iv(beta, M, Tau)
        return M, Tau, Z

    def plot_reconstruction(self, beta=None, m_i=None, tau_i=None, iv_i=None, index=None, betas=None, moneyness=None, tau=None, iv=None, num_points=30, figAngle=-70, day_label=""):
        """
        Plots the reconstructed IVS for a given day and overlaps the actual observations.
        
        Parameters:
        - beta: 1D array of Beta coefficients for the day. (Optional if index, betas are provided)
        - m_i: 1D array of moneyness observations. (Optional if index, moneyness are provided)
        - tau_i: 1D array of time to expiry observations. (Optional if index, tau are provided)
        - iv_i: 1D array of implied volatility observations. (Optional if index, iv are provided)
        - index: Day index to select from betas, moneyness, tau, iv datasets.
        - betas: 2D array of daily betas.
        - moneyness: List of daily moneyness vectors.
        - tau: List of daily time to expiry vectors.
        - iv: List of daily implied volatility vectors.
        - num_points: Number of grid points along each dimension.
        - figAngle: Azimuth viewing angle.
        - day_label: Label string for title.
        """
        if index is not None:
            if betas is not None:
                beta = betas[index]
            if moneyness is not None and tau is not None and iv is not None:
                m_i = moneyness[index]
                tau_i = tau[index]
                iv_i = iv[index]
            if not day_label:
                day_label = f" (Day {index})"
                
        if beta is None or m_i is None or tau_i is None or iv_i is None:
            raise ValueError("Must provide either (beta, m_i, tau_i, iv_i) or (index, betas, moneyness, tau, iv).")
            
        m_clean, tau_clean, iv_clean = self._clean_data_for_day(m_i, tau_i, iv_i)
        
        M, Tau, Z = self.reconstruct_surface(beta, m_bounds=self.range_moneyness, tau_bounds=self.range_tau, num_points=num_points)
        
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
        ax.invert_xaxis()
        
        ax.view_init(elev=15, azim=figAngle)
        plt.legend()
        plt.show()

    def plot_basis_functions(self, num_points=30, figAngle=-70):
        """
        Plots the 5 deterministic basis functions defined by François et al. (2022).
        
        Parameters:
        - num_points: Number of grid points along each dimension.
        - figAngle: Azimuth viewing angle for 3D subplots.
        """
        m_vals = np.linspace(self.range_moneyness[0], self.range_moneyness[1], num_points)
        tau_vals = np.linspace(max(1e-4, self.range_tau[0]), self.range_tau[1], num_points)
        M, Tau = np.meshgrid(m_vals, tau_vals)
        
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle("François et al. (2022) Deterministic Basis Functions", fontsize=16)
        
        for idx, (func, name) in enumerate(zip(self.basis_functions, self.basis_names)):
            Z = func(M, Tau)
            
            ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
            surf = ax.plot_surface(M, Tau, Z, cmap='viridis', edgecolor='none', alpha=0.85)
            
            ax.set_title(f"Basis {idx + 1}: {name}", fontsize=12)
            ax.set_xlabel("Moneyness (m)")
            ax.set_ylabel("Time to Expiry (tau)")
            ax.set_zlabel("Basis Value")
            ax.view_init(elev=15, azim=figAngle)
            ax.invert_xaxis()
            
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
            
        plt.tight_layout()
        plt.show()

    def _compute_day_ss_res_tot(self, m_clean, tau_clean, iv_clean, beta_i):
        """
        Computes residual sum of squares (SS_res) and total sum of squares (SS_tot) for one day.
        """
        if len(iv_clean) == 0 or np.isnan(beta_i).any():
            return np.nan, np.nan

        iv_pred = self.predict_iv(beta_i, m_clean, tau_clean)
        ss_res = np.sum((iv_clean - iv_pred) ** 2)
        ss_tot = np.sum((iv_clean - np.mean(iv_clean)) ** 2)
        return ss_res, ss_tot

    def compute_variance_explained(self, moneyness, tau, iv, betas):
        """
        Computes R^2 (proportion of variance explained) for each day and overall across all days.
        
        Parameters:
        - moneyness: List of 1D array-like moneyness values.
        - tau: List of 1D array-like time to expiry values.
        - iv: List of 1D array-like implied volatility values.
        - betas: 2D numpy array of shape (N_days, 5) or list of 1D beta vectors.
        
        Returns:
        - daily_r2: List of R^2 values per observation date.
        - total_r2: Overall R^2 across all observations.
        """
        daily_r2 = []
        ss_res_list = []
        ss_tot_list = []

        for i in range(len(iv)):
            m_clean, tau_clean, iv_clean = self._clean_data_for_day(moneyness[i], tau[i], iv[i])
            beta_i = betas[i]

            ss_res, ss_tot = self._compute_day_ss_res_tot(m_clean, tau_clean, iv_clean, beta_i)

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

    def compute_arbitrage_metrics(self, beta, m_grid=None, tau_grid=None):
        """
        Computes calendar and butterfly arbitrage metrics on a grid according to
        Gatheral & Jacquier (2014) and Chundary (2024) / FuNVol paper Section 4.
        
        Parameters:
        - beta: 1D array of 5 Beta coefficients.
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
        
        iv = self.predict_iv(beta, M, T)
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
            
        if hasattr(S_val, "__len__") and not isinstance(S_val, (str, bytes)) and len(S_val) != len(log_m):
            S_val = float(np.mean(S_val))
        if hasattr(r_val, "__len__") and not isinstance(r_val, (str, bytes)) and len(r_val) != len(log_m):
            r_val = float(np.mean(r_val))
        if hasattr(q_val, "__len__") and not isinstance(q_val, (str, bytes)) and len(q_val) != len(log_m):
            q_val = float(np.mean(q_val))

        K = S_val * np.exp(log_m)
        tau_safe = np.maximum(tau, 1e-10)
        iv_safe = np.maximum(iv, 1e-10)
        
        d1 = (-log_m + (r_val - q_val + 0.5 * iv_safe**2) * tau_safe) / (iv_safe * np.sqrt(tau_safe))
        d2 = d1 - iv_safe * np.sqrt(tau_safe)
        
        call_prices = S_val * np.exp(-q_val * tau_safe) * norm.cdf(d1) - K * np.exp(-r_val * tau_safe) * norm.cdf(d2)
        call_prices = np.where(tau <= 0, np.maximum(S_val - K, 0), call_prices)
        
        return call_prices

    def compute_price_arbitrage_metrics(self, beta, m_grid=None, tau_grid=None, day_index=0, S_val=None, r_val=None, q_val=None):
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
        
        iv_flat = self.predict_iv(beta, M, T).flatten()
        call_prices_flat = self.ivs_to_price_surface(coords, iv_flat, day_index=day_index, S_val=S_val, r_val=r_val, q_val=q_val)
        
        if S_val is None:
            S_val = self.S[day_index] if self.S is not None and day_index < len(self.S) else 1.0
            
        c_flat = call_prices_flat / S_val
        c = c_flat.reshape(M.shape)
        m_abs = np.exp(m_grid)
        
        P1 = np.zeros_like(c)
        P2 = np.zeros_like(c)
        P3 = np.zeros_like(c)
        
        # 1. Calendar spread arbitrage (P1): tau_j * (c(m_i, tau_j) - c(m_i, tau_{j+1})) / (tau_{j+1} - tau_j)
        for j in range(len(tau_grid) - 1):
            for i in range(len(m_grid)):
                val = tau_grid[j] * (c[j, i] - c[j+1, i]) / (tau_grid[j+1] - tau_grid[j])
                P1[j, i] = max(0, val)
                
        # 2. Call spread arbitrage (P2): (c(m_{i+1}, tau_j) - c(m_i, tau_j)) / (m_{i+1} - m_i)
        for j in range(len(tau_grid)):
            for i in range(len(m_grid) - 1):
                val = (c[j, i+1] - c[j, i]) / (m_abs[i+1] - m_abs[i])
                P2[j, i] = max(0, val)
                
        # 3. Butterfly spread arbitrage (P3): 
        for j in range(len(tau_grid)):
            for i in range(1, len(m_grid) - 1):
                left_diff = (c[j, i] - c[j, i-1]) / (m_abs[i] - m_abs[i-1])
                right_diff = (c[j, i+1] - c[j, i]) / (m_abs[i+1] - m_abs[i])
                val = left_diff - right_diff
                P3[j, i] = max(0, val)
                
        return P1, P2, P3

    def plot_arbitrage_violations(self, beta, m_grid=None, tau_grid=None, num_points=50, figAngle=-70, day_label=""):
        """
        Plots where calendar and butterfly arbitrage violations occur on the reconstructed surface.
        """
        if m_grid is None:
            m_grid = np.linspace(self.range_moneyness[0], self.range_moneyness[1], num_points)
        if tau_grid is None:
            tau_min = max(1e-4, self.range_tau[0])
            tau_grid = np.linspace(tau_min, self.range_tau[1], num_points)
            
        calendar_metrics, butterfly_metrics = self.compute_arbitrage_metrics(beta, m_grid=m_grid, tau_grid=tau_grid)
        
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

    def measure_surface_arbitrage(self, beta, m_grid=None, tau_grid=None, tolerance=0.0):
        """
        Measures calendar and butterfly arbitrage violations on a continuous grid for a single surface.
        
        Returns:
        - summary: Dictionary containing violation counts, violation sums, and percentages.
        """
        calendar_metrics, butterfly_metrics = self.compute_arbitrage_metrics(beta, m_grid=m_grid, tau_grid=tau_grid)
        
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

    def measure_all_surface_arbitrage(self, betas, m_grid=None, tau_grid=None, tolerance=0.0):
        """
        Measures surface arbitrage across a sequence of fitted or generated daily Beta parameters.
        
        Returns:
        - DataFrame containing daily surface arbitrage metrics.
        """
        records = []
        for i, beta in enumerate(betas):
            summary = self.measure_surface_arbitrage(beta, m_grid=m_grid, tau_grid=tau_grid, tolerance=tolerance)
            summary['day_index'] = i
            records.append(summary)
        return pd.DataFrame(records)

    def measure_raw_arbitrage(self, m_i, tau_i, iv_i, day_index=0, S_val=None, r_val=None, q_val=None, tolerance=1e-8):
        """
        Measures static arbitrage in discrete option data for a single day.
        """
        if S_val is None:
            S_val = self.S[day_index] if self.S is not None and day_index < len(self.S) else 1.0
        if r_val is None:
            r_val = self.r[day_index] if self.r is not None and day_index < len(self.r) else 0.0
        if q_val is None:
            q_val = self.q[day_index] if self.q is not None and day_index < len(self.q) else 0.0

        m_clean, tau_clean, iv_clean, S_clean, r_clean, q_clean = self._clean_data_for_day(
            m_i, tau_i, iv_i, S_val, r_val, q_val
        )
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
        call_prices = self.ivs_to_price_surface(coords, iv_clean, day_index=day_index, S_val=S_clean, r_val=r_clean, q_val=q_clean)
        
        S_scalar = float(np.mean(S_clean)) if hasattr(S_clean, "__len__") and not isinstance(S_clean, (str, bytes)) else float(S_clean)
        strikes = S_scalar * np.exp(m_clean)
        maturities = np.round(tau_clean, 6)
        
        vert_violations, vert_sum, vert_returns, butt_violations, butt_sum, butt_returns = self._check_vertical_and_butterfly_arbitrage(
            call_prices, strikes, maturities, r_clean, tolerance
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
    forwardMoneynessFrancois = [(1/np.sqrt(t))*np.log((1/m)*np.exp((r-q)*t)) for m, t, r, q in zip(moneyness, tau, rfRate, dividendRate)]
    minFmoney = np.min([np.min(m) for m in forwardMoneynessFrancois])
    maxFmoney = np.max([np.max(m) for m in forwardMoneynessFrancois])

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
    forwardMoneynessFrancois_test = [(1/np.sqrt(t))*np.log((1/m)*np.exp((r-q)*t)) for m, t, r, q in zip(moneyness_test, tau_test, rfRate_test, dividendRate_test)]


    flattenIV_test = [v for vDay in iv_test for v in vDay]
    meanIV_test = np.mean(flattenIV)
    ivCentered_test = [v - meanIV for v in iv_test]

    ivLog_test = [np.log(v) for v in iv_test]

    #%% Estimate Francois Beta Parameters for Training Data
    extractor = FrancoisBetaExtractor(T_max=5.0, T_conv=0.25, range_moneyness= (-1,2))
    betas_train = extractor.compute_betas(forwardMoneynessFrancois, tau, iv)

    daily_r2_train, total_r2_train = extractor.compute_variance_explained(forwardMoneynessFrancois, tau, iv, betas_train)

    #%% Training Data Statistics
    val_mses = []
    val_scores = []

    for i in range(len(uniqueDates)):
        m_clean, tau_clean, iv_clean = extractor._clean_data_for_day(forwardMoneynessFrancois[i], tau[i], iv[i])
        if len(iv_clean) == 0:
            continue
        y_hat_val = extractor.predict_iv(betas_train[i], m_clean, tau_clean)
        val_mses.append(np.mean((iv_clean - y_hat_val) ** 2))
        val_scores.append(betas_train[i])

    val_rmse = np.sqrt(val_mses)

    #%% Testing Data Statistics
    betas_test = extractor.compute_betas(forwardMoneynessFrancois_test, tau_test, iv_test)
    daily_r2_test, total_r2_test = extractor.compute_variance_explained(forwardMoneynessFrancois_test, tau_test, iv_test, betas_test)

    val_mses_test = []
    val_scores_test = []

    for i in range(len(uniqueDates_test)):
        m_clean, tau_clean, iv_clean = extractor._clean_data_for_day(forwardMoneynessFrancois_test[i], tau_test[i], iv_test[i])
        if len(iv_clean) == 0:
            continue
        y_hat_val = extractor.predict_iv(betas_test[i], m_clean, tau_clean)
        val_mses_test.append(np.mean((iv_clean - y_hat_val) ** 2))
        val_scores_test.append(betas_test[i])

    val_rmse_test = np.sqrt(val_mses_test)

    #%% Display Goodness of Fit Summary
    print("=== Francois et al. (2022) Model - Goodness of Fit Summary ===")
    print(f"Training Set  - Mean MSE: {np.mean(val_mses):.6e} | Mean RMSE: {np.mean(val_rmse):.6f} | Total R^2: {total_r2_train:.4%}")
    print(f"Testing Set   - Mean MSE: {np.mean(val_mses_test):.6e} | Mean RMSE: {np.mean(val_rmse_test):.6f} | Total R^2: {total_r2_test:.4%}")

    #%% Arbitrage Analysis (Gatheral & Jacquier / FuNVol Section 4)
    print("\n=== Francois et al. (2022) Model - Arbitrage Analysis ===")
    
    # 1. Continuous Surface Arbitrage on fitted model surfaces
    extractor_spx = FrancoisBetaExtractor(T_max=5.0, T_conv=0.25, S=S, r=rfRate, q=dividendRate)
    arb_surface_train = extractor_spx.measure_all_surface_arbitrage(betas_train)
    arb_surface_test = extractor_spx.measure_all_surface_arbitrage(betas_test)
    
    print("Fitted Surface Arbitrage (Grid-based: d_tau w < 0 or g(m) < 0):")
    print(f"  [Train] Mean Calendar Violations: {arb_surface_train['calendar_violations'].mean():.2f} / {arb_surface_train['total_grid_points'].iloc[0]} ({arb_surface_train['calendar_violation_pct'].mean():.2%}) | Sum: {arb_surface_train['calendar_violation_sum'].mean():.4f}")
    print(f"  [Train] Mean Butterfly Violations: {arb_surface_train['butterfly_violations'].mean():.2f} / {arb_surface_train['total_grid_points'].iloc[0]} ({arb_surface_train['butterfly_violation_pct'].mean():.2%}) | Sum: {arb_surface_train['butterfly_violation_sum'].mean():.4f}")
    print(f"  [Test]  Mean Calendar Violations: {arb_surface_test['calendar_violations'].mean():.2f} / {arb_surface_test['total_grid_points'].iloc[0]} ({arb_surface_test['calendar_violation_pct'].mean():.2%}) | Sum: {arb_surface_test['calendar_violation_sum'].mean():.4f}")
    print(f"  [Test]  Mean Butterfly Violations: {arb_surface_test['butterfly_violations'].mean():.2f} / {arb_surface_test['total_grid_points'].iloc[0]} ({arb_surface_test['butterfly_violation_pct'].mean():.2%}) | Sum: {arb_surface_test['butterfly_violation_sum'].mean():.4f}")
    
    # 2. Raw Discrete Data Arbitrage
    arb_raw_train = extractor_spx.measure_all_raw_arbitrage(forwardMoneynessFrancois, tau, iv, S=S, r=rfRate, q=dividendRate)
    print("\nRaw Data Discrete Price Arbitrage:")
    print(f"  [Train] Mean Vertical Violations: {arb_raw_train['vertical_violations'].mean():.2f} | Mean Butterfly: {arb_raw_train['butterfly_violations'].mean():.2f} | Mean Calendar: {arb_raw_train['calendar_violations'].mean():.2f}")
    print(f"  [Train] Mean Total Violations: {arb_raw_train['total_violations'].mean():.2f} out of {arb_raw_train['total_nb_observations'].mean():.1f} options/day")

    #%% Save Beta representation fit
    with open("/Users/macbook/Documents/global_O_Research/O_Research/data/SPX_data/SPX_Francois_betas.pkl", "wb") as f:
        pickle.dump(betas_train, f)
        pickle.dump(betas_test, f)






