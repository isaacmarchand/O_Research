#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import pickle as pickle

from pygam import LinearGAM, te

from FDApy.visualization import plot
from FDApy import DenseFunctionalData
from FDApy.representation import DenseArgvals, DenseValues
from FDApy.preprocessing import UFPCA, LocalPolynomial, PSplines


#%%
class Arbitrage:
    def __init__(self):
        pass

    def compute_arbitrage_callPrice(self, callPrice_meshes, tau_mesh, K_meshes, Ss, onlyTotal = True):
        7
    
    def ratesMesh(self, tau_mesh, tau, rfRate, dividendRate):
        """
        Fore given tau mesh, we interpolate rfRate mesh and dividendRate mesh
        """
        rfRateMesh_list = []
        dividendRateMesh_list = []
        
        for i in range(len(tau)):
            tauUni, indices = np.unique(np.asarray(tau[i]), return_index=True)
            rfRateUni = np.asarray(rfRate[i])[indices]
            dividendRateUni = np.asarray(dividendRate[i])[indices]
            
            rfRateMesh_list.append(np.interp(tau_mesh, tauUni, rfRateUni))
            dividendRateMesh_list.append(np.interp(tau_mesh, tauUni, dividendRateUni))
        
        return rfRateMesh_list, dividendRateMesh_list
    
    def BS_call_normalized_Price(self, IV, S, K, tau, rfRate, dividendRate):
        """
        Computes the Black-Scholes Call price.

        Parameters:
        - IV: List of N 2D arrays of implied volatilities.
        - S: List of N scalars (spot prices).
        - K: List of N 2D arrays of strike prices.
        - tau: 2D array of time to expiry.
        - rfRate: List of N 2D arrays of risk-free rates.
        - dividendRate: List of N 2D arrays of dividend rates.

        Returns:
        - List of N 2D arrays of normalized call prices.
        """
        call_prices = []
        for i in range(len(S)):
            d1 = (np.log(S[i] / K[i]) + (rfRate[i] - dividendRate[i] + 0.5 * IV[i]**2) * tau) / (IV[i] * np.sqrt(tau))
            d2 = d1 - IV[i] * np.sqrt(tau)
            call_price = np.exp(-dividendRate[i] * tau) * norm.cdf(d1) - (K[i]/S[i]) * np.exp(-rfRate[i] * tau) * norm.cdf(d2)
            call_prices.append(call_price)
        return call_prices

    def timeMoneyness_to_strike(self, tau, S, rfRate, dividendRate, timeMoneyness):
        """
        Take the tau dependent moneyness used in the IVS modeling: log(F/K)/sqrt(tau)
        and outputs the strike price K.
        The forward is: F=Se^{(r-q)*tau}.

        Parameters:
        - tau: 2D array of time to expiry.
        - S: List of N scalars (spot prices).
        - rfRate: List of N 2D arrays of risk-free rates.
        - dividendRate: List of N 2D arrays of dividend rates.
        - timeMoneyness: 2D array of time-dependent moneyness.

        Returns:
        - List of N 2D arrays of strike prices.
        """
        forwardMoneyness = np.exp(timeMoneyness * np.sqrt(tau))

        strikes = []
        for i in range(len(S)):
            strike = S[i] / (forwardMoneyness * np.exp(-(rfRate[i] - dividendRate[i]) * tau))
            strikes.append(strike)

        return strikes


#%%
class SurfaceProjector:
    
    def __init__(self):
        
        self.T_max = 5
        self.T_conv = .25
        
        # Initialize the 5 basis functions.
        # Currently using arbitrary placeholder functions (e.g., simple polynomials).
        # You can replace these lambda functions with your actual basis functions.
        self.basis_functions = [
            lambda m, tau: np.ones_like(m),                                  # Basis 1: Long term ATM level
            lambda m, tau: np.exp(-np.sqrt(tau/self.T_conv)),                # Basis 2: Time to maturity slope
            lambda m, tau: m * (m <= 0) + (np.exp(2 * m) - 1) / (np.exp(2 * m) + 1)*(m > 0),   # Basis 3: Moneyness slope
            lambda m, tau: (1 - np.exp(-(m**2))) * np.log(self.T_max / tau),       # Basis 4: Smile attenuation
            lambda m, tau: (1 - np.exp((3 * m)**3)) * np.log(tau / self.T_max) * (m < 0) # Basis 5: Smirk
        ]
        
    def project(self, moneyness, tau, iv, showFig = False):
        """
        Projects the discrete implied volatility data onto the basis functions.

        Parameters:
        - moneyness: list of 1D array-like of moneyness values for each observation
        - tau: list of 1D array-like of time to expiry values for each observation
        - iv: list of 1D array-like of implied volatility values for each observation

        Returns:
        - coefficients: List of array of estimated coefficients for the basis functions
        """
        coef = []
        alpha = []
            
        for i in range(len(iv)):
            # Ensure inputs are flat numpy arrays
            m_flat = np.asarray(moneyness[i]).flatten()
            tau_flat = np.asarray(tau[i]).flatten()
            iv_flat = np.asarray(iv[i]).flatten()

            # Filter out any NaNs (missing data) to prevent regression errors
            valid_mask = ~np.isnan(iv_flat) & ~np.isnan(m_flat) & ~np.isnan(tau_flat)
            m_clean = m_flat[valid_mask]
            tau_clean = tau_flat[valid_mask]
            iv_clean = iv_flat[valid_mask]

            # Build the design matrix X (features)
            # Evaluate every basis function on the discrete (moneyness, tau) points
            X = np.column_stack([func(m_clean, tau_clean) for func in self.basis_functions])

            # Fit the linear regression model. 
            # Using fit_intercept=False assuming one of the basis functions handles the constant intercept
            reg = LinearRegression(fit_intercept=False)
            reg.fit(X, iv_clean)
            
            if showFig:
                self.plot_surface_and_actual(m_clean, tau_clean, iv_clean, reg.coef_)


            coef.append(reg.coef_)
            #alpha.append(reg.alpha_)
    
        #return coef, alpha
        return coef

    def compute_residuals(self, moneyness, tau, iv, coefficients_ls, scatterPlot=False):
        """
        Visualizes the residuals (actual - projected) of the implied volatility data.
        """
        residuals_ls = []
        
        for i in range(len(coefficients_ls)):
            
            coefficients = coefficients_ls[i]
            if len(coefficients) != len(self.basis_functions):
                raise ValueError("Number of coefficients must match the number of basis functions.")

            m_flat = np.asarray(moneyness[i]).flatten()
            tau_flat = np.asarray(tau[i]).flatten()
            iv_flat = np.asarray(iv[i]).flatten()

            # Filter out any NaNs (missing data)
            valid_mask = ~np.isnan(iv_flat) & ~np.isnan(m_flat) & ~np.isnan(tau_flat)
            m_clean = m_flat[valid_mask]
            tau_clean = tau_flat[valid_mask]
            iv_clean = iv_flat[valid_mask]

            # Calculate projected IV exactly at the valid discrete data points
            iv_projected = np.zeros_like(iv_clean)
            for coeff, func in zip(coefficients, self.basis_functions):
                iv_projected += coeff * func(m_clean, tau_clean)

            # Calculate residuals
            residuals = iv_clean - iv_projected
            
            if scatterPlot:
                self.plot_actual_residuals(m_clean, tau_clean, residuals)
                
            residuals_ls.append(residuals)
            
        
        return residuals_ls

    def compute_variance_explained(self, moneyness, tau, iv, coefficients_ls):
        """
        Computes the proportion of variance explained (R^2) by the basis representation 
        for each observation in the data.
        
        Parameters:
        - moneyness: list of 1D array-like of moneyness values
        - tau: list of 1D array-like of time to expiry values
        - iv: list of 1D array-like of implied volatility values
        - coefficients_ls: list of array of estimated coefficients (from the project method)
        
        Returns:
        - variance_explained_ls: List of R^2 values representing the proportion of variance explained
        """
        variance_explained_ls = []
        ss_res_ls = []
        ss_tot_ls = []
        
        for i in range(len(coefficients_ls)):
            coefficients = coefficients_ls[i]
            
            m_flat = np.asarray(moneyness[i]).flatten()
            tau_flat = np.asarray(tau[i]).flatten()
            iv_flat = np.asarray(iv[i]).flatten()

            # Filter out any NaNs (missing data)
            valid_mask = ~np.isnan(iv_flat) & ~np.isnan(m_flat) & ~np.isnan(tau_flat)
            m_clean = m_flat[valid_mask]
            tau_clean = tau_flat[valid_mask]
            iv_clean = iv_flat[valid_mask]

            # Calculate projected IV exactly at the valid discrete data points
            iv_projected = np.zeros_like(iv_clean)
            for coeff, func in zip(coefficients, self.basis_functions):
                iv_projected += coeff * func(m_clean, tau_clean)

            # Calculate sum of squared residuals and total sum of squares
            ss_res = np.sum((iv_clean - iv_projected) ** 2)
            ss_tot = np.sum((iv_clean - np.mean(iv_clean)) ** 2)
            
            # Compute R-squared (proportion of variance explained)
            if ss_tot == 0:
                r_squared = 1.0 if ss_res == 0 else 0.0
            else:
                r_squared = 1 - (ss_res / ss_tot)
                
            ss_res_ls.append(ss_res)
            ss_tot_ls.append(ss_tot)
            variance_explained_ls.append(r_squared)
            
        totR_squared = 1-(np.sum(ss_res_ls)/np.sum(ss_tot_ls))
        return variance_explained_ls, totR_squared

    def grid_etrapolation(self, moneyness, tau, residuals, m_bounds=np.exp((-.15, .1)), tau_bounds=(1/365, 1), num_points=20, plotFig=False):
        
        m_vals = np.linspace(m_bounds[0], m_bounds[1], num_points)
        tau_vals = np.linspace(tau_bounds[0]**(1/2), tau_bounds[1]**(1/2), num_points)**(2)
        M, Tau = np.meshgrid(m_vals, tau_vals)
        m_tau_mesh = np.column_stack([M.flatten(), Tau.flatten()])
        
        residuals_mesh_ls = []
        for i in range(len(residuals)):
            print(i,'/',len(residuals), sep='', end=' -- ') if i%200 == 0 else ""
            m_flat = np.asarray(moneyness[i]).flatten()
            tau_flat = np.asarray(tau[i]).flatten()
            residuals_flat = np.asarray(residuals[i]).flatten()

            # Filter out any NaNs (missing data)
            valid_mask = ~np.isnan(residuals_flat) & ~np.isnan(m_flat) & ~np.isnan(tau_flat)
            m_clean = m_flat[valid_mask]
            tau_clean = tau_flat[valid_mask]
            m_tau_obs = np.column_stack([m_clean, tau_clean])
            residuals_clean = residuals_flat[valid_mask]
            
            # Fit a GAM with tensor product interaction between moneyness (0) and tau (1)
            # Using .gridsearch() to automatically find the optimal smoothing penalty
            gam = LinearGAM(te(0, 1)).gridsearch(m_tau_obs, residuals_clean, progress=False)
            residuals_mesh = gam.predict(m_tau_mesh)
            residuals_mesh_ls.append(residuals_mesh)
            
            if plotFig:
                self.plot_extrapolated_residuals(m_tau_mesh, m_tau_obs, residuals_mesh, residuals_clean)
            
            
        
        return m_tau_mesh, residuals_mesh_ls        
     
    def compute_eigen_functions(self, moneyness, tau, residuals):
        """
        Computes the eigenfunctions of the residuals.
        """
        
        #first use local polynomial to get dense grid
        m_tau_grid, Residuals = self.grid_etrapolation(moneyness, tau, residuals)
        
        input_m = np.unique(m_tau_grid[:, 0])
        input_tau = np.unique(m_tau_grid[:, 1])
        
        argVals = DenseArgvals({
            'input_dim_0': input_tau,
            'input_dim_1': input_m
        })
        
        Residuals = np.array(Residuals)
        residualGrid = Residuals.reshape(-1, len(input_tau), len(input_m))
        
        residualVals = DenseValues(residualGrid)
        fDataFormat = DenseFunctionalData(argvals=argVals, values=residualVals)
        
        self.fpca = UFPCA(n_components=10, method="inner-product")
        self.fpca.fit(fDataFormat)
    
    def compute_scores(self, moneyness, tau, residuals):
        self.compute_eigen_functions(moneyness, tau, residuals)
        scores = self.fpca.transform()
        
        return scores
    
    def reconstruct_surface(self, coefficients, scores, m_bounds=np.exp((-.15, .1)), tau_bounds=(1/365, 1), num_points=20):
        """
        Reconstructs the implied volatility surfaces by combining the base projection 
        (from the 5 initial basis functions) and the FPCA reconstruction of the residuals.
        
        Parameters:
        - coefficients: 1D or 2D array of basis function coefficients.
        - scores: 1D or 2D array of FPCA scores.
        - m_bounds, tau_bounds, num_points: Grid parameters (should match those used in FPCA).
        
        Returns:
        - M, Tau: 2D meshgrids of moneyness and tau.
        - reconstructed_surfaces: 3D array (or 2D if single observation) of the reconstructed IV surfaces.
        """
        coefficients = np.atleast_2d(coefficients)
        scores = np.atleast_2d(scores)
        
        if coefficients.shape[0] != scores.shape[0]:
            raise ValueError("Number of observations in coefficients and scores must match.")
            
        m_vals = np.linspace(m_bounds[0], m_bounds[1], num_points)
        tau_vals = np.linspace(tau_bounds[0]**(1/2), tau_bounds[1]**(1/2), num_points)**(2)
        M, Tau = np.meshgrid(m_vals, tau_vals)
        
        n_obs = coefficients.shape[0]
        base_surfaces = np.zeros((n_obs, num_points, num_points))
        
        for i in range(n_obs):
            for coeff, func in zip(coefficients[i], self.basis_functions):
                base_surfaces[i] += coeff * func(M, Tau)
                
        if not hasattr(self, 'fpca'):
            raise ValueError("FPCA has not been fitted. Please call compute_scores() first.")
            
        # Reconstruct the dense residual grid from the FPCA scores
        reconstructed_residuals_fd = self.fpca.inverse_transform(scores)
        reconstructed_residuals = reconstructed_residuals_fd.values
        
        # Combine the parametric base surface and the non-parametric FPCA residuals
        reconstructed_surfaces = base_surfaces + reconstructed_residuals
        
        if reconstructed_surfaces.shape[0] == 1:
            return M, Tau, reconstructed_surfaces[0]
            
        return M, Tau, reconstructed_surfaces

    def compute_full_variance_explained(self, moneyness, tau, iv, coefficients_ls, scores, m_bounds=np.exp((-.15, .1)), tau_bounds=(1/365, 1), num_points=20):
        """
        Computes the proportion of variance explained (R^2) by the FULL reconstructed surface
        (base functions + FPCA residuals) for each observation in the data.
        """
        variance_explained_ls = []
        ss_res_ls = []
        ss_tot_ls = []
        
        if not hasattr(self, 'fpca'):
            raise ValueError("FPCA has not been fitted. Please call compute_scores() first.")
            
        reconstructed_residuals_fd = self.fpca.inverse_transform(scores)
        reconstructed_residuals = reconstructed_residuals_fd.values  # shape: (n_obs, len(tau_vals), len(m_vals))
        
        m_vals = np.linspace(m_bounds[0], m_bounds[1], num_points)
        tau_vals = np.linspace(tau_bounds[0]**(1/2), tau_bounds[1]**(1/2), num_points)**(2)
        
        for i in range(len(coefficients_ls)):
            coefficients = coefficients_ls[i]
            
            m_flat = np.asarray(moneyness[i]).flatten()
            tau_flat = np.asarray(tau[i]).flatten()
            iv_flat = np.asarray(iv[i]).flatten()

            valid_mask = ~np.isnan(iv_flat) & ~np.isnan(m_flat) & ~np.isnan(tau_flat)
            m_clean = m_flat[valid_mask]
            tau_clean = tau_flat[valid_mask]
            iv_clean = iv_flat[valid_mask]

            # 1. Base projected IV exactly at the valid discrete data points
            iv_base = np.zeros_like(iv_clean)
            for coeff, func in zip(coefficients, self.basis_functions):
                iv_base += coeff * func(m_clean, tau_clean)
                
            # 2. Residuals from FPCA evaluated at the discrete data points via interpolation
            interpolator = RegularGridInterpolator((tau_vals, m_vals), reconstructed_residuals[i], bounds_error=False, fill_value=None)
            points = np.column_stack([tau_clean, m_clean])
            iv_fpca_res = interpolator(points)
            
            # Combine base and FPCA residual
            iv_projected = iv_base + iv_fpca_res

            ss_res = np.sum((iv_clean - iv_projected) ** 2)
            ss_tot = np.sum((iv_clean - np.mean(iv_clean)) ** 2)
            
            r_squared = 1.0 if ss_tot == 0 and ss_res == 0 else (0.0 if ss_tot == 0 else 1 - (ss_res / ss_tot))
                
            ss_res_ls.append(ss_res)
            ss_tot_ls.append(ss_tot)
            variance_explained_ls.append(r_squared)
            
        totR_squared = 1-(np.sum(ss_res_ls)/np.sum(ss_tot_ls))
        return variance_explained_ls, totR_squared    
    
    def plot_basis_functions(self, m_bounds=np.exp((-.15, .1)), tau_bounds=(1/365, 1), num_points=30):
        """
        Visualizes the basis functions over a 2D grid of moneyness and time to expiry.
        
        Parameters:
        - m_bounds: Tuple of (min, max) for moneyness
        - tau_bounds: Tuple of (min, max) for time to expiry
        - num_points: Integer number of points to sample along each axis
        """
        # Create a meshgrid for moneyness and time to expiry
        m_vals = np.linspace(m_bounds[0], m_bounds[1], num_points)
        tau_vals = np.linspace(tau_bounds[0], tau_bounds[1], num_points)
        M, Tau = np.meshgrid(m_vals, tau_vals)

        num_basis = len(self.basis_functions)
        cols = int(np.ceil(np.sqrt(num_basis)))
        rows = int(np.ceil(num_basis / cols))
        
        basisFeaturName = ['Long term ATM level', 'Time to maturity slope', 'Moneyness slope', 'Smile attenuation', 'Smirk']
        
        fig = plt.figure(figsize=(5 * cols, 4 * rows))

        for i, func in enumerate(self.basis_functions):
            ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
            Z = func(M, Tau)
            
            ax.plot_surface(M, Tau, Z, cmap='viridis', edgecolor='none', alpha=0.8)
            ax.set_title(basisFeaturName[i])
            ax.set_xlabel("Moneyness (m)")
            ax.set_ylabel("Time to Expiry (tau)")
            
        plt.tight_layout()
        plt.show()

    def plot_projected_surface(self, coefficients, m_bounds=np.exp((-.15, .1)), tau_bounds=(1/365, 1), num_points=30, figAngle=-70):
        """
        Visualizes the fitted implied volatility surface given the estimated coefficients.
        """
        if len(coefficients) != len(self.basis_functions):
            raise ValueError("Number of coefficients must match the number of basis functions.")

        # Create a meshgrid for moneyness and time to expiry
        m_vals = np.linspace(m_bounds[0], m_bounds[1], num_points)
        tau_vals = np.linspace(tau_bounds[0], tau_bounds[1], num_points)
        M, Tau = np.meshgrid(m_vals, tau_vals)

        # Initialize the projected surface Z with zeros
        Z = np.zeros_like(M)

        # Add the weighted contribution of each basis function
        for coeff, func in zip(coefficients, self.basis_functions):
            Z += coeff * func(M, Tau)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot_surface(M, Tau, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        ax.set_title("Projected Implied Volatility Surface")
        ax.set_xlabel("Moneyness (m)")
        ax.set_ylabel("Time to Expiry (tau)")
        ax.set_zlabel("Implied Volatility (IV)")
        
        ax.view_init(elev=15, azim=figAngle)
        
        plt.show()

    def plot_actual_surface(self, moneyness, tau, iv, scatterPlot=True,  figAngle=-70):
        """
        Visualizes the actual discrete implied volatility data as a 3D surface or scatter plot.
        """
        m_flat = np.asarray(moneyness).flatten()
        tau_flat = np.asarray(tau).flatten()
        iv_flat = np.asarray(iv).flatten()

        # Filter out any NaNs (missing data)
        valid_mask = ~np.isnan(iv_flat) & ~np.isnan(m_flat) & ~np.isnan(tau_flat)
        m_clean = m_flat[valid_mask]
        tau_clean = tau_flat[valid_mask]
        iv_clean = iv_flat[valid_mask]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        if scatterPlot:
            ax.scatter(m_clean, tau_clean, iv_clean, c=iv_clean, cmap='viridis', alpha=0.8)
            ax.set_title("Actual Implied Volatility Data")
        else:
            # Use plot_trisurf to generate a surface from unstructured discrete data
            ax.plot_trisurf(m_clean, tau_clean, iv_clean, cmap='viridis', edgecolor='none', alpha=0.8)
            ax.set_title("Actual Implied Volatility Surface")
            
        ax.set_xlabel("Moneyness (m)")
        ax.set_ylabel("Time to Expiry (tau)")
        ax.set_zlabel("Implied Volatility (IV)")
        
        ax.view_init(elev=15, azim=figAngle)
        
        plt.show()

    def plot_surface_and_actual(self, moneyness, tau, iv, coefficients, m_bounds=np.exp((-.15, .1)), tau_bounds=(1/365, 1), num_points=30,  figAngle=-70):
        """
        Visualizes both the projected implied volatility surface and the actual discrete data points.
        """
        if len(coefficients) != len(self.basis_functions):
            raise ValueError("Number of coefficients must match the number of basis functions.")

        # Create a meshgrid for moneyness and time to expiry
        m_vals = np.linspace(m_bounds[0], m_bounds[1], num_points)
        tau_vals = np.linspace(tau_bounds[0], tau_bounds[1], num_points)
        M, Tau = np.meshgrid(m_vals, tau_vals)

        # Initialize the projected surface Z with zeros
        Z = np.zeros_like(M)

        # Add the weighted contribution of each basis function
        for coeff, func in zip(coefficients, self.basis_functions):
            Z += coeff * func(M, Tau)

        # Clean actual data
        m_flat = np.asarray(moneyness).flatten()
        tau_flat = np.asarray(tau).flatten()
        iv_flat = np.asarray(iv).flatten()

        valid_mask = ~np.isnan(iv_flat) & ~np.isnan(m_flat) & ~np.isnan(tau_flat)
        m_clean = m_flat[valid_mask]
        tau_clean = tau_flat[valid_mask]
        iv_clean = iv_flat[valid_mask]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot_surface(M, Tau, Z, cmap='viridis', edgecolor='none', alpha=0.6)
        ax.scatter(m_clean, tau_clean, iv_clean, edgecolors='k',facecolors='none', marker='o', alpha=0.3, label='Actual Data')
        
        ax.set_title("Projected Surface vs Actual Implied Volatility")
        ax.set_xlabel("Moneyness (m)")
        ax.set_ylabel("Time to Expiry (tau)")
        ax.set_zlabel("Implied Volatility (IV)")
        ax.legend()
        
        # Adjust the viewing angle (elevation) to look at it from a lower perspective
        ax.view_init(elev=15, azim=figAngle)
        
        plt.show()

    def plot_actual_residuals(self, moneyness, tau, residuals, figAngle=-70):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(moneyness, tau, residuals, c=residuals, cmap='coolwarm', alpha=0.8)
        ax.set_title("Implied Volatility Residuals (Actual - Projected)")
        ax.set_xlabel("Moneyness (m)")
        ax.set_ylabel("Time to Expiry (tau)")
        ax.set_zlabel("Residual")
        ax.view_init(elev=15, azim=figAngle)
        plt.show()

    def plot_extrapolated_residuals(self, m_tau_grid, m_tau_obs, residuals_extrapolated, residuals, figAngle=-70):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        nMoney = len(np.unique(m_tau_grid[:, 0]))
        nTau = len(np.unique(m_tau_grid[:, 1]))
        # Reshape extrapolated points to 2D grid for surface plotting
        Z_mesh = residuals_extrapolated.reshape(nMoney, nTau)
        M = (m_tau_grid[:, 0]).reshape(nMoney, nTau)
        Tau = (m_tau_grid[:, 1]).reshape(nMoney, nTau)

        ax.plot_surface(M, Tau, Z_mesh, cmap='viridis', alpha=0.6, edgecolor='none')
        
        # Plot the actual residual points
        ax.scatter(m_tau_obs[:,0], m_tau_obs[:,1], residuals, c='r', marker='o', alpha=1.0, label='Actual Residuals')
        
        ax.set_title(f"Extrapolated Residual Surface vs Actual")
        ax.set_xlabel("Moneyness (m)")
        ax.set_ylabel("Time to Expiry (tau)")
        ax.set_zlabel("Residual")
        ax.legend()
        
        ax.view_init(elev=15, azim=figAngle)
        
        plt.show()

    def plot_reconstructed_surface_and_actual(self, moneyness, tau, iv, coefficients, scores, m_bounds=np.exp((-.15, .1)), tau_bounds=(1/365, 1), num_points=20, figAngle=-70):
        """
        Visualizes the reconstructed implied volatility surface (base + FPCA residuals)
        alongside the actual discrete data points for a single observation.
        """
        # Reconstruct the dense surface
        M, Tau, Z = self.reconstruct_surface(coefficients, scores, m_bounds=m_bounds, tau_bounds=tau_bounds, num_points=num_points)
        
        # Clean actual data
        m_flat = np.asarray(moneyness).flatten()
        tau_flat = np.asarray(tau).flatten()
        iv_flat = np.asarray(iv).flatten()

        valid_mask = ~np.isnan(iv_flat) & ~np.isnan(m_flat) & ~np.isnan(tau_flat)
        m_clean = m_flat[valid_mask]
        tau_clean = tau_flat[valid_mask]
        iv_clean = iv_flat[valid_mask]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot_surface(M, Tau, Z, cmap='viridis', edgecolor='none', alpha=0.6)
        ax.scatter(m_clean, tau_clean, iv_clean, edgecolors='k', facecolors='none', marker='o', alpha=0.3, label='Actual Data')
        
        ax.set_title("Reconstructed Surface vs Actual Implied Volatility")
        ax.set_xlabel("Moneyness (m)")
        ax.set_ylabel("Time to Expiry (tau)")
        ax.set_zlabel("Implied Volatility (IV)")
        ax.legend()
        
        ax.view_init(elev=15, azim=figAngle)
        
        plt.show()

    def plot_reconstructed_residuals_and_actual(self, moneyness, tau, residuals, scores, m_bounds=np.exp((-.15, .1)), tau_bounds=(1/365, 1), num_points=20, figAngle=-70):
        """
        Visualizes the reconstructed residual surface (from FPCA) alongside the actual discrete residual points.
        """
        scores = np.atleast_2d(scores)
        if not hasattr(self, 'fpca'):
            raise ValueError("FPCA has not been fitted. Please call compute_scores() first.")
            
        # Reconstruct the dense residual grid from the FPCA scores
        reconstructed_residuals_fd = self.fpca.inverse_transform(scores)
        Z = reconstructed_residuals_fd.values[0]
        
        m_vals = np.linspace(m_bounds[0], m_bounds[1], num_points)
        tau_vals = np.linspace(tau_bounds[0]**(1/2), tau_bounds[1]**(1/2), num_points)**(2)
        M, Tau = np.meshgrid(m_vals, tau_vals)
        
        # Clean actual data
        m_flat = np.asarray(moneyness).flatten()
        tau_flat = np.asarray(tau).flatten()
        res_flat = np.asarray(residuals).flatten()

        valid_mask = ~np.isnan(res_flat) & ~np.isnan(m_flat) & ~np.isnan(tau_flat)
        m_clean = m_flat[valid_mask]
        tau_clean = tau_flat[valid_mask]
        res_clean = res_flat[valid_mask]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot_surface(M, Tau, Z, cmap='coolwarm', edgecolor='none', alpha=0.6)
        ax.scatter(m_clean, tau_clean, res_clean, edgecolors='k', facecolors='none', marker='o', alpha=0.8, label='Actual Residuals')
        
        ax.set_title("Reconstructed Residual Surface vs Actual Residuals")
        ax.set_xlabel("Moneyness (m)")
        ax.set_ylabel("Time to Expiry (tau)")
        ax.set_zlabel("Residual")
        ax.legend()
        
        ax.view_init(elev=15, azim=figAngle)
        
        plt.show()

    def plot_eigenfunctions(self, figAngle=-70):
        """
        Visualizes all the eigenfunctions calculated by the FPCA.
        """
        if not hasattr(self, 'fpca'):
            raise ValueError("FPCA has not been fitted. Please call compute_scores() first.")
            
        eigenfunctions = self.fpca.eigenfunctions
        n_components = eigenfunctions.n_obs
        
        tau_grid = eigenfunctions.argvals['input_dim_0']
        m_grid = eigenfunctions.argvals['input_dim_1']
        
        M, Tau = np.meshgrid(m_grid, tau_grid)
        
        cols = int(np.ceil(np.sqrt(n_components)))
        rows = int(np.ceil(n_components / cols))
        
        fig = plt.figure(figsize=(6 * cols, 5 * rows))
        
        for i in range(n_components):
            ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
            Z = eigenfunctions.values[i]
            
            ax.plot_surface(M, Tau, Z, cmap='coolwarm', edgecolor='none', alpha=0.8)
            ax.set_title(f"Eigenfunction {i+1}")
            ax.set_xlabel("Moneyness (m)")
            ax.set_ylabel("Time to Expiry (tau)")
            ax.set_zlabel("Amplitude")
            ax.view_init(elev=15, azim=figAngle)
            
        plt.tight_layout()
        plt.show()
   
#%%
if __name__ == '__main__':
    #%% Load SPX Data formated
    with open("/Users/macbook/Documents/global_O_Research/O_Research/data/SPX_data/SPX_lists_traded.pkl", "rb") as f:
        uniqueDates = pickle.load(f)
        tau = pickle.load(f)
        moneyness = pickle.load(f)
        iv = pickle.load(f)
        S = pickle.load(f)
        rfRate = pickle.load(f)
        dividendRate = pickle.load(f)
        
        
    forwardMoneynessFrancois = [(1/np.sqrt(t))*np.log((1/m)*np.exp((r-q)*t)) for m, t, r, q in zip(moneyness, tau, rfRate, dividendRate)]

    #%% Compute K-L expansion of residuals (~40min) 
    projector = SurfaceProjector()
    coefficients = projector.project(forwardMoneynessFrancois, tau, iv, showFig = False)
    variance_explained, totVarExplained = projector.compute_variance_explained(forwardMoneynessFrancois, tau, iv, coefficients)
    print(f"Average variance explained: {np.mean(variance_explained):.2%}")
    plt.hist(variance_explained, 30)
    residuals = projector.compute_residuals(forwardMoneynessFrancois, tau, iv, coefficients, scatterPlot=False)
    scoresData = projector.compute_scores(forwardMoneynessFrancois, tau, residuals)
    with open("/Users/macbook/Documents/global_O_Research/O_Research/data/SPX_data/fdaOfResiduals_SPX.pkl", "wb") as f:
        pickle.dump(scoresData, f)
        pickle.dump(projector.fpca, f)
     
    
    
    #%% Look at specific surfaces
    
    projector = SurfaceProjector()
    coefficients = projector.project(forwardMoneynessFrancois, tau, iv, showFig = False)
    variance_explained, totVarExplained = projector.compute_variance_explained(forwardMoneynessFrancois, tau, iv, coefficients)
    print(f"Average variance explained: {np.mean(variance_explained):.2%}")
    _=plt.hist(variance_explained, 30)
    residuals = projector.compute_residuals(forwardMoneynessFrancois, tau, iv, coefficients, scatterPlot=False)
    with open("/Users/macbook/Documents/global_O_Research/O_Research/data/SPX_data/fdaOfResiduals_SPX.pkl", "rb") as f:
        scoresData = pickle.load(f)
        projector.fpca = pickle.load(f)
     
    varianceRecon_explained, totVarReconExplained = projector.compute_full_variance_explained(forwardMoneynessFrancois, tau, iv, coefficients, scoresData)
    print(f"Average variance explained: {np.mean(varianceRecon_explained):.2%}")
    _=plt.hist(varianceRecon_explained, 30)
    
    dateOfInterest = ['2006-05-08', '2008-12-01', '2019-12-31']
    idOfInterest = np.where(np.isin(uniqueDates, dateOfInterest))[0]
    moneyInterest = [forwardMoneynessFrancois[i] for i in idOfInterest]
    tauInterest = [tau[i] for i in idOfInterest]
    ivInterest = [iv[i] for i in idOfInterest]
    
    ID = 0
    ID_data = idOfInterest[ID]
    _= projector.plot_surface_and_actual(moneyInterest[ID], tauInterest[ID], ivInterest[ID], coefficients[ID_data])
    _= projector.plot_reconstructed_surface_and_actual(moneyInterest[ID], tauInterest[ID], ivInterest[ID], coefficients[ID_data], scoresData[ID_data])
    
    #%% Show surfaces that have low explanation of their variance from functional representation
    projectorSpecific = SurfaceProjector()
    coefficients = projectorSpecific.project(forwardMoneynessFrancois, tau, iv, showFig = False)
    variance_explained, totVarExplained = projectorSpecific.compute_variance_explained(forwardMoneynessFrancois, tau, iv, coefficients)
    residuals = projectorSpecific.compute_residuals(forwardMoneynessFrancois, tau, iv, coefficients, scatterPlot=False)
    with open("/Users/macbook/Documents/global_O_Research/O_Research/data/SPX_data/fdaOfResiduals_SPX.pkl", "rb") as f:
        scoresData = pickle.load(f)
        projectorSpecific.fpca = pickle.load(f)
        
    varianceRecon_explained, totVarReconExplained = projector.compute_full_variance_explained(forwardMoneynessFrancois, tau, iv, coefficients, scoresData)
        
    badVarianceID = np.where(np.array(variance_explained)<.65)[0]
    moneyBadVar = [forwardMoneynessFrancois[i] for i in badVarianceID]
    tauBadVar = [tau[i] for i in badVarianceID]
    ivBadVar = [iv[i] for i in badVarianceID]
    uniqueDatesBadVar = [uniqueDates[i] for i in badVarianceID]
    
    print(f"Average variance explained for those that use to have < 65% variance explained: {np.mean([varianceRecon_explained[i] for i in badVarianceID]):.2%}")
    
    ID = 1
    ID_data = badVarianceID[ID]
    _= projectorSpecific.plot_surface_and_actual(moneyBadVar[ID], tauBadVar[ID], ivBadVar[ID], coefficients[ID_data])
    _= projectorSpecific.plot_reconstructed_surface_and_actual(moneyBadVar[ID], tauBadVar[ID], ivBadVar[ID], coefficients[ID_data], scoresData[ID_data])
    print(uniqueDatesBadVar[ID])
    #%% Show same example surface as in francois 2022
    projectorSpecific = SurfaceProjector()
    dateOfInterest = ['2006-05-08', '2008-12-01', '2019-12-31']
    idOfInterest = np.where(np.isin(uniqueDates, dateOfInterest))[0]
    moneyInterest = [forwardMoneynessFrancois[i] for i in idOfInterest]
    tauInterest = [tau[i] for i in idOfInterest]
    ivInterest = [iv[i] for i in idOfInterest]
    
    coefficientsInterest = projectorSpecific.project(moneyInterest, tauInterest, ivInterest, showFig = True)
    variance_explainedInterest, totVarExplainedInterest = projectorSpecific.compute_variance_explained(moneyInterest, tauInterest, ivInterest, coefficientsInterest)
    print(f"Average variance explained: {np.mean(variance_explained):.2%}")
    residualsInterest = projectorSpecific.compute_residuals(moneyInterest, tauInterest, ivInterest, coefficientsInterest, scatterPlot=True)
    
    m_tau_mesh, residuals_mesh = projectorSpecific.grid_etrapolation(moneyInterest, tauInterest, residualsInterest, plotFig=True)
    scoresOfInterest = projectorSpecific.compute_scores(moneyInterest, tauInterest, residualsInterest)
    
    
    #%% Show time series data from projection of basis functions
    ts_coefs = np.array(coefficients)
    uniqueDatesPD = pd.to_datetime(uniqueDates)
    
    basisFeaturName = ['Long term ATM level', 'Time to maturity slope', 'Moneyness slope', 'Smile attenuation', 'Smirk']
        
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    for i in range(5):
        axes[i].plot(uniqueDatesPD, ts_coefs[:, i])
        axes[i].set_ylabel(basisFeaturName[i])
        axes[i].grid(True)
        
    plt.tight_layout()
    plt.show()
    
    #%% show time series of first 10 scores from K-L expansion
    uniqueDatesPD = pd.to_datetime(uniqueDates)
    fig, axes = plt.subplots(10, 1, figsize=(10, 12), sharex=True)
    for i in range(10):
        axes[i].plot(uniqueDatesPD, scoresData[:, i])
        axes[i].set_ylabel(f"Score {i+1}")
        axes[i].grid(True)
        
    plt.tight_layout()
    plt.show()
    
    #%% compute arbitrage (not working for now)
    
    projector = SurfaceProjector()
    coefficients = projector.project(forwardMoneynessFrancois, tau, iv, showFig = False)
    with open("/Users/macbook/Documents/global_O_Research/O_Research/data/SPX_data/fdaOfResiduals_SPX.pkl", "rb") as f:
        scoresData = pickle.load(f)
        projector.fpca = pickle.load(f)
     
    varianceRecon_explained, totVarReconExplained = projector.compute_full_variance_explained(forwardMoneynessFrancois, tau, iv, coefficients, scoresData)
    print(f"Average variance explained: {np.mean(varianceRecon_explained):.2%}")
    
    m_mesh, tau_mesh, iv_mesh = projector.reconstruct_surface(coefficients, scoresData)
    
    computeArb = Arbitrage()
    
    rfRate_meshes, dividendRate_meshes = computeArb.ratesMesh(tau_mesh, tau, rfRate, dividendRate)
    
    uniqueS = [price.unique() for price in S]
    
    strike_mesh = computeArb.timeMoneyness_to_strike(tau_mesh, uniqueS, rfRate_meshes, dividendRate_meshes, m_mesh)
    
    callP_mesh = computeArb.BS_call_normalized_Price(iv_mesh, uniqueS, strike_mesh, tau_mesh, rfRate_meshes, dividendRate_meshes)
    
    #%% use K-L decomp directly
    projector = SurfaceProjector()
    
    # #Visualizing the GAM smoothing directly on observations
    
    # dateOfInterest = ['2006-05-08', '2008-12-01', '2019-12-31']
    # idOfInterest = np.where(np.isin(uniqueDates, dateOfInterest))[0]
    # moneyInterest = [forwardMoneynessFrancois[i] for i in idOfInterest]
    # tauInterest = [tau[i] for i in idOfInterest]
    # ivInterest = [iv[i] for i in idOfInterest]
    # _=projector.grid_etrapolation(moneyInterest, tauInterest, ivInterest, m_bounds=np.exp((-.15, .1)), tau_bounds=(1/365, 1), plotFig=True)
    
    
    scoresData = projector.compute_scores(forwardMoneynessFrancois, tau, iv)
    with open("/Users/macbook/Documents/global_O_Research/O_Research/data/SPX_data/fdaOfIVS_SPX.pkl", "wb") as f:
        pickle.dump(scoresData, f)
        pickle.dump(projector.fpca, f)
    
    #%% show time series of first 5 scores from K-L expansion
    projector = SurfaceProjector()
    with open("/Users/macbook/Documents/global_O_Research/O_Research/data/SPX_data/fdaOfIVS_SPX.pkl", "rb") as f:
        scoresData = pickle.load(f)
        projector.fpca = pickle.load(f)
    uniqueDatesPD = pd.to_datetime(uniqueDates)
    fig, axes = plt.subplots(10, 1, figsize=(10, 12), sharex=True)
    for i in range(10):
        axes[i].plot(uniqueDatesPD, scoresData[:, i])
        axes[i].set_ylabel(f"Score {i+1}")
        axes[i].grid(True)
        
    plt.tight_layout()
    plt.show()
    
#%%
