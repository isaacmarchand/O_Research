#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.special import roots_legendre
from sklearn.linear_model import LinearRegression

import pickle as pickle


#%%

class FPCA:
    def __init__(self, moneyness, tau, iv, nb_spline_moneyness = 10, nb_spline_tau = 10, order_moneyness = 4, order_tau = 4):
        
        # Clean data once to avoid repeated computation in iterations
        self.cleaned_data = []
        for i in range(len(iv)):
            m_f = np.asarray(moneyness[i]).flatten()
            t_f = np.asarray(tau[i]).flatten()
            iv_f = np.asarray(iv[i]).flatten()
            mask = ~np.isnan(iv_f) & ~np.isnan(m_f) & ~np.isnan(t_f)
            self.cleaned_data.append((m_f[mask], t_f[mask], iv_f[mask]))
        
        #range of moneyness and time to expiry (tau) for which the model will be trained on
        self.range_moneyness = [-.15, .1] #range set for logMoneyness (can be change sepending on type of moneyness)
        self.range_tau = [0, 1]
        
        
        #initiate b-splines tensor product that will be used to estimate eigen functions
        self.nb_spline_moneyness = nb_spline_moneyness
        self.nb_spline_tau = nb_spline_tau
        self.order_moneyness = order_moneyness      #order of splines
        self.order_tau = order_tau
        
        self.BSplines_2d(self.range_moneyness, self.range_tau, 
                         self.nb_spline_moneyness, self.nb_spline_tau,
                         self.order_moneyness, self.order_tau)
        
        #set starting weight beta
        self.beta0 = np.zeros((self.nb_spline_moneyness, self.nb_spline_tau))+0.1
        #self.beta0 = np.repeat(np.linspace(0.1,.01,10),10).reshape(10,10).transpose() #example with tte slope as starting point
        
        #Create list of betas containing all the matrix beta fitted so far for the FPCs 
        self.betaList = []
       
    def first_FPC_fit(self, threshold = 1e-4, maxit = 10):
        """
        Estimate the first FPC and it's scores

        Parameters:
        - treshold: indicate what change in MSE do we consider as reaching convergence
        - maxit: maximum number of iterations

        Returns:
        - scores: List of array of estimated scores for the first FPC
        - betaFitted: 2D array of dimension (nb_spline_moneyness X nb_spline_tau) specifying the first FPC
        """

        beta = self.beta0
        
        # Rescale the beta s.t. 2-norm of eigen function is 1
        norm = self.compute_norm(beta)
        beta = beta / norm
        
        old_mse = 1e10 # Large initial value
        old_Beta = beta
        maxBetaChange = 1e10
        j = 0
        
        #create score Matrix nb_days X 1 -> will add columns as nb of FPC increases
        self.scoreMat = np.zeros((len(self.cleaned_data),1))
        
        while j < maxit:
            scores = []
            mse_list = []  
            for i in range(len(self.cleaned_data)):
                m_clean, tau_clean, iv_clean = self.cleaned_data[i]
                
                if len(iv_clean) == 0:
                    scores.append(0.0)
                    continue

                # Build the design matrix X (features)
                # Evaluate every basis function on the discrete (moneyness, tau) points
                X = self.evaluator(beta, np.column_stack([m_clean, tau_clean])).reshape(-1, 1)

                # Fit the linear regression model for alpha_i
                reg = LinearRegression(fit_intercept=False)
                reg.fit(X, iv_clean)

                alpha_i = reg.coef_[0]
                scores.append(alpha_i)
                mse_list.append(np.mean((iv_clean - reg.predict(X))**2))
            
            avg_mse = np.mean(mse_list)
            print(f'FPC 1, Iteration {j+1} : MSE change = {np.abs(avg_mse - old_mse) / old_mse}, Max Beta Change = {maxBetaChange}')
            if np.abs(avg_mse - old_mse) / old_mse < threshold:
                break
            
            if maxBetaChange < threshold:
                break
            
            old_mse = avg_mse
            
            # Minimize loss for beta
            # sum_XX @ beta_flat = sum_Xy
            n_beta = self.nb_spline_moneyness * self.nb_spline_tau
            sum_XX = np.zeros((n_beta, n_beta))
            sum_Xy = np.zeros(n_beta)
            
            for i in range(len(self.cleaned_data)):
                m_clean, tau_clean, iv_clean = self.cleaned_data[i]
                if len(iv_clean) == 0: continue
                
                # Basis matrices
                D_m = BSpline.design_matrix(m_clean, self.t_m, self.order_moneyness - 1).toarray()
                D_t = BSpline.design_matrix(tau_clean, self.t_t, self.order_tau - 1).toarray()
                
                # Row-wise Kronecker product using einsum
                # X_i[j, a*s2 + b] = D_m[j, a] * D_t[j, b]
                X_i = np.einsum('ja,jb->jab', D_m, D_t).reshape(len(iv_clean), -1)
                
                alpha_i = scores[i]
                sum_XX += (alpha_i**2) * (X_i.T @ X_i)
                sum_Xy += alpha_i * (X_i.T @ iv_clean)
            
            # Solve normal equations
            beta_flat = np.linalg.solve(sum_XX, sum_Xy)
            beta = beta_flat.reshape(self.nb_spline_moneyness, self.nb_spline_tau)
            
            # Rescale the beta s.t. 2-norm of eigen function is 1
            norm = self.compute_norm(beta)
            beta = beta / norm
            j+=1
            
            maxBetaChange = np.max(np.abs(beta - old_Beta))
            old_Beta = beta
        
        self.scoreMat[:, 0] = scores
        self.betaList.append(beta)
        return scores, beta
        
    def subsequent_FPC_fit(self, threshold = 1e-4, maxit = 10):
        """
        Estimate the subsequent FPC and their scores conditional on the FPC being ortogonL TO ll previous FPCs

        Parameters:
        - treshold: indicate what change in MSE do we consider as reaching convergence
        - maxit: maximum number of iterations

        Returns:
        - scores: List of array of estimated scores for the first FPC
        - betaList: list of 2D array of dimension (nb_spline_moneyness X nb_spline_tau) specifying all the previously fitted FPCs
        """
        
        beta = self.beta0
        
        # Rescale the beta s.t. 2-norm of eigen function is 1
        norm = self.compute_norm(beta)
        beta = beta / norm
        
        old_mse = 1e10 # Large initial value
        old_Beta = beta
        maxBetaChange = 1e10
        j = 0
        
        # Number of current FPC being fitted
        curr_fpc_idx = len(self.betaList) + 1
        
        # Add a column for the new FPC scores
        self.scoreMat = np.column_stack((self.scoreMat, np.zeros((len(self.cleaned_data), 1))))
        
        while j < maxit:
            mse_list = []  
            for i in range(len(self.cleaned_data)):
                m_clean, tau_clean, iv_clean = self.cleaned_data[i]
                
                if len(iv_clean) == 0:
                    self.scoreMat[i, :] = 0.0
                    continue

                # Build the design matrix X (features)
                # Columns are evaluations of all FPCs (previous + current) at the points
                X = np.zeros((len(iv_clean), curr_fpc_idx))
                for k in range(len(self.betaList)):
                    X[:, k] = self.evaluator(self.betaList[k], np.column_stack([m_clean, tau_clean]))

                # Evaluate current FPC candidate on the discrete (moneyness, tau) points
                X[:, -1] = self.evaluator(beta, np.column_stack([m_clean, tau_clean]))

                # Fit the linear regression model for all scores alpha_i
                reg = LinearRegression(fit_intercept=False)
                reg.fit(X, iv_clean)

                alpha_i = reg.coef_
                self.scoreMat[i, :] = alpha_i
                mse_list.append(np.mean((iv_clean - reg.predict(X))**2))
            
            avg_mse = np.mean(mse_list)
            print(f'FPC {curr_fpc_idx}, Iteration {j+1} : MSE change = {np.abs(avg_mse - old_mse) / old_mse}, Max Beta Change = {maxBetaChange}')
            if np.abs(avg_mse - old_mse) / old_mse < threshold:
                break
            
            if maxBetaChange < threshold:
                break
            
            old_mse = avg_mse
            
            # Minimize loss for beta subject to orthogonality constraints
            n_beta = self.nb_spline_moneyness * self.nb_spline_tau
            sum_XX = np.zeros((n_beta, n_beta))
            sum_Xy = np.zeros(n_beta)
            
            for i in range(len(self.cleaned_data)):
                m_clean, tau_clean, iv_clean = self.cleaned_data[i]
                if len(iv_clean) == 0: continue
                
                # Basis matrices
                D_m = BSpline.design_matrix(m_clean, self.t_m, self.order_moneyness - 1).toarray()
                D_t = BSpline.design_matrix(tau_clean, self.t_t, self.order_tau - 1).toarray()
                
                # Row-wise Kronecker product using einsum to build basis design matrix
                X_i = np.einsum('ja,jb->jab', D_m, D_t).reshape(len(iv_clean), -1)
                
                alpha_i = self.scoreMat[i, :]
                curr_alpha = alpha_i[-1]
                
                # Residuals: iv_clean - sum_{m < J} alpha_{im} * psi_m
                prev_fit = np.zeros(len(iv_clean))
                for k in range(len(self.betaList)):
                    prev_fit += alpha_i[k] * self.evaluator(self.betaList[k], np.column_stack([m_clean, tau_clean]))
                
                resid_i = iv_clean - prev_fit
                
                sum_XX += (curr_alpha**2) * (X_i.T @ X_i)
                sum_Xy += curr_alpha * (X_i.T @ resid_i)
            
            # Constraints matrix A: A @ vec(beta) = 0 for each previous FPC
            # The inner product <psi_J, psi_m> = vec(beta_J)^T (W_t @ W_m) vec(beta_m)
            # which is equivalent to vec(beta_J)^T vec(W_m @ beta_m @ W_t) = 0
            A_list = []
            for prev_beta in self.betaList:
                A_list.append((self.W_m @ prev_beta @ self.W_t).flatten())
            A = np.array(A_list)
            
            # Solve normal equations with linear equality constraints using KKT system:
            # [ sum_XX  A.T ] [ vec(beta) ] = [ sum_Xy ]
            # [ A        0  ] [ lambda    ]   [   0    ]
            KKT_A = np.block([
                [sum_XX, A.T],
                [A, np.zeros((len(self.betaList), len(self.betaList)))]
            ])
            KKT_b = np.concatenate([sum_Xy, np.zeros(len(self.betaList))])
            
            try:
                sol = np.linalg.solve(KKT_A, KKT_b)
                beta_flat = sol[:n_beta]
            except np.linalg.LinAlgError:
                print('error in KKT system')
                # Fallback to least squares if KKT matrix is singular
                beta_flat, _, _, _ = np.linalg.lstsq(KKT_A, KKT_b, rcond=None)
                beta_flat = beta_flat[:n_beta]
            
            
            beta = beta_flat.reshape(self.nb_spline_moneyness, self.nb_spline_tau)
            
            # Rescale the beta s.t. 2-norm of eigen function is 1
            norm = self.compute_norm(beta)
            beta = beta / norm
            j += 1
            
            maxBetaChange = np.max(np.abs(beta - old_Beta))
            old_Beta = beta
            
        self.betaList.append(beta)
        return self.scoreMat[:, -1].tolist(), beta

    def subsequent_FPC_fit_NotOrthogonal(self, threshold = 1e-4, maxit = 10):
        """
        Estimate the subsequent FPC and their scores conditional on the FPC being ortogonL TO ll previous FPCs

        Parameters:
        - treshold: indicate what change in MSE do we consider as reaching convergence
        - maxit: maximum number of iterations

        Returns:
        - scores: List of array of estimated scores for the first FPC
        - betaList: list of 2D array of dimension (nb_spline_moneyness X nb_spline_tau) specifying all the previously fitted FPCs
        """
        
        beta = self.beta0
        
        # Rescale the beta s.t. 2-norm of eigen function is 1
        norm = self.compute_norm(beta)
        beta = beta / norm
        
        old_mse = 1e10 # Large initial value
        old_Beta = beta
        maxBetaChange = 1e10
        j = 0
        
        # Number of current FPC being fitted
        curr_fpc_idx = len(self.betaList) + 1
        
        # Add a column for the new FPC scores
        self.scoreMat = np.column_stack((self.scoreMat, np.zeros((len(self.cleaned_data), 1))))
        
        while j < maxit:
            mse_list = []  
            for i in range(len(self.cleaned_data)):
                m_clean, tau_clean, iv_clean = self.cleaned_data[i]
                
                if len(iv_clean) == 0:
                    self.scoreMat[i, :] = 0.0
                    continue

                # Build the design matrix X (features)
                # Columns are evaluations of all FPCs (previous + current) at the points
                X = np.zeros((len(iv_clean), curr_fpc_idx))
                for k in range(len(self.betaList)):
                    X[:, k] = self.evaluator(self.betaList[k], np.column_stack([m_clean, tau_clean]))

                # Evaluate current FPC candidate on the discrete (moneyness, tau) points
                X[:, -1] = self.evaluator(beta, np.column_stack([m_clean, tau_clean]))

                # Fit the linear regression model for all scores alpha_i
                reg = LinearRegression(fit_intercept=False)
                reg.fit(X, iv_clean)

                alpha_i = reg.coef_
                self.scoreMat[i, :] = alpha_i
                mse_list.append(np.mean((iv_clean - reg.predict(X))**2))
            
            avg_mse = np.mean(mse_list)
            print(f'FPC {curr_fpc_idx}, Iteration {j+1} : MSE change = {np.abs(avg_mse - old_mse) / old_mse}, Max Beta Change = {maxBetaChange}')
            if np.abs(avg_mse - old_mse) / old_mse < threshold:
                break
            
            if maxBetaChange < threshold:
                break
            
            old_mse = avg_mse
            
            # Minimize loss for beta subject to orthogonality constraints
            n_beta = self.nb_spline_moneyness * self.nb_spline_tau
            sum_XX = np.zeros((n_beta, n_beta))
            sum_Xy = np.zeros(n_beta)
            
            for i in range(len(self.cleaned_data)):
                m_clean, tau_clean, iv_clean = self.cleaned_data[i]
                if len(iv_clean) == 0: continue
                
                # Basis matrices
                D_m = BSpline.design_matrix(m_clean, self.t_m, self.order_moneyness - 1).toarray()
                D_t = BSpline.design_matrix(tau_clean, self.t_t, self.order_tau - 1).toarray()
                
                # Row-wise Kronecker product using einsum to build basis design matrix
                X_i = np.einsum('ja,jb->jab', D_m, D_t).reshape(len(iv_clean), -1)
                
                alpha_i = self.scoreMat[i, :]
                curr_alpha = alpha_i[-1]
                
                # Residuals: iv_clean - sum_{m < J} alpha_{im} * psi_m
                prev_fit = np.zeros(len(iv_clean))
                for k in range(len(self.betaList)):
                    prev_fit += alpha_i[k] * self.evaluator(self.betaList[k], np.column_stack([m_clean, tau_clean]))
                
                resid_i = iv_clean - prev_fit
                
                sum_XX += (curr_alpha**2) * (X_i.T @ X_i)
                sum_Xy += curr_alpha * (X_i.T @ resid_i)
            
            beta_flat = np.linalg.solve(sum_XX, sum_Xy)
            beta = beta_flat.reshape(self.nb_spline_moneyness, self.nb_spline_tau)
            
            # Rescale the beta s.t. 2-norm of eigen function is 1
            norm = self.compute_norm(beta)
            beta = beta / norm
            j += 1
            
            maxBetaChange = np.max(np.abs(beta - old_Beta))
            old_Beta = beta
            
        self.betaList.append(beta)
        return self.scoreMat[:, -1].tolist(), beta

    def reconstruct_surface(self, coords, betaList, scores):
        """
        Reconstructs the IV surface at given coordinates using provided betas and scores.
        
        Parameters:
        - coords: 2D array of shape (n, 2)
        - betaList: List of 2D beta matrices
        - scores: 1D array of scores corresponding to each beta
        
        Returns:
        - reconstructed_iv: 1D array of length n
        """
        reconstructed_iv = np.zeros(len(coords))
        for k in range(len(betaList)):
            reconstructed_iv += scores[k] * self.evaluator(betaList[k], coords)
        return reconstructed_iv

    def compute_explained_variance(self):
        """
        Computes the proportion of variance explained by each FPC using reconstruction.
        
        Returns:
        - prop_var_marginal: List of marginal proportion of variance explained by each FPC.
        """
        num_fpcs = len(self.betaList)
        total_tss = 0
        cumulative_rss = np.zeros(num_fpcs)
        
        for i in range(len(self.cleaned_data)):
            m_clean, tau_clean, iv_clean = self.cleaned_data[i]
            if len(iv_clean) == 0: continue
            
            coords = np.column_stack([m_clean, tau_clean])
            total_tss += np.sum(iv_clean**2)
            
            for k in range(1, num_fpcs + 1):
                # Using first k components
                y_hat = self.reconstruct_surface(coords, self.betaList[:k], self.scoreMat[i, :k])
                cumulative_rss[k-1] += np.sum((iv_clean - y_hat)**2)
                
        # Cumulative proportion of variance explained
        prop_var_cum = 1 - (cumulative_rss / total_tss)
        
        # Marginal proportion of variance explained
        prop_var_marginal = np.zeros(num_fpcs)
        prop_var_marginal[0] = prop_var_cum[0]
        if num_fpcs > 1:
            prop_var_marginal[1:] = np.diff(prop_var_cum)
            
        return prop_var_marginal.tolist()
    
    def _compute_gram_matrix(self, t, order):
        """
        Computes the Gram matrix (inner products) of B-spline basis functions.
        """
        n_basis = len(t) - order
        W = np.zeros((n_basis, n_basis))
        
        # Quadrature points for degree 2k
        quad_points, quad_weights = roots_legendre(order)
        
        # Unique knots
        uknots = np.unique(t)
        
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

    def compute_inner_product(self, beta1, beta2):
        """
        Computes the L2 inner product between two 2D eigenfunctions defined by coefficients beta1 and beta2.
        <psi1, psi2> = Tr(B1.T @ W_m @ B2 @ W_t)
        """
        inner_prod = np.trace(beta1.T @ self.W_m @ beta2 @ self.W_t)
        return inner_prod

    def compute_norm(self, beta):
        """
        Computes the L2 norm of the 2D eigenfunction defined by coefficients beta.
        ||psi||^2 = <psi, psi>
        """
        return np.sqrt(self.compute_inner_product(beta, beta))
    
    def BSplines_2d(self, range_moneyness=[-.15, .1], range_tau=[0, 1],
                    nb_spline_moneyness=10, nb_spline_tau=10,
                    order_moneyness=4, order_tau=4):
        """
        order_moneyness, order_tau are the order of the spline used in both dimensions respectively while k_m and k_tau are the respective degrees
        """
        k_m = order_moneyness - 1
        k_t = order_tau - 1
        
        # Generate clamped knot vectors
        t_breakPts_m = np.linspace(range_moneyness[0], range_moneyness[1], nb_spline_moneyness - k_m + 1) #all breakPts 
        self.t_m = np.concatenate(([range_moneyness[0]] * k_m, t_breakPts_m, [range_moneyness[1]] * k_m)) #all knots (k are added at each extremities)
        
        ### Here if we used uniformly space knots, should probably use sqrt(tau) so observations are more 
        ### uniformly ditributed between the knots. Otherwise we can concentrate the knots around smaller 
        ### tau where there's more curvature and more observations. (Here I use **2, but could also be data 
        ### driven based on the quantiles)
        t_breakPts_t = np.linspace(range_tau[0], range_tau[1], nb_spline_tau - k_t + 1)**2  #all breakPts 
        ## only needed if range tau doesn't start at 0 and want to use **2
        #t_breakPts_t = (((np.linspace(range_tau[0], range_tau[1], nb_spline_tau - k_t + 1)-range_tau[0])/(range_tau[1]-range_tau[0]))**2)*(range_tau[1]-range_tau[0])+range_tau[0]              #all breakPts 
        ##
        self.t_t = np.concatenate(([range_tau[0]] * k_t, t_breakPts_t, [range_tau[1]] * k_t))           #all knots (k are added at each extremities)
        
        # Pre-compute Gram matrices for normalization
        self.W_m = self._compute_gram_matrix(self.t_m, order_moneyness)
        self.W_t = self._compute_gram_matrix(self.t_t, order_tau)
        
        def evaluator(beta, coords):
            """
            Evaluates the tensor product B-spline surface.
            
            Parameters:
            - beta: 2D array of dimension (nb_spline_moneyness X nb_spline_tau)
            - coords: 2D array of shape (n, 2), where column 0 is moneyness and column 1 is tau.
            
            Returns:
            - 1D array of length n containing the evaluated IVs.
            """
            # Ensure coordinates are within the knot boundaries to avoid extrapolation errors
            x_m = np.clip(np.asarray(coords[:, 0]), range_moneyness[0], range_moneyness[1])
            x_t = np.clip(np.asarray(coords[:, 1]), range_tau[0], range_tau[1])
            
            # Compute the sparse design matrices for both dimensions
            D_m = BSpline.design_matrix(x_m, self.t_m, order_moneyness - 1)
            D_t = BSpline.design_matrix(x_t, self.t_t, order_tau - 1)
            
            # Compute tensor product evaluation
            res = np.sum((D_m @ beta) * D_t.toarray(), axis=1)
            
            return res
            
        self.evaluator = evaluator
        
    def plot_eigen_functions(self, beta, num_points=30, figAngle=-70):
        """
        Plots the B-spline surface given a coefficient matrix beta.
        
        Parameters:
        - beta: 2D array of dimension (nb_spline_moneyness X nb_spline_tau)
        - num_points: Number of points along each dimension for the grid.
        - figAngle: Viewing angle (azimuth).
        """
        m_vals = np.linspace(self.range_moneyness[0], self.range_moneyness[1], num_points)
        tau_vals = np.linspace(self.range_tau[0], self.range_tau[1], num_points)
        M, Tau = np.meshgrid(m_vals, tau_vals)
        
        coords = np.column_stack([M.flatten(), Tau.flatten()])
        
        Z_flat = self.evaluator(beta, coords)
        Z = Z_flat.reshape(M.shape)
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot_surface(M, Tau, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        ax.set_title("B-Spline Evaluated Surface")
        ax.set_xlabel("Moneyness (m)")
        ax.set_ylabel("Time to Expiry (tau)")
        
        ax.view_init(elev=15, azim=figAngle)
        plt.show()

    def plot_reconstruction(self, index, num_points=30, figAngle=-70):
        """
        Plots the reconstructed IVS for a given day and overlaps the actual observations.
        
        Parameters:
        - index: Index of the day to plot.
        - num_points: Number of points along each dimension for the grid.
        - figAngle: Viewing angle (azimuth).
        """
        m_clean, tau_clean, iv_clean = self.cleaned_data[index]
        scores = self.scoreMat[index, :]
        
        # Grid for surface
        m_vals = np.linspace(self.range_moneyness[0], self.range_moneyness[1], num_points)
        tau_vals = np.linspace(self.range_tau[0], self.range_tau[1], num_points)
        M, Tau = np.meshgrid(m_vals, tau_vals)
        coords_grid = np.column_stack([M.flatten(), Tau.flatten()])
        
        # Reconstruct on grid
        Z_flat = self.reconstruct_surface(coords_grid, self.betaList, scores)
        Z = Z_flat.reshape(M.shape)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot reconstructed surface
        ax.plot_surface(M, Tau, Z, cmap='viridis', edgecolor='none', alpha=0.5)
        
        # Plot actual observations
        ax.scatter(m_clean, tau_clean, iv_clean, color='red', s=10, label='Actual Observations')
        
        ax.set_title(f"Reconstructed IVS vs Actual Observations (Day {index})")
        ax.set_xlabel("Moneyness (m)")
        ax.set_ylabel("Time to Expiry (tau)")
        ax.set_zlabel("Implied Volatility")
        
        ax.view_init(elev=15, azim=figAngle)
        plt.legend()
        plt.show()

    def compute_arbitrage_metrics(self, scores, m_grid=None, tau_grid=None):
        """
        Computes calendar and butterfly arbitrage metrics on a grid.
        
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
            # Use a small epsilon to avoid tau=0 which causes division by zero in butterfly metric
            tau_min = max(1e-4, self.range_tau[0])
            tau_grid = np.linspace(tau_min, self.range_tau[1], 50)
            
        M, T = np.meshgrid(m_grid, tau_grid)
        coords = np.column_stack([M.flatten(), T.flatten()])
        
        # iv is sigma(m, tau)
        iv_flat = self.reconstruct_surface(coords, self.betaList, scores)
        iv = iv_flat.reshape(M.shape)
        
        # w(m, tau) = iv^2 * tau
        w = (iv**2) * T
        
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
        # Avoid tau=0 for butterfly metric division
        tau_min = max(1e-4, self.range_tau[0])
        tau_vals = np.linspace(tau_min, self.range_tau[1], num_points)
        
        calendar_metrics, butterfly_metrics = self.compute_arbitrage_metrics(scores, m_grid=m_vals, tau_grid=tau_vals)
        
        calendar_metrics = np.minimum(0,calendar_metrics)
        butterfly_metrics = np.minimum(0,butterfly_metrics)
        
        M, T = np.meshgrid(m_vals, tau_vals)
        
        fig = plt.figure(figsize=(16, 7))
        
        # Plot Calendar Spread violations
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(M, T, calendar_metrics, cmap='RdYlGn', edgecolor='none', alpha=0.8)
        # Add a zero-plane for reference
        ax1.plot_surface(M, T, np.zeros_like(M), color='black', alpha=0.2)
        ax1.set_title(f"Calendar Spread Metric (d_tau w) - Day {index}\n(Violation if < 0)")
        ax1.set_xlabel("Moneyness (m)")
        ax1.set_ylabel("Time to Expiry (tau)")
        ax1.view_init(elev=15, azim=figAngle)
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

        # Plot Butterfly violations
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(M, T, butterfly_metrics, cmap='RdYlGn', edgecolor='none', alpha=0.8)
        # Add a zero-plane for reference
        ax2.plot_surface(M, T, np.zeros_like(M), color='black', alpha=0.2)
        ax2.set_title(f"Butterfly Metric (g(m)) - Day {index}\n(Violation if < 0)")
        ax2.set_xlabel("Moneyness (m)")
        ax2.set_ylabel("Time to Expiry (tau)")
        ax2.view_init(elev=15, azim=figAngle)
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
        
        plt.tight_layout()
        plt.show()

#%%
if __name__ == '__main__':
    
    with open("/Users/macbook/Documents/global_O_Research/O_Research/data/SPX_data/SPX_lists.pkl", "rb") as f:
        uniqueDates = pickle.load(f)
        tau = pickle.load(f)
        moneyness = pickle.load(f)
        iv = pickle.load(f)
        S = pickle.load(f)
        rfRate = pickle.load(f)
        dividendRate = pickle.load(f)
        
    # #select small sample to test with
    # uniqueDates = uniqueDates[0:100]
    # tau = tau[0:100]
    # moneyness = moneyness[0:100]
    # iv = iv[0:100]
    # S = S[0:100]
    # rfRate = rfRate[0:100]
    # dividendRate = dividendRate[0:100]
    
    logMoneyness = [np.log(m) for m in moneyness]
    sqrtTau = [np.sqrt(t) for t in tau]             #Should probably work with sqrt(tau\) if we use uniformly spaced knots in the B-spline
    
    flattenIV = [v for vDay in iv for v in vDay]
    meanIV = np.mean(flattenIV)
    ivCentered = [v-meanIV for v in iv]
    
    ivLog = [np.log(v) for v in iv]
    
    #%% Estimate the First few FPCs
    fpca = FPCA(logMoneyness, tau, iv, nb_spline_moneyness = 10, nb_spline_tau = 12, order_moneyness = 4, order_tau = 4)
    alpha1, beta1 = fpca.first_FPC_fit(maxit=20)
    fpca.plot_eigen_functions(beta1, num_points=50, figAngle=-70)
    
    alpha2, beta2 = fpca.subsequent_FPC_fit(maxit=30)
    fpca.plot_eigen_functions(beta2, num_points=50, figAngle=-70)
    
    alpha3, beta3 = fpca.subsequent_FPC_fit(maxit=30)
    fpca.plot_eigen_functions(beta3, num_points=50, figAngle=-70)
    
    alpha4, beta4 = fpca.subsequent_FPC_fit(maxit=30)
    fpca.plot_eigen_functions(beta4, num_points=50, figAngle=-70)
    
    fpca.compute_explained_variance()

    with open("/Users/macbook/Documents/global_O_Research/O_Research/data/SPX_data/FPCA_Approx_logM_tau_iv.pkl", "wb") as f:
        pickle.dump(fpca.scoreMat, f)
        pickle.dump(fpca.betaList, f)

    #%% Load Basis representation fit
    fpca = FPCA(logMoneyness, tau, iv, nb_spline_moneyness = 10, nb_spline_tau = 12, order_moneyness = 4, order_tau = 4)
    with open("/Users/macbook/Documents/global_O_Research/O_Research/data/SPX_data/FPCA_Approx_logM_tau_iv.pkl", "rb") as f:
        fpca.scoreMat = pickle.load(f)
        fpca.betaList = pickle.load(f)
        
    #%% Measure Static Arbitrage
    nbDays = len(fpca.cleaned_data)
    nbCalendar = np.zeros(nbDays)
    nbButterfly = np.zeros(nbDays)
    for i in range(nbDays):
        day_scores = fpca.scoreMat[i, :]
        calendar_metrics, butterfly_metrics = fpca.compute_arbitrage_metrics(day_scores)
        nbCalendar[i] = np.sum(calendar_metrics < 0)
        nbButterfly[i] = np.sum(butterfly_metrics < 0)
    
    plt.plot(nbCalendar)  
    plt.plot(nbButterfly)
    
    # Plot violations for the first day
    fpca.plot_arbitrage_violations(0)
        
        
    #%% (NOT GOOD) Estimate the First few FPCs where they are not orthogonal with each other (basically simpy LeastSquare Basis)
    fpcaNotOrtho = FPCA(logMoneyness, sqrtTau, iv, nb_spline_moneyness = 10, nb_spline_tau = 12, order_moneyness = 4, order_tau = 4)
    _, beta1 = fpcaNotOrtho.first_FPC_fit(maxit=20)
    fpcaNotOrtho.plot_eigen_functions(beta1, num_points=50, figAngle=-70)
    
    alpha2, beta2 = fpcaNotOrtho.subsequent_FPC_fit_NotOrthogonal(maxit=30)
    fpcaNotOrtho.plot_eigen_functions(beta2, num_points=50, figAngle=-70)
    
    alpha3, beta3 = fpcaNotOrtho.subsequent_FPC_fit_NotOrthogonal(maxit=30)
    fpcaNotOrtho.plot_eigen_functions(beta3, num_points=50, figAngle=-70)
    
# %%
