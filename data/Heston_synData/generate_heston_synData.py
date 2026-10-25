"""
Synthetic implied volatility surface (IVS) data set generated from the Heston (1993) model.

The generator mimics the irregular and sparse observation design of the traded DJX data
(`data/DJX_data/DJX_lists_traded_*.pkl`) while providing the noise-free ground truth:

    1. Latent daily paths of the spot S_t, the instantaneous variance v_t, the short rate r_t
       (Vasicek) and, optionally, a drifting long-run variance theta_t are simulated under P
       (full-truncation Euler with intra-day sub-steps, calendar time = days / 365).
    2. Each day, a realistic expiry calendar is listed (weekly Fridays, monthly third Fridays,
       quarterly third Fridays) together with a fixed-increment strike grid whose width grows
       with sqrt(tau). A random, liquidity-weighted subset of maturities and strikes is "traded".
    3. Out-of-the-money option prices are computed under Q with the Lewis (2001) single-integral
       formula using a Black-Scholes control variate, integrated with Gauss-Legendre quadrature.
    4. Prices are inverted to Black-Scholes implied volatilities (vectorized bisection). Quotes
       below a minimum price are discarded (they would not trade).

Purpose: an ARBITRAGE-FREE benchmark. By default the observed IVs are the exact Heston IVs (no
observation noise) and rates/dividends are zero (carry='zero_rates'), so that the forward equals the
spot and ln(K/S) = ln(K/F). Under this setting the static no-arbitrage conditions used by
FPCA_penalized_arbPenal (monotone total variance in tau at fixed ln(K/S), Gatheral's g(m) >= 0, and the
price-based vertical / butterfly / calendar checks at fixed strike) are exact, and the generated data
satisfy all of them (see check_static_arbitrage_grid and the verification cell in __main__).
Optional bid-ask noise (design['noise_scale'] > 0) and non-zero rates (carry='zero_carry' or
'stochastic') are kept for robustness experiments; with carry='stochastic' (r != q) the IV-space
conditions at fixed ln(K/S) are no longer exact no-arbitrage conditions.

Observations are restricted to the domain of the FPCA model, ln(K/S) in [-0.15, 0.15] and tau in (0, 1],
and the trading calendar (hence the train/test split) is taken from the DJX / SPX lists.

Output (same layout as the SPX / DJX `*_lists_*.pkl`, i.e. 7 objects pickled sequentially):
    uniqueDates (object np.ndarray of str), tau (list of pd.Series 'tte', years), moneyness (list of pd.Series, K/S),
    iv (list of pd.Series 'impl_volatility_computed', noisy), S (list of np.float64),
    rfRate (list of pd.Series), dividendRate (list of pd.Series)

plus the ground truth: noise-free IVs and prices per observation (CSV), the latent daily state, the
true surface on a fixed (log-moneyness, tau) grid for every day, and the generating parameters.

Pure Heston with constant Q-parameters (theta_vol = 0) implies that the whole surface on a given day
is a deterministic function of the single state v_t (for fixed rates): a 1-state, non-linear benchmark.
Setting theta_vol > 0 lets the long-run variance drift day to day ("parameter-drifting" Heston),
which adds a second latent driver of the term structure.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr, roots_legendre


class HestonSyntheticIVS:
    """
    Simulate an irregularly and sparsely observed Heston implied volatility surface panel.

    Parameters
    ----------
    kappa, theta, sigma, rho : float
        Heston risk-neutral parameters (mean-reversion speed, long-run variance, vol-of-vol, correlation).
    v0 : float
        Initial instantaneous variance.
    kappa_P, theta_P : float or None
        Variance dynamics under P (default: same as Q, i.e. no variance risk premium).
    mu : float
        Drift of the spot under P.
    S0 : float
        Initial spot.
    carry : {'zero_rates', 'zero_carry', 'stochastic'}
        'zero_rates': r = q = 0 (forward = spot, no discounting; all arbitrage checks are exact).
        'zero_carry': Vasicek r and q(tau) = r(tau) (forward = spot, but prices are discounted).
        'stochastic': Vasicek r and constant dividend yield q.
    q : float
        Continuous dividend yield (constant), used only with carry='stochastic'.
    r0, r_mean, r_speed, r_vol : float
        Vasicek short-rate dynamics (ignored with carry='zero_rates'); r_slope adds a deterministic
        term-structure slope r(tau) = r_t + r_slope * (1 - exp(-tau / 0.5)).
    theta_vol, theta_speed : float
        Volatility and mean-reversion speed of the OU process driving log(theta_t / theta).
        theta_vol = 0 recovers the pure (constant-parameter) Heston model.
    seed : int
        Seed of the random generator (the whole data set is reproducible).
    """

    def __init__(self, kappa=3.0, theta=0.04, sigma=0.45, rho=-0.7, v0=0.03,
                 kappa_P=None, theta_P=None, mu=0.07, S0=100.0, carry="zero_rates", q=0.018,
                 r0=0.01, r_mean=0.025, r_speed=0.3, r_vol=0.006, r_slope=0.004,
                 theta_vol=0.0, theta_speed=1.0, seed=20261025):
        if carry not in ("zero_rates", "zero_carry", "stochastic"):
            raise ValueError("carry must be 'zero_rates', 'zero_carry' or 'stochastic'")
        self.kappa, self.theta, self.sigma, self.rho, self.v0 = kappa, theta, sigma, rho, v0
        self.kappa_P = kappa if kappa_P is None else kappa_P
        self.theta_P = theta if theta_P is None else theta_P
        self.mu, self.S0, self.carry, self.q = mu, S0, carry, q
        self.r0, self.r_mean, self.r_speed, self.r_vol, self.r_slope = r0, r_mean, r_speed, r_vol, r_slope
        self.theta_vol, self.theta_speed = theta_vol, theta_speed
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Numerical settings of the pricer
        self.n_quad = 1024         # Gauss-Legendre nodes for the Lewis integral
        self.trunc_level = 40.0    # integrand truncated where the Gaussian envelope reaches exp(-trunc_level)
        self.v_floor = 1e-4        # floor on the variance used for pricing
        gl_x, gl_w = roots_legendre(self.n_quad)
        self._gl_x, self._gl_w = 0.5 * (gl_x + 1.0), 0.5 * gl_w   # mapped to [0, 1]

    # ------------------------------------------------------------------------------------------
    # Heston pricing (risk-neutral)
    # ------------------------------------------------------------------------------------------
    def _heston_cf(self, w, T, v0, theta):
        """Characteristic function of ln(S_T / F_T) at complex argument w ("little trap" form)."""
        kappa, sigma, rho = self.kappa, self.sigma, self.rho
        iw = 1j * w
        beta = kappa - rho * sigma * iw
        d = np.sqrt(beta ** 2 + sigma ** 2 * (iw + w ** 2))
        g = (beta - d) / (beta + d)
        edt = np.exp(-d * T)
        C = kappa * theta / sigma ** 2 * ((beta - d) * T - 2.0 * np.log((1.0 - g * edt) / (1.0 - g)))
        D = (beta - d) / sigma ** 2 * (1.0 - edt) / (1.0 - g * edt)
        return np.exp(C + D * v0)

    @staticmethod
    def _bs_otm_norm(k, s):
        """Black-Scholes OTM price normalized by the forward (undiscounted); k = ln(K/F), s = sigma*sqrt(T)."""
        s = np.maximum(s, 1e-300)
        d1 = -k / s + 0.5 * s
        d2 = d1 - s
        call = ndtr(d1) - np.exp(k) * ndtr(d2)
        put = np.exp(k) * ndtr(-d2) - ndtr(-d1)
        return np.where(k >= 0, call, put)

    @staticmethod
    def _bs_vega_norm(k, s, T):
        """Black-Scholes vega (w.r.t. sigma) normalized by the forward (undiscounted)."""
        d1 = -k / s + 0.5 * s
        return np.exp(-0.5 * d1 ** 2) / np.sqrt(2.0 * np.pi) * np.sqrt(T)

    def heston_otm_price_norm(self, k, T, v0, theta=None):
        """
        Heston OTM option price (calls for K >= F, puts for K < F), undiscounted and normalized by F.

        Lewis (2001) formula with a Black-Scholes control variate whose variance equals the expected
        average variance over [0, T]; this removes the cancellation that plagues deep OTM / short-dated
        quotes and makes the integrand decay fast.
        """
        theta = self.theta if theta is None else theta
        k = np.atleast_1d(np.asarray(k, dtype=float))
        v0 = max(v0, self.v_floor)

        # Control variate total volatility
        kT = self.kappa * T
        avg_var = theta + (v0 - theta) * (1.0 - np.exp(-kT)) / kT
        s_cv = np.sqrt(avg_var * T)

        # Truncation of the integral: the CF decays like a Gaussian for small u (short maturities) and
        # only exponentially for large u, |phi(u)| ~ exp(-u sqrt(1-rho^2) (v0 + kappa theta T) / sigma)
        v_lo = max(min(v0, avg_var), self.v_floor)
        U_gauss = np.sqrt(2.0 * self.trunc_level / (v_lo * T))
        U_exp = self.trunc_level * self.sigma / (np.sqrt(1.0 - self.rho ** 2) * (v0 + self.kappa * theta * T))
        U = max(U_gauss, U_exp)
        u = U * self._gl_x
        wts = U * self._gl_w

        phi_h = self._heston_cf(u - 0.5j, T, v0, theta)
        phi_bs = np.exp(-0.5 * s_cv ** 2 * (u ** 2 + 0.25))
        integrand = np.real(np.exp(-1j * np.outer(k, u)) * (phi_h - phi_bs)[None, :]) / (u ** 2 + 0.25)[None, :]
        delta = -np.exp(0.5 * k) / np.pi * (integrand @ wts)
        return self._bs_otm_norm(k, s_cv) + delta

    def implied_vol(self, p_norm, k, T, n_iter=100):
        """Vectorized bisection inversion of normalized OTM prices to Black-Scholes implied volatilities."""
        p_norm, k = np.asarray(p_norm, dtype=float), np.asarray(k, dtype=float)
        lo = np.full_like(p_norm, 1e-8)
        hi = np.full_like(p_norm, 5.0 * np.sqrt(np.maximum(T, 1e-12)))
        hi = np.maximum(hi, 3.0)
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            too_high = self._bs_otm_norm(k, mid) > p_norm
            hi = np.where(too_high, mid, hi)
            lo = np.where(too_high, lo, mid)
        iv = 0.5 * (lo + hi) / np.sqrt(T)
        upper_bound = np.where(k >= 0, 1.0, np.exp(k))   # OTM price must lie in (0, min(1, e^k))
        return np.where((p_norm > 0) & (p_norm < upper_bound), iv, np.nan)

    # ------------------------------------------------------------------------------------------
    # Latent state simulation (P-measure)
    # ------------------------------------------------------------------------------------------
    def _build_calendar(self, start_date, n_days):
        return pd.bdate_range(start=start_date, periods=n_days)

    def _simulate_latent_paths(self, dates, n_sub=20):
        """Full-truncation Euler for (ln S, v), exact OU/Vasicek steps for r and log theta."""
        n = len(dates)
        dt_days = np.diff(dates.values).astype("timedelta64[D]").astype(float) / 365.0
        rng = self.rng

        logS, v, r, y = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
        logS[0], v[0], y[0] = np.log(self.S0), self.v0, 0.0
        r[0] = 0.0 if self.carry == "zero_rates" else self.r0
        sq_rho = np.sqrt(1.0 - self.rho ** 2)

        for t in range(n - 1):
            dt = dt_days[t]
            h = dt / n_sub
            x_cur, v_cur = logS[t], v[t]
            Z = rng.standard_normal((n_sub, 2))
            q_t = self.dividend_curve(r[t], 0.0)
            for j in range(n_sub):
                v_pos = max(v_cur, 0.0)
                sv = np.sqrt(v_pos * h)
                x_cur += (self.mu - q_t - 0.5 * v_pos) * h + sv * Z[j, 0]
                v_cur += self.kappa_P * (self.theta_P - v_pos) * h \
                    + self.sigma * sv * (self.rho * Z[j, 0] + sq_rho * Z[j, 1])
            logS[t + 1], v[t + 1] = x_cur, v_cur

            e_r = np.exp(-self.r_speed * dt)
            r[t + 1] = self.r_mean + (r[t] - self.r_mean) * e_r \
                + self.r_vol * np.sqrt((1.0 - e_r ** 2) / (2.0 * self.r_speed)) * rng.standard_normal()
            if self.carry == "zero_rates":
                r[t + 1] = 0.0
            e_y = np.exp(-self.theta_speed * dt)
            y[t + 1] = y[t] * e_y \
                + self.theta_vol * np.sqrt((1.0 - e_y ** 2) / (2.0 * self.theta_speed)) * rng.standard_normal()

        return pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "S": np.exp(logS),
            "v": np.maximum(v, self.v_floor),
            "v_raw": v,
            "r_short": r,
            "q_short": self.dividend_curve(r, 0.0),
            "theta_t": self.theta * np.exp(y - 0.5 * self.theta_vol ** 2 / (2.0 * self.theta_speed)),
        })

    def rate_curve(self, r_short, tau):
        if self.carry == "zero_rates":
            return np.zeros_like(np.asarray(r_short + tau, dtype=float))
        return r_short + self.r_slope * (1.0 - np.exp(-np.asarray(tau) / 0.5))

    def dividend_curve(self, r_short, tau):
        if self.carry == "stochastic":
            return np.full_like(np.asarray(r_short + tau, dtype=float), self.q)
        return self.rate_curve(r_short, tau)          # zero carry: q(tau) = r(tau), forward = spot

    # ------------------------------------------------------------------------------------------
    # Observation design (which options are "traded" on a given day)
    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _build_expiry_calendar(start_date, end_date):
        """All Fridays plus a flag for monthly (third Friday) and quarterly expiries."""
        fridays = pd.date_range(start=start_date, end=end_date, freq="W-FRI")
        third = (fridays.day >= 15) & (fridays.day <= 21)
        quarterly = third & fridays.month.isin([3, 6, 9, 12])
        return pd.DataFrame({"expiry": fridays, "monthly": third, "quarterly": quarterly})

    def _list_expiries(self, date, expiries, design):
        """Expiries listed on `date`: weeklies up to 6 weeks, monthlies up to 9 months, quarterlies up to max_tau."""
        days = (expiries["expiry"] - date).dt.days.values
        tau = days / 365.0
        listed = (days >= 1) & (
            (tau <= 6.0 / 52.0)
            | (expiries["monthly"].values & (tau <= 0.75))
            | (expiries["quarterly"].values & (tau <= design["max_tau"]))
        )
        listed &= tau <= design["max_tau"]
        return expiries.loc[listed, "expiry"].values, tau[listed], expiries.loc[listed, "monthly"].values

    def _sample_maturities(self, tau_listed, monthly):
        """Liquidity decays with maturity; monthly expiries are more liquid than weeklies."""
        d = self.design
        p = d["p_maturity"] * np.exp(-tau_listed / d["tau_liquidity"]) * np.where(monthly, 1.0, d["weekly_discount"])
        keep = self.rng.random(len(tau_listed)) < np.clip(p, 0.0, 1.0)
        if not keep.any():                                  # always trade at least the front monthly
            idx = np.flatnonzero(monthly)
            keep[idx[0] if len(idx) else 0] = True
        return keep

    @staticmethod
    def _strike_increment(S, rel_step):
        """Exchange-like 'nice' strike increment closest to rel_step * S (keeps strike density stationary in K/S)."""
        nice = np.array([0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0])
        return nice[np.argmin(np.abs(np.log(nice / (rel_step * S))))]

    def _sample_strikes(self, S, F, tau, atm_vol, liquidity, put_boost):
        """
        Fixed-increment strike grid listed within |ln(K/S)| <= max_abs_logm, width ~ sqrt(tau);
        traded strikes follow a Gaussian liquidity profile in standardized moneyness.
        """
        d = self.design
        inc = self._strike_increment(S, d["strike_step_rel"])
        half_width = min(d["max_abs_logm"], d["n_sd_listed"] * atm_vol * np.sqrt(tau) + d["base_logm"])
        K_lo = np.ceil(S * np.exp(-half_width) / inc) * inc
        K_hi = np.floor(S * np.exp(half_width) / inc) * inc
        K = np.arange(K_lo, K_hi + 0.5 * inc, inc)
        K = K[np.abs(np.log(K / S)) <= d["max_abs_logm"]]
        if len(K) == 0:
            return K
        z = np.log(K / F) / (atm_vol * np.sqrt(tau))
        z_scale = np.where(z < 0, d["z_liquidity"] * put_boost, d["z_liquidity"])
        p = liquidity * np.exp(-0.5 * (z / z_scale) ** 2)
        return K[self.rng.random(len(K)) < np.clip(p, 0.0, 1.0)]

    def _add_observation_noise(self, iv_true, p_norm, k, tau, F):
        """Price-space bid-ask noise: half-spread = max(tick, rel. spread * price), mapped to vol via vega."""
        d = self.design
        if d["noise_scale"] == 0:
            return iv_true.copy()
        s = iv_true * np.sqrt(tau)
        half_spread = 0.5 * np.maximum(d["tick_rel"], d["rel_spread"] * p_norm)
        vega = self._bs_vega_norm(k, s, tau)
        eps = d["noise_scale"] * half_spread * self.rng.standard_normal(len(k)) / vega
        eps = np.clip(eps, -d["max_rel_noise"] * iv_true, d["max_rel_noise"] * iv_true)
        return iv_true + eps

    # ------------------------------------------------------------------------------------------
    # Main generation
    # ------------------------------------------------------------------------------------------
    def _price_day(self, date, state, expiries):
        """
        Generate all traded quotes of a single day. Day-level liquidity and put-wing tilt and
        maturity-level liquidity are random, so the observed locations change irregularly from day to day.
        """
        d = self.design
        S, v, r_short, theta_t = state["S"], state["v"], state["r_short"], state["theta_t"]
        exp_dates, tau_listed, monthly = self._list_expiries(date, expiries, d)
        if len(tau_listed) == 0:
            return []
        keep = self._sample_maturities(tau_listed, monthly)
        day_liquidity = d["p_strike"] * self.rng.lognormal(0.0, d["liquidity_dispersion"])
        put_boost = d["put_wing_boost"] * self.rng.lognormal(0.0, d["wing_dispersion"])

        rows = []
        for tau in tau_listed[keep]:
            r, q = self.rate_curve(r_short, tau), self.dividend_curve(r_short, tau)
            F = S * np.exp((r - q) * tau)
            atm_vol = np.sqrt(theta_t + (v - theta_t) * (1 - np.exp(-self.kappa * tau)) / (self.kappa * tau))
            liquidity = min(1.0, day_liquidity * self.rng.lognormal(0.0, d["maturity_dispersion"]))
            K = self._sample_strikes(S, F, tau, atm_vol, liquidity, put_boost)
            if len(K) == 0:
                continue
            k = np.log(K / F)
            p_norm = self.heston_otm_price_norm(k, tau, v, theta_t)
            traded = p_norm * F >= d["min_price_rel"] * S * np.exp(r * tau)
            if not traded.any():
                continue
            K, k, p_norm = K[traded], k[traded], p_norm[traded]
            iv_true = self.implied_vol(p_norm, k, tau)
            ok = np.isfinite(iv_true)
            K, k, p_norm, iv_true = K[ok], k[ok], p_norm[ok], iv_true[ok]
            iv_obs = self._add_observation_noise(iv_true, p_norm, k, tau, F)
            disc_F = F * np.exp(-r * tau)
            rows.append(pd.DataFrame({
                "date": state["date"],
                "tte": int(round(tau * 365)),
                "tau": tau,
                "strike_price": K,
                "assetPrices": S,
                "forward": F,
                "moneyness": K / S,
                "logMoneyness": np.log(K / S),
                "impl_volatility_computed": iv_obs,
                "impl_volatility_true": iv_true,
                "price_true": p_norm * disc_F,
                "midPrices": self._bs_otm_norm(k, iv_obs * np.sqrt(tau)) * disc_F,
                "rfRate": r,
                "dividendRate": q,
                "cp_flag": np.where(k >= 0, "C", "P"),
            }))
        return rows

    def simulate(self, dates=None, n_days=3020, start_date="2013-01-02", design=None):
        """
        Simulate the latent paths and the sparse daily option panel.

        dates : sequence of dates (e.g. the DJX/SPX trading days); if None, n_days business days from start_date.
        Returns the long-format DataFrame of all quotes (also stored in self.data) and the
        latent daily state (self.latent).
        """
        self.design = self.default_design() if design is None else {**self.default_design(), **design}
        dates = self._build_calendar(start_date, n_days) if dates is None else pd.DatetimeIndex(pd.to_datetime(dates))
        self.latent = self._simulate_latent_paths(dates)
        expiries = self._build_expiry_calendar(dates[0], dates[-1] + pd.Timedelta(days=int(370 * self.design["max_tau"]) + 40))

        rows = []
        for t, date in enumerate(dates):
            for _ in range(50):          # redraw the day's design until enough quotes trade
                day_rows = self._price_day(date, self.latent.iloc[t], expiries)
                if sum(len(r) for r in day_rows) >= self.design["min_quotes_per_day"]:
                    break
            rows.extend(day_rows)
        self.data = pd.concat(rows, ignore_index=True)
        return self.data, self.latent

    @staticmethod
    def default_design():
        return {
            "max_tau": 1.0,              # longest maturity (years); model domain tau in [0, 1]
            "max_abs_logm": 0.15,        # max |ln(K/S)| listed; model domain ln(K/S) in [-0.15, 0.15]
            "p_maturity": 0.90,         # traded probability of a (near) maturity
            "tau_liquidity": 0.3,        # maturity liquidity decay (years)
            "weekly_discount": 0.7,      # weeklies are less often traded than monthlies
            "strike_step_rel": 0.005,    # listed strike increment ~ 0.5% of S (rounded to a 'nice' value)
            "n_sd_listed": 3.0,          # listed width in standard deviations
            "base_logm": 0.03,           # minimum listed half-width in log-moneyness
            "p_strike": 0.15,           # base traded probability of an ATM strike
            "liquidity_dispersion": 0.5, # lognormal dispersion of the day liquidity
            "maturity_dispersion": 0.4,  # lognormal dispersion of the liquidity of each traded maturity
            "z_liquidity": 1.2,          # liquidity width in standardized moneyness
            "put_wing_boost": 1.4,       # OTM puts are traded further away from ATM ...
            "wing_dispersion": 0.25,     # ... by a random amount each day
            "min_quotes_per_day": 4,     # a day is redrawn until at least this many quotes trade
            "min_price_rel": 1e-4,       # options cheaper than 1e-4 * S do not trade
            "tick_rel": 5e-4,            # quoted bid-ask tick relative to S
            "rel_spread": 0.04,          # relative bid-ask spread
            "noise_scale": 0.0,          # std of the mid deviation in units of the half-spread (0 = exact Heston IVs)
            "max_rel_noise": 0.5,        # cap of |noise| relative to the true IV
        }

    # ------------------------------------------------------------------------------------------
    # Ground truth on a fixed grid
    # ------------------------------------------------------------------------------------------
    def true_surface_grid(self, logm_grid, tau_grid):
        """
        Noise-free implied volatility on a fixed (ln(K/S), tau) grid for every simulated day.

        Returns an array of shape (n_days, len(tau_grid), len(logm_grid)); NaN where the
        option is too cheap for a reliable inversion (deep OTM, short maturity).
        """
        out = np.full((len(self.latent), len(tau_grid), len(logm_grid)), np.nan)
        for t, state in self.latent.iterrows():
            for j, tau in enumerate(tau_grid):
                r, q = self.rate_curve(state["r_short"], tau), self.dividend_curve(state["r_short"], tau)
                k = logm_grid - (r - q) * tau               # ln(K/F) = ln(K/S) - (r - q) tau
                p_norm = self.heston_otm_price_norm(k, tau, state["v"], state["theta_t"])
                iv = self.implied_vol(p_norm, k, tau)
                out[t, j] = np.where(p_norm > 1e-7, iv, np.nan)
        return out

    @staticmethod
    def check_static_arbitrage_grid(iv_grid, logm_grid, tau_grid, tol=1e-10):
        """
        Static arbitrage diagnostics of IV surfaces on a fixed grid (exact when forward = spot):
            calendar : total variance w = sigma^2 tau non-decreasing in tau at fixed ln(K/S)
            butterfly: Gatheral's g(m) = (1 - m w_m / 2w)^2 - w_m^2 / 4 (1/w + 1/4) + w_mm / 2 >= 0
        iv_grid has shape (n_days, len(tau_grid), len(logm_grid)); NaN cells are ignored.
        Returns a dict with the number of checked cells, the number of violations and the extreme values.
        """
        w = iv_grid ** 2 * tau_grid[None, :, None]
        dw = np.diff(w, axis=1) / np.diff(tau_grid)[None, :, None]

        m, h = logm_grid, np.diff(logm_grid)
        w_m = (w[:, :, 2:] - w[:, :, :-2]) / (m[2:] - m[:-2])
        w_mm = 2.0 * ((w[:, :, 2:] - w[:, :, 1:-1]) / h[1:] - (w[:, :, 1:-1] - w[:, :, :-2]) / h[:-1]) / (h[1:] + h[:-1])
        wc, mc = w[:, :, 1:-1], m[1:-1]
        g = (1.0 - mc * w_m / (2.0 * wc)) ** 2 - w_m ** 2 / 4.0 * (1.0 / wc + 0.25) + w_mm / 2.0

        return {
            "calendar_cells": int(np.isfinite(dw).sum()),
            "calendar_violations": int(np.sum(dw < -tol)),
            "calendar_min_dw_dtau": float(np.nanmin(dw)),
            "butterfly_cells": int(np.isfinite(g).sum()),
            "butterfly_violations": int(np.sum(g < -tol)),
            "butterfly_min_g": float(np.nanmin(g)),
        }

    # ------------------------------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _dump_lists(df, path):
        """
        Store per-day lists in the exact layout of `SPX_lists_*.pkl` / `DJX_lists_traded_*.pkl`:
        uniqueDates (object ndarray of 'YYYY-MM-DD'), then per-day pd.Series (global row index) of
        tau [name 'tte', years], moneyness K/S [no name], IV [name 'impl_volatility_computed'],
        S (np.float64), rfRate [name 'rfRate'], dividendRate [name 'dividendRate'].
        """
        groups = [g for _, g in df.groupby("date", sort=True)]
        with open(path, "wb") as f:
            pickle.dump(np.array([g["date"].iloc[0] for g in groups], dtype=object), f)
            pickle.dump([g["tau"].rename("tte") for g in groups], f)
            pickle.dump([g["moneyness"].rename(None) for g in groups], f)
            pickle.dump([g["impl_volatility_computed"] for g in groups], f)
            pickle.dump([np.float64(g["assetPrices"].iloc[0]) for g in groups], f)
            pickle.dump([g["rfRate"] for g in groups], f)
            pickle.dump([g["dividendRate"] for g in groups], f)

    def save(self, out_dir, n_test_days=250, prefix="Heston", grid=None):
        """Save training/testing lists, the full quote panel, the latent state and the parameters."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dates = np.sort(self.data["date"].unique())
        test_dates = set(dates[-n_test_days:])
        is_test = self.data["date"].isin(test_dates)

        self.data.to_csv(out_dir / f"{prefix}_allData.csv", index=False)
        self.latent.to_csv(out_dir / f"{prefix}_latentState.csv", index=False)
        self._dump_lists(self.data, out_dir / f"{prefix}_lists.pkl")
        self._dump_lists(self.data[~is_test], out_dir / f"{prefix}_lists_training.pkl")
        self._dump_lists(self.data[is_test], out_dir / f"{prefix}_lists_testing.pkl")

        # true IVs aligned with the lists (same order) for noise-free evaluation
        groups = [g for _, g in self.data.groupby("date", sort=True)]
        with open(out_dir / f"{prefix}_lists_trueIV.pkl", "wb") as f:
            pickle.dump(np.array([g["date"].iloc[0] for g in groups], dtype=object), f)
            pickle.dump([g["impl_volatility_true"] for g in groups], f)

        if grid is not None:
            np.savez_compressed(out_dir / f"{prefix}_trueSurfaceGrid.npz", dates=self.latent["date"].values,
                                logMoneyness=grid["logm"], tau=grid["tau"], iv=grid["iv"].astype(np.float32))

        params = {
            "heston_Q": {"kappa": self.kappa, "theta": self.theta, "sigma": self.sigma, "rho": self.rho, "v0": self.v0},
            "heston_P": {"kappa_P": self.kappa_P, "theta_P": self.theta_P, "mu": self.mu, "S0": self.S0},
            "rates": {"carry": self.carry, "q": self.q, "r0": self.r0, "r_mean": self.r_mean, "r_speed": self.r_speed,
                      "r_vol": self.r_vol, "r_slope": self.r_slope},
            "theta_dynamics": {"theta_vol": self.theta_vol, "theta_speed": self.theta_speed},
            "design": self.design,
            "seed": self.seed,
            "n_days": int(len(dates)),
            "n_test_days": int(n_test_days),
            "train_period": [str(dates[0]), str(dates[-n_test_days - 1])],
            "test_period": [str(dates[-n_test_days]), str(dates[-1])],
        }
        with open(out_dir / f"{prefix}_params.json", "w") as f:
            json.dump(params, f, indent=2)


#%%
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    OUT_DIR = "/Users/macbook/Documents/global_O_Research/O_Research/data/Heston_synData"

    #%% Sanity check of the pricer against the Fang & Oosterlee (2008) reference value
    ###################################################################################################
    ref = HestonSyntheticIVS(kappa=1.5768, theta=0.0398, sigma=0.5751, rho=-0.5711, v0=0.0175)
    call_atm = 100.0 * ref.heston_otm_price_norm(0.0, 1.0, 0.0175)[0]
    print(f"ATM call (S=K=100, T=1, r=q=0): {call_atm:.9f}  (reference 5.785155450)")

    #%% Trading calendar: same days and train/test split as the DJX / SPX lists
    ###################################################################################################
    calendar = []
    for split in ["training", "testing"]:
        with open(f"/Users/macbook/Documents/global_O_Research/O_Research/data/DJX_data/DJX_lists_traded_{split}.pkl", "rb") as f:
            calendar.append(pickle.load(f))
    n_test_days = len(calendar[1])
    calendar = np.concatenate(calendar)

    #%% Simulate the latent paths and the sparse option panel
    ###################################################################################################
    gen = HestonSyntheticIVS(seed=20261025)
    data, latent = gen.simulate(dates=calendar)

    counts = data.groupby("date").size()
    print(f"days: {len(counts)} | quotes: {len(data)} | quotes/day min/median/max: "
          f"{counts.min()}/{counts.median():.0f}/{counts.max()}")
    print(f"maturities/day (median): {data.groupby('date')['tau'].nunique().median():.0f}")
    print(data[["moneyness", "tau", "impl_volatility_computed", "impl_volatility_true"]].describe(percentiles=[.05, .5, .95]))

    #%% True surface on a fixed grid (ground truth for reconstruction errors)
    ###################################################################################################
    logm_grid = np.linspace(-0.15, 0.15, 31)
    tau_grid = np.concatenate([np.array([7, 14, 21, 30, 45, 60, 75, 90]) / 365, np.linspace(0.3, 1.0, 15)])
    iv_grid = gen.true_surface_grid(logm_grid, tau_grid)

    #%% Verify the absence of static arbitrage
    ###################################################################################################
    # (i) IV-space conditions on the dense true grid (calendar on total variance, Gatheral's g)
    print(gen.check_static_arbitrage_grid(iv_grid, logm_grid, tau_grid))

    # (ii) price-based raw-data checks of FPCA_penalized_arbPenal on the observed (sparse) quotes
    import sys
    sys.path.insert(0, "/Users/macbook/Documents/global_O_Research/O_Research")
    from factorRepresentation_dayWeighted_penalizedEigenBasis_ArbPenality import FPCA_penalized_arbPenal

    groups = [g for _, g in data.groupby("date", sort=True)]
    fpca_check = FPCA_penalized_arbPenal([g["logMoneyness"] for g in groups], [g["tau"] for g in groups],
                                         [g["impl_volatility_computed"] for g in groups],
                                         S=[g["assetPrices"].iloc[0] for g in groups],
                                         r=[g["rfRate"].values for g in groups], q=[g["dividendRate"].values for g in groups])
    raw_arb = fpca_check.measure_all_raw_arbitrage(tolerance=1e-8)
    print(raw_arb[["calendar_violations", "vertical_violations", "butterfly_violations", "total_nb_observations"]].sum())

    #%% Save
    ###################################################################################################
    gen.save(OUT_DIR, n_test_days=n_test_days, prefix="Heston", grid={"logm": logm_grid, "tau": tau_grid, "iv": iv_grid})

    #%% Load (same pattern as the DJX data)
    ###################################################################################################
    with open(f"{OUT_DIR}/Heston_lists_training.pkl", "rb") as f:
        uniqueDates = pickle.load(f)
        tau = pickle.load(f)
        moneyness = pickle.load(f)
        iv = pickle.load(f)
        S = pickle.load(f)
        rfRate = pickle.load(f)
        dividendRate = pickle.load(f)
    logMoneyness = [np.log(m) for m in moneyness]

    #%% Diagnostics plots
    ###################################################################################################
    img_dir = Path(OUT_DIR) / "images"
    img_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(pd.to_datetime(latent["date"]), np.sqrt(latent["v"]), lw=0.8, label=r"$\sqrt{v_t}$")
    atm30 = iv_grid[:, 3, 15]
    axes[0].plot(pd.to_datetime(latent["date"]), atm30, lw=0.8, label="ATM 30d IV")
    axes[0].legend()
    axes[0].set_title("Latent volatility")
    axes[1].hist(counts.values, bins=40)
    axes[1].set_title("Quotes per day")
    axes[2].scatter(data["logMoneyness"], data["tau"], s=1, alpha=0.05)
    axes[2].set_xlabel(r"$\ln(K/S)$")
    axes[2].set_ylabel(r"$\tau$")
    axes[2].set_title("Observation locations (all days)")
    fig.tight_layout()
    fig.savefig(img_dir / "Heston_overview.png", dpi=150)

    day = 1000
    d_day = data[data["date"] == latent["date"].iloc[day]]
    M, T = np.meshgrid(logm_grid, tau_grid)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(M, T, iv_grid[day], alpha=0.4, cmap="viridis")
    ax.scatter(d_day["logMoneyness"], d_day["tau"], d_day["impl_volatility_computed"], c="r", s=10, label="observed quotes")
    ax.set_xlabel(r"$\ln(K/S)$")
    ax.set_ylabel(r"$\tau$")
    ax.set_zlabel("IV")
    ax.set_title(f"True Heston surface and observed quotes on {latent['date'].iloc[day]}")
    ax.view_init(elev=25, azim=-70)
    ax.legend()
    fig.savefig(img_dir / "Heston_sampleDay.png", dpi=150)
    plt.show()
