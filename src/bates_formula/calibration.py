import numpy as np
from scipy.optimize import minimize
from .bates_model import BatesModel

def calibrate_bates_model(market_prices, strikes, S, T, r, q):
    """
    Calibrate the Bates model to market data by minimizing the sum of squared errors.

    Parameters:
    - market_prices: List of market option prices.
    - strikes: List of strike prices.
    - S: Spot price.
    - T: Time to maturity.
    - r: Risk-free rate.
    - q: Dividend yield.

    Returns:
    - A dictionary of calibrated Bates model parameters.
    """
    def bates_objective(params):
        V0, kappa, eta, theta, rho, jump_intensity, jump_mean, jump_stddev = params
        bates = BatesModel(S, r, q, V0, kappa, eta, theta, rho, T, strikes=strikes,
                           jump_intensity=jump_intensity, jump_mean=jump_mean, jump_stddev=jump_stddev)
        model_prices = bates.price_options()
        return np.sum((np.array(model_prices) - np.array(market_prices))**2)

    # Initial guess for Bates model parameters
    initial_params = [0.05, 0.5, 0.05, 0.2, -0.75, 0.1, 0.0, 0.2]
    bounds = [
        (0.01, 0.5),  # V0
        (0.1, 2),     # kappa
        (0.01, 0.5),  # eta
        (0.1, 0.5),   # theta
        (-0.99, 0.99),# rho
        (0.01, 1),    # jump_intensity
        (-0.5, 0.5),  # jump_mean
        (0.01, 0.5)   # jump_stddev
    ]

    # Minimize the objective function
    result = minimize(bates_objective, initial_params, bounds=bounds)

    # Extract calibrated parameters
    calibrated_params = result.x

    return {
        "V0": calibrated_params[0],
        "kappa": calibrated_params[1],
        "eta": calibrated_params[2],
        "theta": calibrated_params[3],
        "rho": calibrated_params[4],
        "jump_intensity": calibrated_params[5],
        "jump_mean": calibrated_params[6],
        "jump_stddev": calibrated_params[7]
    }