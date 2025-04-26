import numpy as np

def cf_Bates(u, S0, r, q, V0, kappa, eta, theta, rho, T, jump_intensity, jump_mean, jump_stddev):
    """
    Characteristic function for the Bates model.

    Parameters:
        u (complex): Fourier transform variable.
        S0 (float): Initial stock price.
        r (float): Risk-free rate.
        q (float): Dividend yield.
        V0 (float): Initial variance.
        kappa (float): Rate of mean reversion.
        eta (float): Long-term mean of volatility.
        theta (float): Volatility of volatility.
        rho (float): Correlation between stock price and volatility.
        T (float): Time to maturity.
        jump_intensity (float): Intensity of jumps.
        jump_mean (float): Mean of jump size.
        jump_stddev (float): Standard deviation of jump size.

    Returns:
        complex: Value of the characteristic function.
    """
    # Parameters for the jump component
    lambda_ = jump_intensity
    mu_J = jump_mean
    sigma_J = jump_stddev

    # Adjusted drift term
    drift = r - q - lambda_ * (np.exp(mu_J + 0.5 * sigma_J**2) - 1)

    # Heston model components
    d = np.sqrt((rho * theta * u * 1j - kappa)**2 + (theta**2) * (u * 1j + u**2))
    g = (kappa - rho * theta * u * 1j - d) / (kappa - rho * theta * u * 1j + d)

    C = (r - q) * u * 1j * T + (kappa * eta / theta**2) * \
        ((kappa - rho * theta * u * 1j - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))

    D = ((kappa - rho * theta * u * 1j - d) / theta**2) * \
        (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))

    # Jump component
    jump_component = lambda_ * T * (np.exp(u * 1j * mu_J - 0.5 * (u**2) * sigma_J**2) - 1)

    # Combine components
    phi = np.exp(C + D * V0 + u * 1j * np.log(S0) + jump_component)

    return phi