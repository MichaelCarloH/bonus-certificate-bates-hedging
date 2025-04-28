import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from .characteristic_function import cf_Bates
from datetime import datetime

class BatesModel:
    def __init__(self, 
                 S0, 
                 r, 
                 q, 
                 V0=0.05, 
                 kappa=0.5, 
                 eta=0.05, 
                 theta=0.2, 
                 rho=-0.75, 
                 T=1, 
                 alpha=1.5, 
                 N=4096, 
                 eta_cm=0.25, 
                 b=None, 
                 strikes=None,
                 jump_intensity=0.1, 
                 jump_mean=0.0, 
                 jump_stddev=0.2):
        """
        Initialize the Bates model with given or default parameters.

        Parameters:
            S0 (float): Initial stock price.
            r (float): Risk-free rate.
            q (float): Dividend yield.
            V0 (float): Initial variance (default = 0.05).
            kappa (float): Rate of mean reversion (default = 0.5).
            eta (float): Long-term mean of volatility (default = 0.05).
            theta (float): Volatility of volatility (default = 0.2).
            rho (float): Correlation between stock price and volatility (default = -0.75).
            T (float): Time to maturity (default = 1).
            alpha (float): Fourier transform parameter (default = 1.5).
            N (int): Number of grid points (default = 4096).
            eta_cm (float): Fourier grid spacing (default = 0.25).
            b (float): Upper limit for Fourier transform integration (default calculated if None).
            strikes (array-like): Specific strike prices for interpolation (default = None).
            jump_intensity (float): Intensity of jumps (default = 0.1).
            jump_mean (float): Mean of jump size (default = 0.0).
            jump_stddev (float): Standard deviation of jump size (default = 0.2).
        """
        # Store model parameters
        self.S0 = S0
        self.r = r
        self.q = q
        self.V0 = V0
        self.kappa = kappa
        self.eta = eta
        self.theta = theta
        self.rho = rho
        self.T = T
        self.alpha = alpha
        self.N = N
        self.eta_cm = eta_cm
        self.strikes = strikes
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_stddev = jump_stddev

        # Calculate lambda_ and b if not provided
        self.lambda_ = (2 * np.pi) / (N * eta_cm)
        self.b = b if b is not None else (N * self.lambda_) / 2

        # Calculate auxiliary variables (log strikes, v, u, rho_cm)
        self.calculate_auxiliary_variables()

    def calculate_auxiliary_variables(self):
        """Calculate log_strikes, v, u, and rho_cm based on model parameters."""
        # Define finer grid for log strikes
        self.log_strikes = np.arange(-self.b, self.b, self.lambda_)

        # Grid for the Fourier transform variable v
        self.v = np.arange(0, self.N * self.eta_cm, self.eta_cm)

        # Define u for Fourier transform
        self.u = self.v - (self.alpha + 1) * 1j

        # Define rho_cm using the characteristic function for Bates model
        self.rho_cm = np.exp(-self.r * self.T) * cf_Bates(self.u, self.S0, self.r, self.q, 
                                                           self.V0, self.kappa, self.eta, 
                                                           self.theta, self.rho, self.T, 
                                                           self.jump_intensity, self.jump_mean, 
                                                           self.jump_stddev) / \
                      (self.alpha**2 + self.alpha - self.v**2 + 1j * (2 * self.alpha + 1) * self.v)

    def compute_fft(self, rule="rectangular", simpson_weights=None):
        """Compute the FFT based on the specified integration rule."""
        if rule == "rectangular":
            return self.compute_fft_rectangular()
        elif rule == "simpson":
            return self.compute_fft_simpson(simpson_weights)
        else:
            raise ValueError(f"Unsupported rule: {rule}")

    def compute_fft_rectangular(self):
        """Computes option prices using FFT with the rectangular rule."""
        fft_result_rectangular = np.fft.fft(np.exp(1j * self.v * self.b) * self.rho_cm * self.eta_cm)

        # Extract real part
        a_rectangular = np.real(fft_result_rectangular)

        # Compute call option prices
        self.calls_rectangular = (1 / np.pi) * np.exp(-self.alpha * self.log_strikes) * a_rectangular

    def compute_fft_simpson(self, simpson_weights):
        """Computes option prices using FFT with Simpson's rule."""
        # Apply Simpson's Rule correction in the FFT computation
        simpson_int = simpson_weights

        # Perform FFT with the Simpson's Rule weights
        fft_result_simpson = np.fft.fft(np.exp(1j * self.v * self.b) * self.rho_cm * self.eta_cm * simpson_int)

        # Extract real part
        a_simpson = np.real(fft_result_simpson)

        # Compute call option prices
        self.calls_simpson = (1 / np.pi) * np.exp(-self.alpha * self.log_strikes) * a_simpson

    def interpolate_prices(self):
        """Interpolate option prices for specific strikes."""
        # Choose interpolation method based on available data
        if hasattr(self, 'calls_rectangular'):
            calls_to_use = self.calls_rectangular
        elif hasattr(self, 'calls_simpson'):
            calls_to_use = self.calls_simpson
        else:
            raise ValueError("No option prices computed. Run compute_fft first.")

        # Interpolation for the specific strikes
        spline = interp1d(np.exp(self.log_strikes), calls_to_use, kind='cubic', fill_value="extrapolate")
        return spline(self.strikes)

    def price_options(self, rule="rectangular", simpson_weights=None):
        """Price options based on selected integration rule."""
        self.compute_fft(rule, simpson_weights)
        return self.interpolate_prices()
    
    
    def price_put_options(self, rule="rectangular", simpson_weights=None):
        """Price put options using put-call parity."""
        # Calculate call prices first
        call_prices = self.price_options(rule, simpson_weights)

        # Ensure self.strikes is a NumPy array
        strikes = np.array(self.strikes)

        # Apply put-call parity to calculate put prices
        put_prices = call_prices - self.S0 * np.exp(-self.q * self.T) + strikes * np.exp(-self.r * self.T)
        return put_prices

    def calculate_greeks(self, rule="rectangular", simpson_weights=None):
        """Calculate option Greeks (Delta, Gamma, Vega, Theta, Rho)."""
        # Calculate base option prices
        base_prices = self.price_options(rule, simpson_weights)

        # Finite difference step size
        epsilon = 1e-5

        # Delta: Sensitivity to changes in the underlying price
        self.S0 += epsilon
        prices_up = self.price_options(rule, simpson_weights)
        self.S0 -= 2 * epsilon
        prices_down = self.price_options(rule, simpson_weights)
        self.S0 += epsilon  # Reset to original value
        delta = (prices_up - prices_down) / (2 * epsilon)

        # Gamma: Sensitivity of Delta to changes in the underlying price
        gamma = (prices_up - 2 * base_prices + prices_down) / (epsilon**2)

        # Vega: Sensitivity to changes in volatility
        self.V0 += epsilon
        prices_vega = self.price_options(rule, simpson_weights)
        self.V0 -= epsilon  # Reset to original value
        vega = (prices_vega - base_prices) / epsilon

        # Theta: Sensitivity to changes in time to maturity
        self.T -= epsilon
        prices_theta = self.price_options(rule, simpson_weights)
        self.T += epsilon  # Reset to original value
        theta = (prices_theta - base_prices) / epsilon

        # Rho: Sensitivity to changes in the risk-free rate
        self.r += epsilon
        prices_rho = self.price_options(rule, simpson_weights)
        self.r -= epsilon  # Reset to original value
        rho = (prices_rho - base_prices) / epsilon

        return {
            "Delta": delta,
            "Gamma": gamma,
            "Vega": vega,
            "Theta": theta,
            "Rho": rho
        }

    def calculate_pricing_error(self, market_prices, strikes, option_type="call", rule="rectangular", simpson_weights=None):
        """Calculate the total pricing error (sum of squared errors) for puts or calls."""
        if option_type == "call":
            model_prices = self.price_options(rule, simpson_weights)
        elif option_type == "put":
            model_prices = self.price_put_options(rule, simpson_weights)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

        # Calculate the sum of squared errors
        pricing_error = np.sum((np.array(model_prices) - np.array(market_prices))**2)
        return pricing_error

    def price_options_for_all_maturities(self, combined_data, start_date):
        """
        Price options for all strikes and maturities in the combined data.

        Parameters:
            combined_data (DataFrame): The combined data containing strikes, maturities, and market prices.
            start_date (datetime): The start date for calculating time to maturity.

        Returns:
            DataFrame: The input DataFrame with an additional column for Bates model prices.
        """
      

        # Filter out-of-the-money (OTM) options
        otm_calls = combined_data[(combined_data['Strike'] > self.S0) & (combined_data['Bid_Call'] >= 0)].copy()
        otm_puts = combined_data[(combined_data['Strike'] < self.S0) & (combined_data['Bid_Put'] >= 0)].copy()
        print("otm_calls: ", otm_calls.shape)
        print("otm_puts: ", otm_puts.shape)

        # Combine OTM calls and puts
        otm_options = pd.concat([otm_calls, otm_puts], ignore_index=True)

        # Calculate time to maturity for each option
        otm_options['Time_to_Maturity'] = otm_options['Maturity'].apply(
            lambda maturity: (datetime.strptime(maturity, "%m/%d/%Y") - start_date).days / 365.0
        )

        # Initialize a list to store Bates prices
        bates_prices = []

        # Iterate over each row to calculate Bates prices
        for _, row in otm_options.iterrows():
            maturity = row['Time_to_Maturity']
            strike = row['Strike']

            # Determine if the option is a call or a put
            if row['Strike'] > self.S0:  # Call option
                price = BatesModel(
                    self.S0, self.r, self.q, self.V0, self.kappa, self.eta, self.theta, self.rho,
                    maturity, strikes=[strike], jump_intensity=self.jump_intensity,
                    jump_mean=self.jump_mean, jump_stddev=self.jump_stddev
                ).price_options()[0]
            else:  # Put option
                price = BatesModel(
                    self.S0, self.r, self.q, self.V0, self.kappa, self.eta, self.theta, self.rho,
                    maturity, strikes=[strike], jump_intensity=self.jump_intensity,
                    jump_mean=self.jump_mean, jump_stddev=self.jump_stddev
                ).price_put_options()[0]

            bates_prices.append(price)

        # Add Bates prices to the DataFrame
        otm_options['Bates_Price'] = bates_prices

        return otm_options

