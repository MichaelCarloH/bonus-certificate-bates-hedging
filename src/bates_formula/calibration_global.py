from src.data_processing.market_data_processor import MarketDataProcessor
from src.bates_formula.calibration import BatesModel
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime

class BatesModelGlobalCalibrator:
    def __init__(self, data_folder, S0, r, q, date = datetime.now()):
        self.data_folder = data_folder
        self.S0 = S0
        self.r = r
        self.q = q
        self.global_calibrated_params = {}

    def calibrate_global_model(self):
        # Ensure all maturities are processed and combined
        processor = MarketDataProcessor(self.data_folder)
        processor.load_and_process_data()
        data_by_maturity = processor.get_data_by_maturity()

        # Combine all data into a single DataFrame
        combined_data = pd.concat(data_by_maturity.values(), ignore_index=True)

        # Extract market prices, strikes, and maturities
        self.market_prices = (combined_data['Bid_Call'] + combined_data['Ask_Call']) / 2  # Mid prices for calls
        self.strikes = combined_data['Strike']
        self.maturities = combined_data['Maturity'].apply(self._calculate_time_to_maturity)

        # Initial guess for parameters
        initial_params = {
            'V0': 0.1, 'kappa': 0.2, 'eta': 0.2, 'theta': 0.5, 'rho': -0.5,
            'jump_intensity': 0.5, 'jump_mean': 0.05, 'jump_stddev': 0.02
        }

        # Bounds for parameters
        bounds = {
            'V0': (0.01, 1.0), 'kappa': (0.01, 2.0), 'eta': (0.01, 2.0), 'theta': (0.01, 1.0), 'rho': (-0.99, 0.0),
            'jump_intensity': (0.01, 2.0), 'jump_mean': (0.01, 0.5), 'jump_stddev': (0.01, 0.5)
        }

        # Minimize the total error
        result = minimize(
            self._total_error,
            x0=list(initial_params.values()),
            bounds=list(bounds.values()),
            method='L-BFGS-B'
        )

        # Store the calibrated parameters
        self.global_calibrated_params = dict(zip(initial_params.keys(), result.x))

    def _total_error(self, params):
        # Unpack parameters
        V0, kappa, eta, theta, rho, jump_intensity, jump_mean, jump_stddev = params

        # Calculate model prices for all maturities
        model_prices = []
        for strike, maturity in zip(self.strikes, self.maturities):
            model_price = BatesModel(
                self.S0, self.r, self.q, V0, kappa, eta, theta, rho, maturity,
                strikes=[strike], jump_intensity=jump_intensity, jump_mean=jump_mean, jump_stddev=jump_stddev
            ).price_options()[0]
            model_prices.append(model_price)

        # Calculate total squared error
        return np.sum((np.array(model_prices) - np.array(self.market_prices))**2)

    def _calculate_time_to_maturity(self, maturity):
        from datetime import datetime
        maturity_date = datetime.strptime(maturity, "%m/%d/%Y")
        current_date = datetime.now()
        return (maturity_date - current_date).days / 365.0

    def get_global_calibrated_params(self):
        return {
            key: self._convert_numpy_to_native(val) for key, val in self.global_calibrated_params.items()
        }

    def _convert_numpy_to_native(self, value):
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        return value

# Example usage:
# global_calibrator = BatesModelGlobalCalibrator("../data/marketDataClose25-04", S0=77.56, r=0.02, q=0.01)
# global_calibrator.calibrate_global_model()
# global_params = global_calibrator.get_global_calibrated_params()
# print(global_params)