from src.data_processing.market_data_processor import MarketDataProcessor
from src.bates_formula.calibration import BatesModel
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime

class BatesModelGlobalCalibrator:
    def __init__(self, S0, r, q, date=None, combined_data=None):
        self.S0 = S0
        self.r = r
        self.q = q
        self.global_calibrated_params = {}
        self.date = date if date else datetime.now()
        self.combined_data = combined_data

    def calibrate_global_model(self, combined_data, date=None):
        # Use the provided date or default to the instance's date
        date = date if date else self.date

        # Calculate time to maturity
        combined_data['Time_to_Maturity'] = combined_data['Maturity'].apply(
            lambda maturity: self._calculate_time_to_maturity(maturity, date)
        )

        # Initial guess for parameters
        initial_params = {
            'V0': 0.1, 'kappa': 0.2, 'eta': 0.2, 'theta': 0.5, 'rho': -0.5,
            'jump_intensity': 0.5, 'jump_mean': 0.05, 'jump_stddev': 0.02
        }

        # Bounds for parameters
        bounds = {
            'V0': (0.01, 1.0), 'kappa': (0.01, 3.0), 'eta': (0.01, 2.0), 'theta': (0.01, 1.0), 'rho': (-0.99, 0.0),
            'jump_intensity': (0.01, 2.0), 'jump_mean': (0.01, 0.5), 'jump_stddev': (0.01, 0.5)
        }

        # Minimize the total error
        result = minimize(
            self._total_error,
            x0=list(initial_params.values()),
            bounds=list(bounds.values()),
            method='L-BFGS-B',
            args=(combined_data, date)
        )

        # Store the calibrated parameters
        self.global_calibrated_params = dict(zip(initial_params.keys(), result.x))

    def _total_error(self, params, combined_data, date):
        # Unpack parameters
        V0, kappa, eta, theta, rho, jump_intensity, jump_mean, jump_stddev = params

        # Create a BatesModel instance
        bates_model = BatesModel(
            self.S0, self.r, self.q, V0, kappa, eta, theta, rho,
            jump_intensity=jump_intensity, jump_mean=jump_mean, jump_stddev=jump_stddev
        )

        # Price options for all maturities
        priced_data = bates_model.price_options_for_all_maturities(combined_data, date)

        # Calculate total squared error
        return np.sum((priced_data['Bates_Price'] - priced_data['Mid_Price'])**2)

    def _calculate_time_to_maturity(self, maturity, date):
        from datetime import datetime
        maturity_date = datetime.strptime(maturity, "%m/%d/%Y")
        return (maturity_date - date).days / 365.0

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
# global_calibrator.calibrate_global_model(combined_data)
# global_params = global_calibrator.get_global_calibrated_params()
# print(global_params)