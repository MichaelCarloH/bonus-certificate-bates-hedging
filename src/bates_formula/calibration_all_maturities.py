from src.data_processing.market_data_processor import MarketDataProcessor
from src.bates_formula.calibration import calibrate_bates_model
import pandas as pd
import numpy as np

class BatesModelCalibrator:
    def __init__(self, data_folder, S0, r, q):
        self.data_folder = data_folder
        self.S0 = S0
        self.r = r
        self.q = q
        self.calibrated_params_by_maturity = {}

    def calibrate_all_maturities(self):
        # Load and process market data
        processor = MarketDataProcessor(self.data_folder)
        processor.load_and_process_data()
        data_by_maturity = processor.get_data_by_maturity()

        # Iterate over each maturity and calibrate the Bates model
        for maturity, df in data_by_maturity.items():
            market_prices = (df['Bid_Call'] + df['Ask_Call']) / 2  # Mid prices for calls
            strikes = df['Strike']
            T = self._calculate_time_to_maturity(maturity)

            # Calibrate the Bates model
            calibrated_params = calibrate_bates_model(market_prices, strikes, self.S0, T, self.r, self.q)
            self.calibrated_params_by_maturity[maturity] = calibrated_params

    def _calculate_time_to_maturity(self, maturity):
        from datetime import datetime
        maturity_date = datetime.strptime(maturity, "%m/%d/%Y")
        current_date = datetime.now()
        return (maturity_date - current_date).days / 365.0

    # Helper function to convert numpy types to native Python types
    def convert_numpy_to_native(value):
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        return value

    def get_calibrated_params(self):
        return {
            maturity: {key: self.convert_numpy_to_native(val) for key, val in params.items()}
            for maturity, params in self.calibrated_params_by_maturity.items()
        }

# Example usage:
# calibrator = BatesModelCalibrator("../data/marketDataClose25-04", S0=77.56, r=0.02, q=0.01)
# calibrator.calibrate_all_maturities()
# calibrated_params = calibrator.get_calibrated_params()
# print(calibrated_params)