# Bonus Certificate Pricing and Hedging

This repository contains the implementation and analysis for pricing and hedging a **Bonus Certificate (BC)** linked to a specific stock (AIG in this case). The Bonus Certificate is an exotic financial product that provides potential upside with a cap, contingent on the stock price and certain barriers.

## Overview

The project includes tools for:
- Pricing Bonus Certificates using the Bates model.
- Simulating stock price paths and calculating option prices.
- Analyzing market data and visualizing results.
- Calibrating the Bates model to market data.

### Structure of the Product

- **Underlying Asset:** Stock of AIG
- **Maturity:** At a specified date
- **Bonus Level (B):** The level at which the product pays out if the stock price is above it at maturity.
- **Barrier Level (L):** The level between which the stock must end up for the bonus to be paid.
- **Lower Barrier (H):** If the stock price hits this barrier at any time during the product's life, the investor will receive the stock price at maturity.

### Payoff Structure

- If the stock price hits the lower barrier \( H \) at least once during the life of the product, the investor receives the value of the stock at maturity.
- **Payoff:** \( \text{payoff}_{BC} = S_T \)

## Project Structure

```
main.py
pyproject.toml
README.md
requirements.txt
uv.lock
data/
    marketDataClose25-04/
        [CSV files with market data]
    pdf/
        [PDF files with additional data]
notebooks/
    bc_pricing.ipynb
    market_data_analysis.ipynb
src/
    bates_formula/
        [Bates model implementation and calibration]
    data_processing/
        [Market data processing tools]
    dividends/
        [Dividend estimation tools]
    interest_rates/
        [Interest rate calculation tools]
    payout/
        [Payout calculation tools]
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd bonus-certificate-bates-hedging
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Main Script
Run the main script to execute the core functionality:
```bash
python main.py
```

### Jupyter Notebooks
- `notebooks/bc_pricing.ipynb`: Demonstrates the implementation of the Bates formula for pricing Bonus Certificates.
- `notebooks/market_data_analysis.ipynb`: Visualizes and analyzes market data.

### Data
Market data is stored in the `data/marketDataClose25-04/` folder. Each CSV file contains market data for a specific date.

## Key Features

### Bates Model
- Simulates stock price paths using the Bates model.
- Calibrates the model to market data.
- Prices options and Bonus Certificates.

### Market Data Analysis
- Processes and visualizes market data.
- Calculates implied volatility and interest rates.

### Payout Calculation
- Computes the payout structure for Bonus Certificates.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.