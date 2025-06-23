# Bonus Certificate Pricing and Hedging

This repository contains the full implementation and analysis for **pricing and hedging a Bonus Certificate (BC)** linked to the stock of American International Group (AIG). The BC is an exotic financial product that offers potential upside with partial downside protection, depending on the behavior of the underlying stock relative to specified barriers.

![image](https://github.com/user-attachments/assets/f5cb0a29-4422-4077-92ec-b0ae979695f6)


## 📘 Overview

This project includes:

- Pricing Bonus Certificates using the **Bates stochastic volatility model** with jumps.
- Calibrating the Bates model to real market option data.
- Simulating stock price paths using **Monte Carlo methods**.
- Implementing a **delta-hedging strategy** to mitigate issuer risk.
- Analyzing investor and bank outcomes under realistic scenarios.

## 💡 Product Structure

- **Underlying Asset:** AIG stock
- **Issue Date:** April 25, 2025
- **Maturity Date:** March 20, 2026
- **Initial Stock Price:** $81.25
- **Bonus Level (B):** $85
- **Barrier Level (H):** $55
- **Certificate Price:** $90

### Payoff Scenarios

- If the stock **never touches the barrier** and ends **below B**:  
  → Payoff = **B**

- If the stock **never touches the barrier** and ends **above B**:  
  → Payoff = **\( S_T \)**

- If the barrier is breached at any time:  
  → Payoff = **\( S_T \)**

This gives the investor capped upside, conditional downside protection, and no early redemption.

![image](https://github.com/user-attachments/assets/243c7b06-4d82-4674-9636-1a5c02d88ce5)

## 📊 Model Implementation

The project uses the **Bates model**, an extension of the Heston model that incorporates jumps:

- Stochastic volatility captured by a mean-reverting process.
- Jump-diffusion behavior through a Poisson process.

### Calibration

The model is calibrated to liquid option market data using the Carr-Madan FFT method. 
![image](https://github.com/user-attachments/assets/f70381cd-b2b6-4afe-951f-96f848a1a8bc)


## 🔧 Project Structure


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
