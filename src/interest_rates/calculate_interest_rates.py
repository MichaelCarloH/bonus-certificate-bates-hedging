import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_interest_rate_from_parity(df, dividend_yield, time_to_maturity):
    """
    Calculate the implied interest rate (r) using put-call parity for each row in the DataFrame.

    Parameters:
    - df: DataFrame containing columns ['Call', 'Put', 'Spot', 'Strike']
    - dividend_yield: The dividend yield (q) as a decimal
    - time_to_maturity: Time to maturity (T) in years

    Returns:
    - DataFrame with an additional column 'InterestRate' containing the calculated interest rates
    """
    # Ensure required columns exist
    required_columns = ['Call', 'Put', 'Spot', 'Strike']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    # Calculate the interest rate (r) for each row
    df['InterestRate'] = -np.log(
        (df['Call'] - df['Put'] - df['Spot'] * np.exp(-dividend_yield * time_to_maturity)) / df['Strike']
    ) / time_to_maturity

    return df


def plot_interest_rates(df):
    """
    Plot the interest rates for each strike price.

    Parameters:
    - df: DataFrame containing columns ['Strike', 'InterestRate']
    """
    if 'Strike' not in df.columns or 'InterestRate' not in df.columns:
        raise ValueError("DataFrame must contain 'Strike' and 'InterestRate' columns.")

    # Plot using matplotlib
    plt.figure(figsize=(10, 6))
    plt.plot(df['Strike'], df['InterestRate'], marker='o', linestyle='-', color='blue')
    plt.title("Interest Rates vs Strike Prices")
    plt.xlabel("Strike Price")
    plt.ylabel("Interest Rate")
    plt.grid(True)
    plt.show()


# Example usage
if __name__ == "__main__":
    # Example DataFrame
    data = {
        'Call': [10, 12, 15],
        'Put': [8, 9, 11],
        'Spot': [100, 100, 100],
        'Strike': [90, 100, 110]
    }
    df = pd.DataFrame(data)

    # Parameters
    dividend_yield = 0.02  # 2% dividend yield
    time_to_maturity = 1  # 1 year

    # Calculate interest rates
    df = calculate_interest_rate_from_parity(df, dividend_yield, time_to_maturity)

    # Plot interest rates
    plot_interest_rates(df)