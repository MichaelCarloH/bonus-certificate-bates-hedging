import plotly.graph_objects as go
import numpy as np

def calculate_bonus_certificate_payout(stock_price_at_maturity, bonus_level, lower_barrier, has_hit_barrier):
    """
    Calculate the payout of a Bonus Certificate based on the stock price at maturity, bonus level,
    lower barrier, and whether the barrier has been hit during the product's lifetime.

    Parameters:
    - stock_price_at_maturity: The stock price at the maturity date (ST).
    - bonus_level: The bonus level (B).
    - lower_barrier: The lower barrier level (H).
    - has_hit_barrier: Boolean indicating whether the stock price has hit the lower barrier (H) during the product's lifetime.

    Returns:
    - payout: The calculated payout of the Bonus Certificate.
    """
    if has_hit_barrier:
        # Scenario 3: The stock price has hit the lower barrier at least once.
        payout = stock_price_at_maturity
    elif stock_price_at_maturity >= bonus_level:
        # Scenario 1: The stock price ends up above the bonus level.
        payout = stock_price_at_maturity
    elif lower_barrier <= stock_price_at_maturity < bonus_level:
        # Scenario 2: The stock price is between the lower barrier and the bonus level,
        # and the barrier has not been hit.
        payout = bonus_level
    else:
        # Fallback case (should not occur based on the scenarios described).
        payout = stock_price_at_maturity

    return payout


def plot_payout(bonus_level, lower_barrier):
    """
    Plot the payout of the Bonus Certificate for both cases: barrier hit and not hit.
    """
    # Generate a range of stock prices for plotting
    stock_prices = np.arange(0, bonus_level * 1.5 + 0.1, 0.1).tolist()

    # Calculate payouts for both cases
    payouts_barrier_hit = [calculate_bonus_certificate_payout(price, bonus_level, lower_barrier, True) for price in stock_prices]
    payouts_barrier_not_hit = [calculate_bonus_certificate_payout(price, bonus_level, lower_barrier, False) for price in stock_prices]

    # Create a Plotly figure
    fig = go.Figure()

    # Add the payout line for barrier hit
    fig.add_trace(go.Scatter(
        x=stock_prices, y=payouts_barrier_hit,
        mode='lines', name='Barrier Hit',
        line=dict(color='red', shape='hv')  # Use 'hv' for step-like behavior
    ))

    # Add the payout line for barrier not hit
    fig.add_trace(go.Scatter(
        x=stock_prices, y=payouts_barrier_not_hit,
        mode='lines', name='Barrier Not Hit',
        line=dict(color='green', shape='hv')  # Use 'hv' for step-like behavior
    ))

    # Customize the layout
    fig.update_layout(
        title="Bonus Certificate Payout Structure",
        xaxis_title="Stock Price at Maturity",
        yaxis_title="Payout",
        template="plotly_white",
        legend=dict(title="Scenario")
    )

    # Show the plot
    fig.show()

# Example usage
if __name__ == "__main__":
    bonus_level = 150  # Example bonus level
    lower_barrier = 100  # Example lower barrier

    # Plot the payout structure
    plot_payout(bonus_level, lower_barrier)