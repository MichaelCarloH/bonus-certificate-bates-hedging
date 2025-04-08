def estimate_dividends(stock_price, dividend_yield, time_period):
    """
    Estimate the dividends for the underlying stock over a specified time period.

    Parameters:
    stock_price (float): The current price of the stock.
    dividend_yield (float): The annual dividend yield as a decimal (e.g., 0.03 for 3%).
    time_period (float): The time period in years for which to estimate dividends.

    Returns:
    float: The estimated dividends over the specified time period.
    """
    return stock_price * dividend_yield * time_period


def estimate_dividend_growth(current_dividend, growth_rate, time_period):
    """
    Estimate the future dividend based on the current dividend and growth rate.

    Parameters:
    current_dividend (float): The current dividend amount.
    growth_rate (float): The annual growth rate of the dividend as a decimal.
    time_period (float): The time period in years for which to estimate future dividends.

    Returns:
    float: The estimated future dividend.
    """
    return current_dividend * (1 + growth_rate) ** time_period


def total_estimated_dividends(stock_price, dividend_yield, growth_rate, time_period):
    """
    Calculate the total estimated dividends over a specified time period.

    Parameters:
    stock_price (float): The current price of the stock.
    dividend_yield (float): The annual dividend yield as a decimal.
    growth_rate (float): The annual growth rate of the dividend as a decimal.
    time_period (float): The time period in years for which to estimate dividends.

    Returns:
    float: The total estimated dividends over the specified time period.
    """
    estimated_dividends = estimate_dividends(stock_price, dividend_yield, time_period)
    future_dividend = estimate_dividend_growth(stock_price * dividend_yield, growth_rate, time_period)
    return estimated_dividends + future_dividend