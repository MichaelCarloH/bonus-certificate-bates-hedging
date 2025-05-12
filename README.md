# Bonus Certificate Pricing and Hedging

This repository contains the implementation and analysis for pricing and hedging a **Bonus Certificate (BC)** linked to a specific stock (AIG in this case). The Bonus Certificate is an exotic financial product that provides potential upside with a cap, contingent on the stock price and certain barriers.

## Overview

A Bonus Certificate (BC) provides an investor with exposure to a stock while offering an additional bonus payment if the stock price stays above a specified barrier level during the life of the product. However, if the stock price falls below a **lower barrier**, the bonus is no longer paid. The product has a single maturity and no principal protection.

### Payoff Structure

The payoff at maturity depends on the stock price at that time, the barrier levels, and whether the stock has breached the lower barrier level during the product's life.

The three main payoff scenarios are:

1. **Stock Price Above Bonus Level (B):**
   - If the stock price ends up above the bonus level \( B \), the investor receives the value of the stock at maturity \( S_T \).
   - **Payoff:** \( \text{payoff}_{BC} = S_T \)

2. **Stock Price Between Barrier (L) and Bonus Level (B):**
   - If the stock price ends up between the barrier \( L \) and the bonus \( B \), and the stock price has never fallen below the lower barrier \( H \), the investor receives the bonus \( B \).
   - **Payoff:** \( \text{payoff}_{BC} = B \)

3. **Stock Price Hits the Lower Barrier (H):**
   - If the stock price hits the lower barrier \( H \) at least once during the life of the product, the investor receives the value of the stock at maturity.
   - **Payoff:** \( \text{payoff}_{BC} = S_T \)

### Structure of the Product

- **Underlying Asset:** Stock of AIG
- **Maturity:** At a specified date
- **Bonus Level (B):** The level at which the product pays out if the stock price is above it at maturity.
- **Barrier Level (L):** The level between which the stock must end up for the bonus to be paid.
- **Lower Barrier (H):** If the stock price hits this barrier at any time during the product's life, the investor will receive the stock price at maturity.

### Bank’s Strategy

To hedge the Bonus Certificate, the bank will:
- **Long the stock** to capture dividends.
- **Sell a Down-and-In Binary Put (DIBP)** at the lower barrier \( L \).
- **Buy a Down-and-In Binary Put (DIBP)** at the higher barrier \( H \).
- These positions are combined to create a **synthetic payout** similar to the Bonus Certificate payoff.
- The bank’s exposure is between the barriers \( H \) and \( L \), and its maximum loss will be covered by a **risk-free bank account**.
