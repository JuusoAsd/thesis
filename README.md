# Data

### Base data
- timestamp: UTC timestamp in milliseconds
- best_bid: current best bid price based on order book
- best_ask: current best ask price based on order book
- mid_price: average of best bid and ask
- low_price: lowest executed price in the previous 1000ms interval
- high_price: highest executed price in the previous 1000ms interval

Example:
- t=0
    - record best bid and ask prices
- t=1000
    - record low price and high price between t=0 and t=1000
    - calculate the mid price

### Indicator data
- intensity:
- vol