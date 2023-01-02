from src.environments.util import FileManager
from csv_parser.AS.estimators import IntensityEstimator
from csv_parser.AS.estimators import VolatilityEstimator


def parse_as_full():
    """
    Parse AS agent data and save it to a single CSV file / folder of files
    PARSED DATA:
        timestamp, best bid, best ask, (trade price/0, trade size/0), current vol estimate, current intensity
    """
    as_files = FileManager("./parsed_data/AvellanedaStoikov/data.csv", headers=True)
    target_file = open(f"./parsed_data/AvellanedaStoikov/AS_full.csv", "w+")
    current_state = as_files.get_next_event()
    intensity = IntensityEstimator(lookback=250_000)
    volatility = VolatilityEstimator()
    while current_state is not None:
        ts = int(current_state[0])
        mid_price = float(current_state[1])
        best_bid = float(current_state[2])
        best_ask = float(current_state[3])
        trade_size = float(current_state[4])
        trade_price = float(current_state[5])

        if trade_size != 0:
            intensity.update_trades([(ts, trade_price, trade_size, mid_price)])
            intensity.calculate_current_values()
        volatility.update_prices(mid_price)
        # volatility.calculate_volatility()

        intensity_estimate = intensity.kappa
        vol_estimate = volatility.volatility

        current_state = as_files.get_next_event()
