from csv_parser.AS.intensity import IntensityEstimator


def test_read_line_trades():
    estimator = IntensityEstimator()
    n = 0
    with open(
        "/home/juuso/Documents/gradu/parsed_data/trades/ADAUSDT-trades-2021-12-21.csv",
        "r",
    ) as f:
        while True:
            n += 1
            if n > 1000:
                break
            line = f.readline().rstrip().split(",")
            price = float(line[1])
            amount = float(line[2])
            time = int(line[4])
            estimator.estimate_intensity([(price, amount)])
            print(estimator.alpha, estimator.kappa)
