import os
from typing import Union
from datetime import datetime, timedelta
import pandas as pd


from dotenv import load_dotenv

load_dotenv()


def get_data_by_dates(start_date: Union[datetime, str], end_date=None, days=None):
    """
    Returns a single dataframe with all the data from the start date to the end date / for duration in days.
    """
    data_path = os.getenv("DATA_PATH")

    start_date = (
        datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(start_date, str)
        else start_date
    )
    if end_date is None:
        if days is None:
            return pd.read_csv(
                os.path.join(data_path, start_date.strftime("%Y_%m_%d"), "data.csv")
            )

        end_date = start_date + timedelta(days=days)
    else:
        end_date = (
            datetime.strptime(end_date, "%Y-%m-%d")
            if isinstance(end_date, str)
            else end_date
        )

    assert start_date < end_date, "start_date must be before end_date"

    # get all files that match the dates
    files = []
    for folder in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, folder)):
            folder_date = datetime.strptime(folder, "%Y_%m_%d")
            if start_date <= folder_date <= end_date:
                files.append(os.path.join(data_path, folder, "data.csv"))

    # compare lenght of files with the number of days
    print(f"Found {len(files)} files")
    if len(files) != (end_date - start_date).days + 1:
        print(
            f"Number of files({len(files)}) does not match the number of days given({(end_date - start_date).days + 1})"
        )

    df = pd.read_csv(files[0])
    for file in files[1:]:
        df = pd.concat([df, pd.read_csv(file)])
    df.sort_values(by="timestamp", inplace=True)
    return df


if __name__ == "__main__":
    print(get_data_by_dates("2021-12-21", days=1))
    print(get_data_by_dates("2021-12-21"))
