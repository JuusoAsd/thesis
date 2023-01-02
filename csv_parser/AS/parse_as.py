from src.environments.util import FileManager

if __name__ == "__main__":
    """
    Parse AS agent data and save it to a single CSV file / folder of files
    PARSED DATA:
        timestamp, best bid, best ask, (trade price/0, trade size/0), current vol estimate, current intensity
    """
    as_files = FileManager(
        "/home/juuso/Documents/gradu/parsed_data/AvellanedaStoikov/data.csv",
    )
    current_state = as_files.get_next_event()
    print(current_state)
