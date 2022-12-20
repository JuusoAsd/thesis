use csv::StringRecord;
use std::path::PathBuf;
mod file_util;
mod orderbook_util;
mod parse_util;
use file_util::{get_first_snapshot_file, get_folder_files, get_folder_update_files, FileHandler};
use parse_util::{parse_records_avellaneda_stoikov, parse_snapshot, parse_updates_v2};

fn parse_data_v1() {
    // v1 parses data as follows:
    // OB full state is recorded every n timestamps (where n is milliseconds), recording is OB state at end of the timestamp
    // For each timestamp, if there is a trade event, the orderbook state is recorded before the trade after that next OB is recorded
    //  - OB(t-1)
    //  - trade(t)
    //  - OB(t)
}

fn parse_data_v2() {
    // v2 updates on v1 by only recording OB updates every n timestamp rather than always recording state before a trade is executed
    // This makes the assumptions more realistic, we might not always have a good view of the data just before trade
    // even if we did, there would not be enough time to execute. v2 has limitation that growing n too large makes data lag too much
    // This is sort of the baseline for data parsing as the only thing done on "parsing-level" is record state every n timestamp

    // OB state recorded every n timestamps, trades happen in between OB updates or before the next update
    //  - OB(t)
    //  - trade(t+n/2)
    //  - trade(t+n)
    //  - OB(t+n)

    let file_count = 1;
    // let target_path = PathBuf::from("/C:/Users/Ville/Documents/gradu/data");
    let target_path = PathBuf::from("/home/juuso/Documents/gradu/parsed_data/orderbook");
    let tracked_levels = 25;
    let timestamp_aggregation = 10;

    // let folder_path =
    //     PathBuf::from("/C:/Users/Ville/Documents/gradu/data/ADAUSDT_T_DEPTH_2021-12-21");
    let folder_path =
        PathBuf::from("/media/juuso/5655B83E58A8FD4F/orderbook/ADAUSDT_T_DEPTH_202211031113(1)");
    let first_snapshot_file: PathBuf = get_first_snapshot_file(&folder_path).unwrap();
    let update_files = get_folder_update_files(&folder_path);
    let (first_ts, ob) = parse_snapshot(&first_snapshot_file, 0);

    let used_files = if file_count != 0 {
        update_files[0..file_count].to_vec()
    } else {
        update_files
    };
    parse_updates_v2(
        first_ts,
        i64::MAX,
        tracked_levels,
        timestamp_aggregation,
        &target_path,
        used_files,
        ob,
    );
}

fn parse_data_AS() {
    let file_count = 2;
    // let target_path =
    //     PathBuf::from(r"C:\Users\Ville\Documents\gradu\parsed_data\AS\data.csv");
    let target_path =
        PathBuf::from("/home/juuso/Documents/gradu/parsed_data/AvellanedaStoikov/data.csv");
    let timestamp_aggregation = 10;

    // let update_path =
    // PathBuf::from(r"C:\Users\Ville\Documents\gradu\data\ADAUSDT_T_DEPTH_2021-12-21");
    let update_path =
        PathBuf::from("/media/juuso/5655B83E58A8FD4F/orderbook/ADAUSDT_T_DEPTH_202211031113(1)");
    let update_headers = StringRecord::from(vec![
        "symbol",
        "timestamp",
        "first_update_id",
        "last_update_id",
        "side",
        "update_type",
        "price",
        "qty",
        "pu",
    ]);

    // let trade_path = PathBuf::from(r"C:\Users\Ville\Documents\gradu\data\trades");
    let trade_path = PathBuf::from("/media/juuso/5655B83E58A8FD4F/trades");
    let trade_header = StringRecord::from(vec![
        "trade_id",
        "price",
        "qty",
        "total_value",
        "timestamp",
        "is_buyer_maker",
    ]);

    let first_snapshot_file: PathBuf = get_first_snapshot_file(&update_path).unwrap();
    let update_files = get_folder_update_files(&update_path);
    let trade_files = get_folder_files(&trade_path);
    let (first_ts, ob) = parse_snapshot(&first_snapshot_file, 0);

    let (update_files_used, trade_files_used) = if file_count != 0 {
        (
            update_files[0..file_count].to_vec(),
            trade_files[0..file_count].to_vec(),
        )
    } else {
        (update_files, trade_files)
    };

    let mut update_handler: FileHandler = FileHandler::new(update_files_used, update_headers);
    let mut trade_handler: FileHandler = FileHandler::new(trade_files_used, trade_header);
    parse_records_avellaneda_stoikov(
        first_ts,
        i64::MAX,
        timestamp_aggregation,
        &target_path,
        update_handler,
        trade_handler,
        ob,
    )
}
fn main() {
    parse_data_AS();
}
