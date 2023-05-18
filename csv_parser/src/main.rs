use csv::StringRecord;
use std::path::PathBuf;
mod file_util;
mod orderbook_util;
mod parse_util;
use csv::WriterBuilder;
use dotenvy;
use file_util::{
    get_csv_files_key, get_csv_files_sorted, get_first_snapshot_file, get_first_timestamp,
    get_folder_update_files, get_last_timestamp, FileHandler,
};
use parse_util::{
    parse_both_records, parse_records_avellaneda_stoikov, parse_snapshot, parse_updates_v2,
};
use std::env;

use crate::orderbook_util::Orderbook;
// use dotenvy;
// dotenvy::from_filename("rust.env");
// from_filename("rust.env").ok();

// fn parse_data_v1() {
//     // v1 parses data as follows:
//     // OB full state is recorded every n timestamps (where n is milliseconds), recording is OB state at end of the timestamp
//     // For each timestamp, if there is a trade event, the orderbook state is recorded before the trade after that next OB is recorded
//     //  - OB(t-1)
//     //  - trade(t)
//     //  - OB(t)
// }

// fn parse_data_v2() {
//     // v2 updates on v1 by only recording OB updates every n timestamp rather than always recording state before a trade is executed
//     // This makes the assumptions more realistic, we might not always have a good view of the data just before trade
//     // even if we did, there would not be enough time to execute. v2 has limitation that growing n too large makes data lag too much
//     // This is sort of the baseline for data parsing as the only thing done on "parsing-level" is record state every n timestamp

//     // OB state recorded every n timestamps, trades happen in between OB updates or before the next update
//     //  - OB(t)
//     //  - trade(t+n/2)
//     //  - trade(t+n)
//     //  - OB(t+n)

//     let file_count = 1;
//     // let target_path = PathBuf::from("/C:/Users/Ville/Documents/gradu/data");
//     let target_path = PathBuf::from("/home/juuso/Documents/gradu/parsed_data/orderbook");
//     let tracked_levels = 25;
//     let timestamp_aggregation = 10;

//     // let folder_path =
//     //     PathBuf::from("/C:/Users/Ville/Documents/gradu/data/ADAUSDT_T_DEPTH_2021-12-21");
//     let folder_path =
//         PathBuf::from("/media/juuso/5655B83E58A8FD4F/orderbook/ADAUSDT_T_DEPTH_202211031113(1)");
//     let first_snapshot_file: PathBuf = get_first_snapshot_file(&folder_path).unwrap();
//     let update_files = get_folder_update_files(&folder_path);
//     let (first_ts, ob) = parse_snapshot(&first_snapshot_file, 0);

//     let used_files = if file_count != 0 {
//         update_files[0..file_count].to_vec()
//     } else {
//         update_files
//     };
//     parse_updates_v2(
//         first_ts,
//         i64::MAX,
//         tracked_levels,
//         timestamp_aggregation,
//         &target_path,
//         used_files,
//         ob,
//     );
// }

// fn parse_data_AS() {
//     let file_count = 2;
//     // let target_path =
//     //     PathBuf::from(r"C:\Users\Ville\Documents\gradu\parsed_data\AS\data.csv");
//     let target_path =
//         PathBuf::from("/home/juuso/Documents/gradu/parsed_data/AvellanedaStoikov/data.csv");
//     let timestamp_aggregation = 10;

//     // let update_path =
//     // PathBuf::from(r"C:\Users\Ville\Documents\gradu\data\ADAUSDT_T_DEPTH_2021-12-21");
//     let update_path =
//         PathBuf::from("/media/juuso/5655B83E58A8FD4F/orderbook/ADAUSDT_T_DEPTH_202211031113(1)");
//     let update_headers = StringRecord::from(vec![
//         "symbol",
//         "timestamp",
//         "first_update_id",
//         "last_update_id",
//         "side",
//         "update_type",
//         "price",
//         "qty",
//         "pu",
//     ]);

//     // let trade_path = PathBuf::from(r"C:\Users\Ville\Documents\gradu\data\trades");
//     let trade_path = PathBuf::from("/media/juuso/5655B83E58A8FD4F/trades");
//     let trade_header = StringRecord::from(vec![
//         "trade_id",
//         "price",
//         "qty",
//         "total_value",
//         "timestamp",
//         "is_buyer_maker",
//     ]);

//     let first_snapshot_file: PathBuf = get_first_snapshot_file(&update_path).unwrap();
//     let update_files = get_folder_update_files(&update_path);
//     let trade_files = get_folder_files(&trade_path);
//     let (first_ts, ob) = parse_snapshot(&first_snapshot_file, 0);

//     let (update_files_used, trade_files_used) = if file_count != 0 {
//         (
//             update_files[0..file_count].to_vec(),
//             trade_files[0..file_count].to_vec(),
//         )
//     } else {
//         (update_files, trade_files)
//     };

//     let mut update_handler: FileHandler = FileHandler::new(update_files_used, update_headers, true);
//     let mut trade_handler: FileHandler = FileHandler::new(trade_files_used, trade_header, false);
//     parse_records_avellaneda_stoikov(
//         first_ts,
//         i64::MAX,
//         timestamp_aggregation,
//         &target_path,
//         update_handler,
//         trade_handler,
//         ob,
//     )
// }

// fn parse_data_time_aggregation(file_count: usize) {
//     // produces a csv with data aggregated on timestamp_aggregation interval
//     // data contains timestamp,best_bid,best_ask,low_price,high_price,buy_volume,sell_volume for the time interval
//     // benchmark speed is 1M per second
//     println!("Parsing data with time aggregation");
//     let target_path = PathBuf::from(env::var("TARGET_PATH_BASE").unwrap());
//     let timestamp_aggregation = 1000;

//     let update_path = PathBuf::from(env::var("UPDATE_PATH").unwrap());
//     let update_headers = StringRecord::from(vec![
//         "symbol",
//         "timestamp",
//         "first_update_id",
//         "last_update_id",
//         "side",
//         "update_type",
//         "price",
//         "qty",
//         "pu",
//     ]);
//     let update_files = get_folder_update_files(&update_path);
//     let first_snapshot_file = get_first_snapshot_file(&update_path).unwrap();

//     let trade_path = PathBuf::from(env::var("TRADE_PATH").unwrap());
//     let trade_header = StringRecord::from(vec![
//         "trade_id",
//         "price",
//         "qty",
//         "total_value",
//         "timestamp",
//         "is_buyer_maker",
//     ]);
//     let all_trade_files = get_csv_files_sorted(&trade_path);

//     let (first_ts, ob) = parse_snapshot(&first_snapshot_file, 0);
//     println!("snapshot file: {:?}", first_snapshot_file);
//     println!("update file: {:?}", update_files);

//     let (update_files_used, trade_files_used) = if file_count != 0 {
//         (
//             update_files[0..file_count].to_vec(),
//             all_trade_files[0..file_count].to_vec(),
//         )
//     } else {
//         (update_files, all_trade_files)
//     };

//     let mut update_handler: FileHandler = FileHandler::new(update_files_used, update_headers, true);
//     let mut trade_handler: FileHandler = FileHandler::new(trade_files_used, trade_header, false);
//     parse_records_aggregate_ts(
//         first_ts,
//         i64::MAX,
//         timestamp_aggregation,
//         &target_path,
//         update_handler,
//         trade_handler,
//         ob,
//     )
// }

// fn parse_interim_data(file_count: usize, redo: bool) {
//     // parses raw trade data into interim data, look at InterimRecord
//     // if redo is true, the data is parsed from scratch
//     println!("Parsing interim data");
//     let target_path = PathBuf::from(env::var("TARGET_PATH_INTERIM").unwrap());
//     let update_path = PathBuf::from(env::var("UPDATE_PATH").unwrap());
//     let update_headers = StringRecord::from(vec![
//         "symbol",
//         "timestamp",
//         "first_update_id",
//         "last_update_id",
//         "side",
//         "update_type",
//         "price",
//         "qty",
//         "pu",
//     ]);

//     let trade_path = PathBuf::from(env::var("TRADE_PATH").unwrap());
//     let trade_header = StringRecord::from(vec![
//         "trade_id",
//         "price",
//         "qty",
//         "total_value",
//         "timestamp",
//         "is_buyer_maker",
//     ]);

//     if !redo {
//         // not redoing, trades can be started from last timestamp
//         // order book must be initialized from earlier snapshot
//         // find last timestamp recorded on the target path
//         let last_ts = get_last_timestamp(&target_path, 0);
//     } else {
//         // redoing, want to start from the first timestamp
//         let first_snapshot_file: PathBuf = get_first_snapshot_file(&update_path).unwrap();
//         let update_files = get_folder_update_files(&update_path);
//         let trade_files = get_csv_files_sorted(&trade_path);
//         let (first_ts, ob) = match parse_snapshot(&first_snapshot_file, 0) {
//             Some((ts, ob)) => (ts, ob),
//             None => panic!("Error parsing snapshot, no file found"),
//         };

//         let (update_files_used, trade_files_used) = if file_count != 0 {
//             (
//                 update_files[0..file_count].to_vec(),
//                 trade_files[0..file_count].to_vec(),
//             )
//         } else {
//             (update_files, trade_files)
//         };

//         let mut update_handler: FileHandler =
//             FileHandler::new(update_files_used, update_headers, true);
//         let mut trade_handler: FileHandler =
//             FileHandler::new(trade_files_used, trade_header, false);
//         parse_records_interim_data(
//             first_ts,
//             i64::MAX,
//             &target_path,
//             update_handler,
//             trade_handler,
//             ob,
//         )
//     }
// }
use std::fs::OpenOptions;

fn parse_data(redo: bool, interim: bool, base: bool) {
    // rather than running 2 separate parsers, run them together to decrease iterations
    // also allows parsing only one of the two
    println!("Parsing data");
    let target_base = PathBuf::from(env::var("TARGET_PATH_BASE").unwrap());
    let target_interim = PathBuf::from(env::var("TARGET_PATH_INTERIM").unwrap());

    let update_path = PathBuf::from(env::var("UPDATE_PATH").unwrap());
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

    let trade_path = PathBuf::from(env::var("TRADE_PATH").unwrap());
    let trade_header = StringRecord::from(vec![
        "trade_id",
        "price",
        "qty",
        "total_value",
        "timestamp",
        "is_buyer_maker",
    ]);
    let mut snapshot_files = get_csv_files_key(&update_path, "depth_snap");
    let mut update_files = get_csv_files_key(&update_path, "depth_update");
    let mut trade_files = get_csv_files_sorted(&trade_path);

    // println!("snapshot files: {:?}", snapshot_files);
    // println!("update files: {:?}", update_files);
    // println!("trade files: {:?}", trade_files);

    let mut order_book = Orderbook::new();
    let first_ts: i64;

    let append = !redo;
    let mut base_file = OpenOptions::new()
        .write(true)
        .append(append)
        .open(target_base.clone())
        .unwrap();
    let mut base_wtr = WriterBuilder::new()
        .has_headers(redo)
        .from_writer(base_file);

    let mut interim_file = OpenOptions::new()
        .write(true)
        .append(append)
        .open(target_interim)
        .unwrap();
    let mut interim_wtr = WriterBuilder::new()
        .has_headers(redo)
        .from_writer(interim_file);

    if redo {
        // we are redoing so start from the first order book snapshot
        let first_snapshot = snapshot_files[0].clone();
        let ts: i64;
        (ts, order_book) = match parse_snapshot(&first_snapshot, 0) {
            Some((ts, ob)) => (ts, ob),
            None => panic!("Error parsing snapshot, no file found"),
        };
        first_ts = ts;
    } else {
        // we are not redoing, find the last timestamp recorded on the target path
        let mut last_ts = get_last_timestamp(&target_base, 0);
        // remove some seconds to make sure we don't miss any data
        last_ts = last_ts - 10_000_000;
        println!("Goal timestamp: {}", last_ts);
        // iterate snapshot files in reverse order until find one with timestamp < last_ts
        for file in snapshot_files.iter().rev() {
            let ts = get_first_timestamp(&file, 1);
            if ts < last_ts {
                // found the first snapshot file with timestamp < last_ts
                // parse it and use it as the starting order book
                let (first_ts, ob) = match parse_snapshot(&file, last_ts) {
                    Some((ts, ob)) => (ts, ob),
                    None => panic!("Error parsing snapshot, no file found"),
                };
                order_book = ob;
                break;
            }
        }

        // iterate update and trade files in reverse until find one with timestamp < last_ts
        // use the remaining files as the update and trade files
        let mut file_count = 0;
        for file in update_files.iter().rev() {
            let ts = get_first_timestamp(&file, 1);
            if ts < last_ts {
                break;
            }
            file_count += 1;
        }
        update_files = update_files[update_files.len() - file_count..update_files.len()].to_vec();

        let mut file_count = 0;
        for file in trade_files.iter().rev() {
            let ts = get_first_timestamp(&file, 4);
            if ts < last_ts {
                break;
            }
            file_count += 1;
        }
        trade_files = trade_files[trade_files.len() - file_count..trade_files.len()].to_vec();
        first_ts = last_ts + 10_000_000;
    }

    println!("update_files: {:?}", update_files.len());
    println!("trade_files: {:?}", trade_files.len());
    println!("mid price: {:?}", order_book.get_midprice());

    let mut update_handler: FileHandler = FileHandler::new(update_files, update_headers, true);
    let mut trade_handler: FileHandler = FileHandler::new(trade_files, trade_header, false);

    if !redo {
        // rewind order book to match the last timestamp
    }

    parse_both_records(
        first_ts,
        i64::MAX,
        1000,
        interim_wtr,
        base_wtr,
        update_handler,
        trade_handler,
        order_book,
        interim,
        base,
    );
}

fn main() {
    match dotenvy::from_filename("parse.env") {
        Ok(_) => println!("Loaded .env file"),
        Err(e) => println!("Error loading .env file: {}", e),
    }
    // parse_interim_data(file_count);
    // parse_data_time_aggregation(file_count);
    // let target_path = PathBuf::from(env::var("TARGET_PATH_INTERIM").unwrap());
    // let current_files = get_csv_files_sorted(&target_path);
    // let last_file = current_files.last().unwrap();
    // let first_ts = get_first_timestamp(&last_file, 0);
    // println!("first_ts: {:?}", first_ts);
    parse_data(true, false, true);
}
