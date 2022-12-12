use serde::{Deserialize, Serialize};

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::BTreeMap;

use csv::Writer;
use std::fs::File;
use std::path::PathBuf;

// #[derive(Debug, Deserialize)]
// struct UpdateRecord {
//     symbol: String,
//     timestamp: i64,
//     #[serde(skip_deserializing)]
//     first_update_id: i64,
//     #[serde(skip_deserializing)]
//     last_update_id: i64,
//     side: Side,
//     update_type: String,
//     price: Decimal,
//     qty: Decimal,
//     pu: i64,
// }

// #[derive(Debug, Deserialize)]
// struct SnapRecord {
//     symbol: String,
//     timestamp: i64,
//     first_update_id: i64,
//     last_update_id: i64,
//     side: Side,
//     update_type: String,
//     price: Decimal,
//     qty: Decimal,
//     pu: i8,
// }

// #[derive(Debug, Deserialize)]
// struct SavedOrderbook {
//     bids: Vec<OrderbookLevel>,
//     asks: Vec<OrderbookLevel>,
// }

// #[derive(Debug, Deserialize)]
// struct Orderbook {
//     // time, level (price), size
//     bids: BTreeMap<Decimal, OrderbookLevel>,
//     asks: BTreeMap<Decimal, OrderbookLevel>,
// }

// impl Orderbook {
//     fn first_n_bids(&self, n: i64) -> BTreeMap<Decimal, OrderbookLevel> {
//         // create an n lenght btreemap of the last n bids
//         let mut result = BTreeMap::new();
//         let mut i = 0;
//         for (price, level) in self.bids.iter().rev() {
//             if i < n {
//                 result.insert(*price, *level);
//                 i += 1;
//             } else {
//                 break;
//             }
//         }
//         result
//     }

//     fn first_n_asks(&self, n: i64) -> BTreeMap<Decimal, OrderbookLevel> {
//         let mut result = BTreeMap::new();
//         let mut i = 0;
//         for (price, level) in self.asks.iter() {
//             if i < n {
//                 result.insert(*price, *level);
//                 i += 1;
//             } else {
//                 break;
//             }
//         }
//         result
//     }

//     fn get_nth_ask(&self, n: i64) -> Option<OrderbookLevel> {
//         let mut i = 0;
//         for (_, level) in self.asks.iter() {
//             if i == n {
//                 return Some(*level);
//             }
//             i += 1;
//         }
//         None
//     }

//     fn get_nth_bid(&self, n: i64) -> Option<OrderbookLevel> {
//         let mut i = 0;
//         for (_, level) in self.bids.iter().rev() {
//             if i == n {
//                 return Some(*level);
//             }
//             i += 1;
//         }
//         None
//     }
// }

// #[derive(Debug, Deserialize, PartialEq, Serialize, Copy, Clone)]
// enum Side {
//     a,
//     b,
// }

// #[derive(Debug, Deserialize, Serialize, Copy, Clone, PartialEq)]
// struct OrderbookLevel {
//     level_update: i64,
//     side: Side,
//     size: Decimal,
//     price: Decimal,
// }

// fn parse_snapshot(path: &str, min_timestamp: i64) -> (i64, Orderbook) {
//     let mut rdr = csv::Reader::from_path(path).unwrap();
//     let mut orderbook = Orderbook {
//         bids: BTreeMap::new(),
//         asks: BTreeMap::new(),
//     };
//     let mut recorded_timestamp = 0;

//     for result in rdr.deserialize() {
//         let record: SnapRecord = match result {
//             Ok(r) => r,
//             Err(e) => {
//                 println!("Error: {}", e);
//                 continue;
//             }
//         };

//         // if( min_timestamp <= record.timestamp) & (recorded_timestamp == 0){
//         //     recorded_timestamp = record.timestamp;
//         // }
//         // else if  (recorded_timestamp != record.timestamp) & (recorded_timestamp != 0) {
//         //     break;
//         // }

//         if record.timestamp >= min_timestamp {
//             if recorded_timestamp == 0 {
//                 recorded_timestamp = record.timestamp;
//             } else if recorded_timestamp != record.timestamp {
//                 break;
//             }
//             let mut level = OrderbookLevel {
//                 level_update: record.timestamp,
//                 side: record.side,
//                 size: record.qty,
//                 price: record.price,
//             };

//             if level.side == Side::a {
//                 orderbook.asks.insert(record.price, level);
//             } else {
//                 orderbook.bids.insert(record.price, level);
//             }
//         }
//     }
//     (recorded_timestamp, orderbook)
// }

// fn parse_updates(
//     update_path: &str,
//     start_timestamp: i64,
//     end_timestamp: i64,
//     tracked_levels: i64,
//     mut full_orderbook: Orderbook,
//     mut writer: Writer<File>,
// ) -> Orderbook {
//     let mut rdr = csv::Reader::from_path(update_path).unwrap();

//     let mut nth_bid = full_orderbook.get_nth_bid(tracked_levels).unwrap();
//     let mut nth_ask = full_orderbook.get_nth_ask(tracked_levels).unwrap();

//     for result in rdr.deserialize() {
//         let record: UpdateRecord = match result {
//             Ok(r) => r,
//             Err(e) => {
//                 println!("Error: {}", e);
//                 continue;
//             }
//         };

//         if (record.timestamp > start_timestamp) & (record.timestamp < end_timestamp) {
//             let level = OrderbookLevel {
//                 level_update: record.timestamp,
//                 side: record.side,
//                 size: record.qty,
//                 price: record.price,
//             };

//             if record.side == Side::a {
//                 if level.size == dec!(0) {
//                     full_orderbook.asks.remove(&record.price);
//                     nth_ask = full_orderbook.get_nth_ask(tracked_levels).unwrap();
//                     nth_ask.level_update = record.timestamp;
//                     writer.serialize(level).unwrap();
//                     writer.serialize(nth_ask).unwrap();
//                 } else {
//                     full_orderbook.asks.insert(record.price, level);
//                     if record.price <= nth_ask.price {
//                         writer.serialize(level).unwrap();
//                     }
//                 }
//             } else {
//                 if level.size == dec!(0) {
//                     full_orderbook.bids.remove(&record.price);
//                     nth_bid = full_orderbook.get_nth_bid(tracked_levels).unwrap();
//                     nth_bid.level_update = record.timestamp;
//                     writer.serialize(level).unwrap();
//                     writer.serialize(nth_bid).unwrap();
//                 } else {
//                     full_orderbook.bids.insert(record.price, level);
//                     if record.price >= nth_bid.price {
//                         writer.serialize(level).unwrap();
//                     }
//                 }
//             }
//         } else if record.timestamp > end_timestamp {
//             break;
//         }
//     }
//     full_orderbook
// }

// fn parse_snapshot_and_updates() {
//     let path_snap = "/home/juuso/Documents/gradu/ADAUSDT_T_DEPTH_2021-10-30/ADAUSDT_T_DEPTH_2021-10-30_depth_snap.csv";
//     let path_update = "/home/juuso/Documents/gradu/ADAUSDT_T_DEPTH_2021-10-30/ADAUSDT_T_DEPTH_2021-10-30_depth_update.csv";
//     // let path_snap = "/media/juuso/5655B83E58A8FD4F/ADAUSDT_T_DEPTH_202211031113(1)/ADAUSDT_T_DEPTH_2021-12-21/ADAUSDT_T_DEPTH_2021-12-21_depth_snap.csv";
//     // let path_update = "/media/juuso/5655B83E58A8FD4F/ADAUSDT_T_DEPTH_202211031113(1)/ADAUSDT_T_DEPTH_2021-12-21/ADAUSDT_T_DEPTH_2021-12-21_depth_update.csv";

//     let target_path = "/home/juuso/Documents/gradu/ADAUSDT_T_DEPTH_2021-10-30/parsed.csv";
//     let mut tracked_levels = 25;

//     let mut writer = csv::Writer::from_path(target_path).unwrap();
//     let (start_timestamp, mut full_orderbook) = parse_snapshot(&path_snap, 0);

//     let open_bids = full_orderbook.first_n_bids(tracked_levels);
//     let open_asks = full_orderbook.first_n_asks(tracked_levels);

//     for (_, level) in open_bids.iter() {
//         writer.serialize(level).unwrap()
//     }
//     for (_, level) in open_asks.iter() {
//         writer.serialize(level).unwrap()
//     }
//     // print orderbook
//     println!("{:?}", full_orderbook);
//     parse_updates(
//         &path_update,
//         start_timestamp,
//         std::i64::MAX,
//         25,
//         &mut full_orderbook,
//         writer,
//     );
// }

// fn read_rowcount(path: &str) -> usize {
//     let mut rdr = csv::Reader::from_path(path).unwrap();
//     let mut count = 0;
//     for _ in rdr.records() {
//         count += 1;
//     }
//     count
// }

// use std::fs;

// fn parse_all_files() {
//     let folder_path = "/media/juuso/5655B83E58A8FD4F/ADAUSDT_T_DEPTH_202211031113(1)";
//     let paths = fs::read_dir(folder_path).unwrap();
//     for path in paths {
//         // take last 10 characters of path
//         let path = path.unwrap().path().to_str().unwrap().to_string();
//         let path = &path[path.len() - 10..];
//         println!("{}", path);
//     }
// }

// fn verify_snapshot_update_match() {
//     let path_snap = "/home/juuso/Documents/gradu/ADAUSDT_T_DEPTH_2021-10-30/ADAUSDT_T_DEPTH_2021-10-30_depth_snap.csv";
//     let path_update = "/home/juuso/Documents/gradu/ADAUSDT_T_DEPTH_2021-10-30/ADAUSDT_T_DEPTH_2021-10-30_depth_update.csv";
//     let target_path = "/home/juuso/Documents/gradu/ADAUSDT_T_DEPTH_2021-10-30/parsed.csv";
//     // read all values from snapshot and save unique timestamps to vector
//     let mut snap_timestamps = Vec::new();
//     let mut snap_reader = csv::Reader::from_path(path_snap).unwrap();
//     for result in snap_reader.deserialize() {
//         let record: SnapRecord = match result {
//             Ok(r) => r,
//             Err(e) => {
//                 println!("Error: {}", e);
//                 continue;
//             }
//         };
//         snap_timestamps.push(record.timestamp);
//     }
//     // sort vector and remove duplicates
//     snap_timestamps.sort();
//     snap_timestamps.dedup();

//     let value_counts = 5;

//     for n in 0..snap_timestamps.len() - 1 {
//         let mut writer = csv::Writer::from_path(target_path).unwrap();
//         let start_timestamp = snap_timestamps[n];
//         let end_timestamp = snap_timestamps[n + 1];
//         let (start_timestamp_parsed, mut full_orderbook_start) =
//             parse_snapshot(&path_snap, start_timestamp);

//         let iterate_bids = full_orderbook_start.first_n_bids(value_counts);

//         full_orderbook_start = parse_updates(
//             &path_update,
//             start_timestamp,
//             end_timestamp,
//             25,
//             full_orderbook_start,
//             writer,
//         );
//         let (end_timestamp_parsed, mut full_orderbook_end) =
//             parse_snapshot(&path_snap, end_timestamp);

//         let iterate_bids = full_orderbook_start.first_n_bids(value_counts);
//         let iterate_asks = full_orderbook_start.first_n_asks(value_counts);
//         let snapshot_bids = full_orderbook_end.first_n_bids(value_counts);
//         let snapshot_asks = full_orderbook_end.first_n_asks(value_counts);

//         for (k, v) in &iterate_bids {
//             if snapshot_bids.contains_key(k) {
//                 let snapshot_value = snapshot_bids.get(k).unwrap();
//                 if v.price != snapshot_value.price || v.size != snapshot_value.size {
//                     println!("BID MISMATCH");
//                     println!("{} {} {}", v.level_update, v.price, v.size);
//                     println!(
//                         "{} {} {}",
//                         snapshot_value.level_update, snapshot_value.price, snapshot_value.size
//                     );
//                 }
//             } else {
//                 println!("BID MISSING {}", k);
//             }
//         }

//         // replicate above for asks
//         for (k, v) in &iterate_asks {
//             if snapshot_asks.contains_key(k) {
//                 let snapshot_value = snapshot_asks.get(k).unwrap();
//                 if v.price != snapshot_value.price || v.size != snapshot_value.size {
//                     println!("ASK MISMATCH");
//                     println!("{} {} {}", k, v.price, v.size);
//                     println!("{} {} {}", k, snapshot_value.price, snapshot_value.size);
//                 }
//             } else {
//                 println!("ASK MISSING {}", k);
//             }
//         }
//         print!("{} ", n);
//     }
// }

fn parse_data_v1() {
    // v1 parses data as follows:
    // OB full state is recorded every n timestamps (where n is milliseconds), recording is OB state at end of the timestamp
    // For each timestamp, if there is a trade event, the orderbook state is recorded before the trade after that next OB is recorded
    //  - OB(t-1)
    //  - trade(t)
    //  - OB(t)
}

mod file_util;
mod orderbook_util;
mod parse_util;

use file_util::{get_file_date, get_first_snapshot_file, get_folder_update_files};
// use orderbook_util::Orderbook;
use parse_util::{parse_snapshot, parse_updates_v2};

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
    let target_path = PathBuf::from("/home/juuso/Documents/gradu/parsed_data/orderbook");
    let tracked_levels = 25;
    let timestamp_aggregation = 10;

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
fn main() {
    parse_data_v2();
}
