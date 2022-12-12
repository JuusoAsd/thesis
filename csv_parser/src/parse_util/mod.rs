use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::fs::File;
use std::path::PathBuf;
use std::time::SystemTime;

use csv::Writer;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};

use super::file_util::get_file_date;
use super::orderbook_util::{Orderbook, OrderbookLevel, OrderbookLevelPartial};

#[derive(Debug, Deserialize)]
struct UpdateRecord {
    symbol: String,
    timestamp: i64,
    #[serde(skip_deserializing)]
    first_update_id: i64,
    #[serde(skip_deserializing)]
    last_update_id: i64,
    side: Side,
    update_type: String,
    price: Decimal,
    qty: Decimal,
    pu: i64,
}

#[derive(Debug, Deserialize)]
struct SnapRecord {
    symbol: String,
    timestamp: i64,
    first_update_id: i64,
    last_update_id: i64,
    side: Side,
    update_type: String,
    price: Decimal,
    qty: Decimal,
    #[serde(skip_deserializing)]
    pu: i8,
}

#[derive(Debug, Deserialize, PartialEq, Serialize, Copy, Clone, Eq, Hash)]
pub enum Side {
    #[serde(rename = "b")]
    Bid,
    #[serde(rename = "a")]
    Ask,
}

pub fn parse_snapshot(path: &PathBuf, min_timestamp: i64) -> (i64, Orderbook) {
    let mut rdr = match csv::Reader::from_path(path) {
        Ok(r) => r,
        Err(e) => {
            panic!("Error reading snapshot file: {}", e);
        }
    };
    let mut orderbook = Orderbook {
        bids: BTreeMap::new(),
        asks: BTreeMap::new(),
    };
    let mut recorded_timestamp = 0;

    for result in rdr.deserialize() {
        let record: SnapRecord = match result {
            Ok(r) => r,
            Err(e) => {
                println!("Error: {}", e);
                continue;
            }
        };
        if record.timestamp >= min_timestamp {
            if recorded_timestamp == 0 {
                recorded_timestamp = record.timestamp;
            } else if recorded_timestamp != record.timestamp {
                break;
            }
            let mut level = OrderbookLevel {
                timestamp: record.timestamp,
                side: record.side,
                size: record.qty,
                price: record.price,
            };

            if level.side == Side::Ask {
                orderbook.asks.insert(record.price, level);
            } else {
                orderbook.bids.insert(record.price, level);
            }
        }
    }
    if recorded_timestamp == 0 {
        panic!("No snapshot found for timestamp {:?}", min_timestamp);
    }
    (recorded_timestamp, orderbook)
}

pub fn parse_updates(
    update_path: &str,
    start_timestamp: i64,
    end_timestamp: i64,
    tracked_levels: i64,
    mut orderbook: Orderbook,
    mut writer: Writer<File>,
) -> Orderbook {
    let mut rdr = csv::Reader::from_path(update_path).unwrap();
    let mut nth_bid = match orderbook.get_nth_bid(tracked_levels) {
        Some(level) => level,
        None => {
            println!("{:?}", orderbook);
            panic!(
                "Not enough bids, looking for {:?}, found {:?}",
                tracked_levels,
                orderbook.get_bid_count()
            )
        }
    };
    let mut nth_ask = match orderbook.get_nth_ask(tracked_levels) {
        Some(level) => level,
        None => {
            panic!(
                "Not enough asks, looking for {:?}, found {:?}",
                tracked_levels,
                orderbook.get_ask_count()
            );
        }
    };

    let mut best_bid = orderbook.get_nth_bid(0).unwrap();
    let mut best_ask = orderbook.get_nth_ask(0).unwrap();

    let mut update_count = 0;
    let mut current_update_set: HashMap<OrderbookLevelPartial, OrderbookLevel> = HashMap::new();
    let mut current_timestamp = 0;

    for result in rdr.deserialize() {
        let record: UpdateRecord = match result {
            Ok(r) => r,
            Err(e) => {
                println!("Error reading update: {}", e);
                continue;
            }
        };
        if (record.timestamp > start_timestamp) & (record.timestamp < end_timestamp) {
            update_count += 1;
            let level = OrderbookLevel {
                timestamp: record.timestamp,
                side: record.side,
                size: record.qty,
                price: record.price,
            };

            if current_timestamp == 0 {
                current_timestamp = record.timestamp;
            } else if current_timestamp != record.timestamp {
                // write the current update set to the csv
                for (_, level) in current_update_set.iter() {
                    let _ = writer.serialize(level);
                }
                current_update_set.clear();
                current_timestamp = record.timestamp;
            }

            match record.side {
                Side::Ask => {
                    if level.size == dec!(0) {
                        // remove if zero
                        orderbook.asks.remove(&level.price);
                    } else {
                        // update if not zero
                        orderbook.asks.insert(level.price, level);
                    }
                    // if record price belongs in top n, we are writing the update
                    // TODO: also need to pop out the nth level if this is a new addition (for example ask go down)
                    if record.price <= nth_ask.price {
                        if level.size == dec!(0) {
                            current_update_set.insert(level.partial(), level);
                            // if the level is zero, we need to find the next nth ask and write that information
                            nth_ask = orderbook.get_nth_ask(tracked_levels).unwrap();
                            nth_ask.timestamp = record.timestamp;
                            current_update_set.insert(nth_ask.partial(), nth_ask);
                        } else if level.price < best_ask.price {
                            // if price is better than best ask we remove worst ask and replace it with new best ask
                            nth_ask.size = dec!(0);
                            nth_ask.timestamp = record.timestamp;
                            current_update_set.insert(nth_ask.partial(), nth_ask);
                            current_update_set.insert(level.partial(), level);
                            nth_ask = orderbook.get_nth_ask(tracked_levels).unwrap();
                            best_ask = orderbook.get_nth_ask(0).unwrap();
                        }
                    }
                }
                Side::Bid => {
                    if level.size == dec!(0) {
                        orderbook.bids.remove(&level.price);
                    } else {
                        orderbook.bids.insert(level.price, level);
                    }
                    if record.price >= nth_bid.price {
                        if level.size == dec!(0) {
                            current_update_set.insert(level.partial(), level);
                            nth_bid = orderbook.get_nth_bid(tracked_levels).unwrap();
                            nth_bid.timestamp = record.timestamp;
                            current_update_set.insert(nth_bid.partial(), nth_bid);
                        } else if level.price > best_bid.price {
                            nth_bid.size = dec!(0);
                            nth_bid.timestamp = record.timestamp;
                            current_update_set.insert(nth_bid.partial(), nth_bid);
                            current_update_set.insert(level.partial(), level);
                            nth_bid = orderbook.get_nth_bid(tracked_levels).unwrap();
                            best_bid = orderbook.get_nth_bid(0).unwrap();
                        }
                    }
                }
            };
        } else if record.timestamp > end_timestamp {
            println!("Reached end of time range {}", record.timestamp);
            break;
        }
    }
    println!("Processed {:?} updates", update_count);
    return orderbook;
}

fn compare_orderbooks(start: Orderbook, end: Orderbook, timestamp: i64) -> Vec<OrderbookLevel> {
    // find out what updates start needs to get to end
    // iterate through end orderbook
    //  - if level in start
    //      - if size is different, write update
    //      - if size is same, do nothing
    //  - if level not in start
    //      - write update

    let mut update_vec = Vec::new();
    for (price, level) in &start.bids {
        match end.bids.get(price) {
            Some(_) => continue,
            None => {
                let mut level_clone = level.clone();
                level_clone.timestamp = timestamp;
                level_clone.size = dec!(0);
                update_vec.push(level_clone);
            }
        }
    }

    for (price, mut level) in end.bids {
        match start.bids.get(&price) {
            Some(start_level) => {
                if start_level.size != level.size {
                    level.timestamp = timestamp;
                    update_vec.push(level);
                }
            }
            None => {
                level.timestamp = timestamp;
                update_vec.push(level);
            }
        }
    }

    for (price, level) in &start.asks {
        match end.asks.get(price) {
            Some(_) => continue,
            None => {
                let mut level_clone = level.clone();
                level_clone.timestamp = timestamp;
                level_clone.size = dec!(0);
                update_vec.push(level_clone);
            }
        }
    }

    for (price, mut level) in end.asks {
        match start.asks.get(&price) {
            Some(start_level) => {
                if start_level.size != level.size {
                    level.timestamp = timestamp;
                    update_vec.push(level);
                }
            }
            None => {
                level.timestamp = timestamp;
                update_vec.push(level);
            }
        }
    }

    update_vec
}

// pub fn parse_updates_v3(
//     update_path: &str,
//     start_timestamp: i64,
//     end_timestamp: i64,
//     tracked_levels: i64,
//     timestamp_aggregation: i64,
//     mut orderbook: Orderbook,
//     mut writer: Writer<File>,
// ) -> Orderbook {
//     let mut update_count = 0;
//     let mut record_start_timestamp = 0;
//     let start_time = SystemTime::now();

//     let mut rdr = csv::Reader::from_path(update_path).unwrap();
//     let mut partial_orderbook = orderbook.get_partial_orderbook(tracked_levels);
//     // Iterate over lines to find non-duplicate updates for each timestamp
//     for result in rdr.deserialize() {
//         let record: UpdateRecord = match result {
//             Ok(r) => r,
//             Err(e) => {
//                 println!("Error reading update: {}", e);
//                 continue;
//             }
//         };
//         // only allow valid timestamps that fit within the range
//         if (record.timestamp > start_timestamp) & (record.timestamp < end_timestamp) {
//             update_count += 1;
//             if update_count % 1_000_000 == 0 {
//                 println!(
//                     "Processed {} updates in {} seconds",
//                     update_count,
//                     start_time.elapsed().unwrap().as_secs()
//                 );
//             }
//             let level = OrderbookLevel {
//                 timestamp: record.timestamp,
//                 side: record.side,
//                 size: record.qty,
//                 price: record.price,
//             };

//             if record_start_timestamp == 0 {
//                 record_start_timestamp = record.timestamp;

//             // When reach new timestamp, parse current update set
//             } else if (record_start_timestamp + timestamp_aggregation) <= record.timestamp {
//                 let mut updates = compare_orderbooks(
//                     partial_orderbook,
//                     orderbook.get_partial_orderbook(tracked_levels),
//                     record.timestamp,
//                 );
//                 for update in updates {
//                     writer.serialize(update).unwrap();
//                 }
//                 record_start_timestamp = record.timestamp;
//                 partial_orderbook = orderbook.get_partial_orderbook(tracked_levels);
//             } else {
//                 match record.side {
//                     Side::Ask => {
//                         if level.size == dec!(0) {
//                             // remove if zero
//                             orderbook.asks.remove(&level.price);
//                         } else {
//                             // update if not zero
//                             orderbook.asks.insert(level.price, level);
//                         }
//                     }
//                     Side::Bid => {
//                         if level.size == dec!(0) {
//                             orderbook.bids.remove(&level.price);
//                         } else {
//                             orderbook.bids.insert(level.price, level);
//                         }
//                     }
//                 };
//             }
//         } else if record.timestamp > end_timestamp {
//             break;
//         }
//     }
//     orderbook
// }

// pub fn parse_updates_snapshot(
//     update_path: &str,
//     start_timestamp: i64,
//     end_timestamp: i64,
//     tracked_levels: i64,
//     timestamp_aggregation: i64,
//     mut orderbook: Orderbook,
//     mut writer: Writer<File>,
// ) -> Orderbook {
//     // Baseline snapshot parsing.
//     // From stream of updates, track the current state in memory.
//     // every timestamp_aggregation, record the current state for best "tracked_levels" count of bid and ask
//     // TODO: timestamps might have gaps, even with varying timestamps, recording state should happen at a constant interval
//     //  - example: for data with timestamps from 0 to 10_000, with timestamp_aggregation of 10, should have 1_000 recordings
//     let mut update_count = 0;
//     let mut record_start_timestamp = 0;
//     let start_time = SystemTime::now();
//     let mut rdr = csv::Reader::from_path(update_path).unwrap();
//     for result in rdr.deserialize() {
//         let record: UpdateRecord = match result {
//             Ok(r) => r,
//             Err(e) => {
//                 println!("Error reading update: {}", e);
//                 continue;
//             }
//         };
//         // only allow valid timestamps that fit within the range
//         if (record.timestamp > start_timestamp) & (record.timestamp < end_timestamp) {
//             update_count += 1;
//             if update_count % 1_000_000 == 0 {
//                 println!(
//                     "Processed {} updates in {} seconds",
//                     update_count,
//                     start_time.elapsed().unwrap().as_secs()
//                 );
//             }
//             let level = OrderbookLevel {
//                 timestamp: record.timestamp,
//                 side: record.side,
//                 size: record.qty,
//                 price: record.price,
//             };

//             if record_start_timestamp == 0 {
//                 record_start_timestamp = record.timestamp;

//             // When reach a timestamp after aggregation window, write current snapshot of orderbook
//             } else if (record_start_timestamp + timestamp_aggregation) <= record.timestamp {
//                 writer = orderbook.write_full_snapshot(writer, tracked_levels, record.timestamp);
//                 record_start_timestamp = record.timestamp;
//             } else {
//                 match record.side {
//                     Side::Ask => {
//                         if level.size == dec!(0) {
//                             // remove if zero
//                             orderbook.asks.remove(&level.price);
//                         } else {
//                             // update if not zero
//                             orderbook.asks.insert(level.price, level);
//                         }
//                     }
//                     Side::Bid => {
//                         if level.size == dec!(0) {
//                             orderbook.bids.remove(&level.price);
//                         } else {
//                             orderbook.bids.insert(level.price, level);
//                         }
//                     }
//                 };
//             }
//         } else if record.timestamp > end_timestamp {
//             break;
//         }
//     }
//     orderbook
// }

pub fn parse_updates_v2(
    start_timestamp: i64,
    end_timestamp: i64,
    tracked_levels: i64,
    timestamp_aggregation: i64,
    target_path: &PathBuf,
    update_paths: Vec<PathBuf>,
    mut orderbook: Orderbook,
) -> () {
    let mut update_count = 0;
    let mut record_start_timestamp = 0;
    let start_time = SystemTime::now();

    for path in update_paths {
        let mut rdr = csv::Reader::from_path(&path).unwrap();

        let mut target_base = PathBuf::from(target_path);
        let mut current_date = get_file_date(path.clone().parent().unwrap());
        target_base.push(current_date);
        target_base.set_extension("csv");
        let mut writer = csv::Writer::from_path(target_base).unwrap();

        for result in rdr.deserialize() {
            let record: UpdateRecord = match result {
                Ok(r) => r,
                Err(e) => {
                    println!("Error reading update: {}", e);
                    continue;
                }
            };
            if (record.timestamp > start_timestamp) & (record.timestamp < end_timestamp) {
                update_count += 1;
                if update_count % 1_000_000 == 0 {
                    println!(
                        "Processed {} updates in {} seconds",
                        update_count,
                        start_time.elapsed().unwrap().as_secs()
                    );
                }
                let level = OrderbookLevel {
                    timestamp: record.timestamp,
                    side: record.side,
                    size: record.qty,
                    price: record.price,
                };

                if record_start_timestamp == 0 {
                    record_start_timestamp = record.timestamp;
                } else if record.timestamp > end_timestamp {
                    break;
                }
                match record.side {
                    Side::Ask => {
                        if level.size == dec!(0) {
                            // remove if zero
                            orderbook.asks.remove(&level.price);
                        } else {
                            // update if not zero
                            orderbook.asks.insert(level.price, level);
                        }
                    }
                    Side::Bid => {
                        if level.size == dec!(0) {
                            orderbook.bids.remove(&level.price);
                        } else {
                            orderbook.bids.insert(level.price, level);
                        }
                    }
                };
                if (record_start_timestamp + timestamp_aggregation) <= record.timestamp {
                    writer = orderbook.write_snapshot_price_size(
                        writer,
                        tracked_levels,
                        record.timestamp,
                    );
                    record_start_timestamp = record.timestamp;
                }
            }
        }
    }
}
