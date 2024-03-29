use csv::Writer;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::cmp::{max, min};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::fs::OpenOptions;
use std::path::PathBuf;
use std::time::SystemTime;

use super::file_util::{
    from_str_custom_bool, get_file_date, ASRecord, AggregateRecord, FileHandler, InterimRecord,
};
use super::orderbook_util::{Orderbook, OrderbookLevel, OrderbookLevelPartial};

#[derive(Debug, Deserialize)]
pub struct UpdateRecord {
    symbol: String,
    pub timestamp: i64,
    #[serde(skip_deserializing)]
    first_update_id: i64,
    #[serde(skip_deserializing)]
    last_update_id: i64,
    pub side: Side,
    update_type: String,
    pub price: Decimal,
    pub qty: Decimal,
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

#[derive(Debug, Deserialize)]
pub struct TradeRecord {
    // 557767490,1.2797,832.0,1064.7104,1640131009832,true
    trade_id: i64,
    price: Decimal,
    qty: Decimal,
    total_value: Decimal,
    timestamp: i64,
    // #[serde(deserialize_with = "from_str_custom_bool")]
    // is_buyer_maker: Option<bool>,
    // #[serde(deserialize_with = "from_str_custom_bool")]
    // random: Option<bool>,
}

#[derive(Debug, Deserialize, PartialEq, Serialize, Copy, Clone, Eq, Hash)]
pub enum Side {
    #[serde(rename = "b")]
    Bid,
    #[serde(rename = "a")]
    Ask,
}

pub fn parse_snapshot(path: &PathBuf, min_timestamp: i64) -> Option<(i64, Orderbook)> {
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
        return None;
    }
    Some((recorded_timestamp, orderbook))
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

pub fn rewind_order_book(to_ts: i64, mut order_book: Orderbook, mut files: FileHandler) {
    let mut record: UpdateRecord = match files.next() {
        Some(r) => files.deserialize_value(&r),
        None => panic!("No records found in file for rewind"),
    };

    if record.timestamp < to_ts {
        order_book.update_parse_record(&record);
    }
    loop {
        match files.next() {
            Some(r) => {
                record = files.deserialize_value(&r);
                if record.timestamp < to_ts {
                    order_book.update_parse_record(&record);
                } else {
                    break;
                }
            }
            None => break,
        }
    }
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

pub fn parse_records_avellaneda_stoikov(
    start_timestamp: i64,
    end_timestamp: i64,
    aggregation_interval: i64,
    target_path: &PathBuf,
    mut update_iter: FileHandler,
    mut trade_iter: FileHandler,
    mut orderbook: Orderbook,
) {
    // this function parses the raw data to be directly useable with avellaneda stoikov -model
    // key components:
    //  1) timestamp
    //  2) Current midprice based on best bid and ask price from orderbook
    //  3, 4) if a trade happemed, the price and size of the trade

    // params:
    //  - start state for orderbook
    //  - start timestamp
    //  - end timestamp
    //  - aggregation interval (how often midprice is recorded IF no trades happen)
    //  - target_path: where to write the parsed data
    //  - update_paths: where to read the raw OB updates from
    //  - trade_paths: where to read the raw trade data from

    // NOTE: If orderbook and trade happen at same timestamp, the trade is recorded first and previous orderbook update is used

    // initialze file handler for writing parsed data
    let start_time = SystemTime::now();
    let mut count = 0;
    let mut writer = csv::Writer::from_path(target_path).unwrap();
    let mut previous_timestamp = start_timestamp;

    // start by moving update and trade iterators to start timestamp
    let mut update_record: UpdateRecord = loop {
        let update_record: UpdateRecord = match update_iter.next() {
            Some(r) => update_iter.deserialize_value(&r),
            None => panic!("No updates found in update file for start timestamp"),
        };
        if update_record.timestamp >= start_timestamp {
            break update_record;
        }
    };

    let mut trade_record: TradeRecord = loop {
        let trade_record: TradeRecord = match trade_iter.next() {
            Some(r) => trade_iter.deserialize_value(&r),
            None => panic!("No trades found in trade file for start timestamp"),
        };
        if trade_record.timestamp >= start_timestamp {
            break trade_record;
        }
    };

    let mut recorded_data = ASRecord::new(
        previous_timestamp,
        orderbook.get_midprice(),
        dec!(0),
        dec!(0),
        dec!(0),
        dec!(0),
        Side::Ask,
    );

    // // start looping updates and trades
    loop {
        count += 1;
        if count % 1_000_000 == 0 {
            println!(
                "Processed {} updates in {} seconds",
                count,
                start_time.elapsed().unwrap().as_secs()
            );
        }

        if update_record.timestamp < trade_record.timestamp {
            // always update orderbook state with latest
            orderbook.update_parse_record(&update_record);

            if update_record.timestamp >= previous_timestamp + aggregation_interval {
                // record values to csv here, use dummy trade with ask side and 0 size as no trade happened
                let mut best_bid = orderbook.get_nth_bid(0).unwrap().price;
                let mut best_ask = orderbook.get_nth_ask(0).unwrap().price;
                recorded_data = ASRecord::new(
                    update_record.timestamp,
                    (best_bid + best_ask) / dec!(2),
                    best_bid,
                    best_ask,
                    dec!(0),
                    dec!(0),
                    Side::Ask,
                );
                writer.serialize(recorded_data).unwrap();
                previous_timestamp = update_record.timestamp;
            }

            update_record = match update_iter.next() {
                Some(r) => update_iter.deserialize_value(&r),
                None => break,
            };
        }
        if update_record.timestamp >= trade_record.timestamp {
            // record values to csv here, update previous timestamp
            let mut best_bid = orderbook.get_nth_bid(0).unwrap().price;
            let mut best_ask = orderbook.get_nth_ask(0).unwrap().price;
            recorded_data = ASRecord::new(
                trade_record.timestamp,
                (best_bid + best_ask) / dec!(2),
                best_bid,
                best_ask,
                trade_record.qty,
                trade_record.price,
                Side::Ask,
            );
            writer.serialize(recorded_data).unwrap();
            previous_timestamp = trade_record.timestamp;

            trade_record = match trade_iter.next() {
                Some(r) => trade_iter.deserialize_value(&r),
                None => break,
            };
        }

        if update_record.timestamp >= end_timestamp {
            break;
        }
    }
}

fn progress_iterators(
    mut update_iter: FileHandler,
    mut trade_iter: FileHandler,
    start_timestamp: i64,
) -> (FileHandler, FileHandler, UpdateRecord, TradeRecord) {
    // progress iterator until start timestamp is reached, return iterator and the first record
    let mut update_record: UpdateRecord = loop {
        let update_record: UpdateRecord = match update_iter.next() {
            Some(r) => update_iter.deserialize_value(&r),
            None => panic!("No updates found in update file for start timestamp"),
        };
        if update_record.timestamp >= start_timestamp {
            break update_record;
        }
    };

    let mut trade_record: TradeRecord = loop {
        let trade_record: TradeRecord = match trade_iter.next() {
            Some(r) => trade_iter.deserialize_value(&r),
            None => panic!("No trades found in trade file for start timestamp"),
        };
        if trade_record.timestamp >= start_timestamp {
            break trade_record;
        }
    };
    (update_iter, trade_iter, update_record, trade_record)
}

// pub fn parse_records_aggregate_ts(
//     start_timestamp: i64,
//     end_timestamp: i64,
//     aggregation_interval: i64,
//     target_path: &PathBuf,
//     mut update_iter: FileHandler,
//     mut trade_iter: FileHandler,
//     mut orderbook: Orderbook,
// ) {
//     let start_time = SystemTime::now();
//     let mut count: i64 = 0;
//     let mut writer = csv::Writer::from_path(target_path).unwrap();
//     let mut previous_timestamp = start_timestamp;

//     let (mut update_iter, mut trade_iter, mut update_record, mut trade_record) =
//         progress_iterators(update_iter, trade_iter, start_timestamp);

//     let mut current_timestamp = min(update_record.timestamp, trade_record.timestamp);
//     let mut mid_price = orderbook.get_midprice();
//     let mut low_price = dec!(0);
//     let mut high_price = dec!(0);
//     let mut buy_volume = dec!(0);
//     let mut sell_volume = dec!(0);
//     let mut prev_best_bid = orderbook.get_best_bid().unwrap().price;
//     let mut prev_best_ask = orderbook.get_best_ask().unwrap().price;
//     loop {
//         count += 1;
//         if count % 1_000_000 == 0 {
//             println!(
//                 "Processed {} updates in {} seconds",
//                 count,
//                 start_time.elapsed().unwrap().as_secs()
//             );
//         }
//         if current_timestamp >= previous_timestamp + aggregation_interval {
//             let to_save = AggregateRecord::new(
//                 current_timestamp,
//                 prev_best_bid,
//                 prev_best_ask,
//                 low_price,
//                 high_price,
//                 buy_volume,
//                 sell_volume,
//             );
//             writer.serialize(to_save).unwrap();
//             // record values, zero out values
//             previous_timestamp = current_timestamp;
//             low_price = dec!(0);
//             high_price = dec!(0);
//             buy_volume = dec!(0);
//             sell_volume = dec!(0);
//             prev_best_bid = orderbook.get_best_bid().unwrap().price;
//             prev_best_ask = orderbook.get_best_ask().unwrap().price;
//         } else if trade_record.timestamp < update_record.timestamp
//             && trade_record.timestamp < previous_timestamp + aggregation_interval
//         {
//             // accumulate values from trade
//             if low_price == dec!(0) {
//                 low_price = trade_record.price;
//             } else {
//                 low_price = min(low_price, trade_record.price);
//             }
//             high_price = max(high_price, trade_record.price);

//             if trade_record.price > mid_price {
//                 buy_volume += trade_record.qty;
//             } else {
//                 sell_volume += trade_record.qty;
//             }

//             trade_record = match trade_iter.next() {
//                 Some(r) => trade_iter.deserialize_value(&r),
//                 None => break,
//             };
//         } else if update_record.timestamp <= trade_record.timestamp
//             && update_record.timestamp < previous_timestamp + aggregation_interval
//         {
//             // always update orderbook state with latest
//             orderbook.update_parse_record(&update_record);

//             update_record = match update_iter.next() {
//                 Some(r) => update_iter.deserialize_value(&r),
//                 None => break,
//             };
//             mid_price = orderbook.get_midprice();
//         }

//         current_timestamp = min(update_record.timestamp, trade_record.timestamp);
//     }
// }

// pub fn parse_records_interim_data(
//     start_timestamp: i64,
//     end_timestamp: i64,
//     target_path: &PathBuf,
//     mut update_iter: FileHandler,
//     mut trade_iter: FileHandler,
//     mut orderbook: Orderbook,
// ) {
//     let start_time = SystemTime::now();
//     let mut count: i64 = 0;
//     let mut writer = csv::Writer::from_path(target_path).unwrap();
//     let mut previous_timestamp = start_timestamp;

//     let (mut update_iter, mut trade_iter, mut update_record, mut trade_record) =
//         progress_iterators(update_iter, trade_iter, start_timestamp);

//     let mut current_timestamp = min(update_record.timestamp, trade_record.timestamp);
//     let mut mid_price = orderbook.get_midprice();
//     let mut low_price = dec!(0);
//     let mut high_price = dec!(0);
//     let mut buy_volume = dec!(0);
//     let mut sell_volume = dec!(0);

//     // // start looping updates and trades
//     loop {
//         count += 1;
//         if count % 1_000_000 == 0 {
//             println!(
//                 "Processed {} updates in {} seconds",
//                 count,
//                 start_time.elapsed().unwrap().as_secs()
//             );
//         }
//         // we should update the latest record, be it a trade or orderbook update
//         if update_record.timestamp < trade_record.timestamp {
//             // order book update, we just update the internal book
//             orderbook.update_parse_record(&update_record);
//             update_record = match update_iter.next() {
//                 Some(r) => update_iter.deserialize_value(&r),
//                 None => break,
//             };
//         }
//         if update_record.timestamp >= trade_record.timestamp {
//             // trade update, record the trade and current mid price
//             mid_price = orderbook.get_midprice();
//             let mut recorded_data = InterimRecord::new(
//                 trade_record.timestamp,
//                 mid_price,
//                 trade_record.qty,
//                 trade_record.price,
//                 Side::Ask,
//             );
//             writer.serialize(recorded_data).unwrap();
//             previous_timestamp = trade_record.timestamp;

//             trade_record = match trade_iter.next() {
//                 Some(r) => trade_iter.deserialize_value(&r),
//                 None => break,
//             };
//         }

//         if update_record.timestamp >= end_timestamp {
//             break;
//         }
//     }
// }

use num_format::{Locale, ToFormattedString};
pub fn parse_both_records(
    start_timestamp: i64,
    end_timestamp: i64,
    aggregation_interval: i64,
    mut interim_wtr: Writer<File>,
    mut base_wtr: Writer<File>,
    update_iter: FileHandler,
    trade_iter: FileHandler,
    mut orderbook: Orderbook,
    update_interim: bool,
    update_base: bool,
) {
    let start_time = SystemTime::now();
    let mut count: i64 = 0;
    let mut previous_timestamp = start_timestamp;
    let (mut update_iter, mut trade_iter, mut update_record, mut trade_record) =
        progress_iterators(update_iter, trade_iter, start_timestamp);

    let mut current_timestamp = min(update_record.timestamp, trade_record.timestamp);
    let mut mid_price = orderbook.get_midprice();
    let mut low_price = dec!(0);
    let mut high_price = dec!(0);
    // let mut buy_volume = dec!(0);
    // let mut sell_volume = dec!(0);

    let mut buy_volume = dec!(0);
    let mut sell_volume = dec!(0);
    let mut prev_best_bid = orderbook.get_best_bid().unwrap().price;
    let mut prev_best_ask = orderbook.get_best_ask().unwrap().price;

    loop {
        count += 1;
        if count % 1_000_000 == 0 {
            println!(
                "Processed {} updates in {} seconds",
                count.to_formatted_string(&Locale::en),
                start_time.elapsed().unwrap().as_secs()
            );
        }
        if current_timestamp >= previous_timestamp + aggregation_interval {
            // record values every aggregate interval
            if update_base {
                let imbalance = match orderbook.get_imbalance(25) {
                    Some(i) => i,
                    None => dec!(0),
                };
                let to_save = AggregateRecord::new(
                    current_timestamp,
                    prev_best_bid,
                    prev_best_ask,
                    low_price,
                    high_price,
                    buy_volume,
                    sell_volume,
                    imbalance,
                );
                base_wtr.serialize(to_save).unwrap();
            }

            // record values, zero out values
            previous_timestamp = current_timestamp;
            low_price = dec!(0);
            high_price = dec!(0);
            buy_volume = dec!(0);
            sell_volume = dec!(0);
            prev_best_bid = orderbook.get_best_bid().unwrap().price;
            prev_best_ask = orderbook.get_best_ask().unwrap().price;
        } else if trade_record.timestamp < update_record.timestamp
            && trade_record.timestamp < previous_timestamp + aggregation_interval
        {
            // accumulate values from trade, record interim data
            if low_price == dec!(0) {
                low_price = trade_record.price;
            } else {
                low_price = min(low_price, trade_record.price);
            }
            high_price = max(high_price, trade_record.price);

            if trade_record.price > mid_price {
                buy_volume += trade_record.qty;
            } else {
                sell_volume += trade_record.qty;
            }

            // record interim data
            if update_interim {
                let recorded_data = InterimRecord::new(
                    trade_record.timestamp,
                    mid_price,
                    trade_record.qty,
                    trade_record.price,
                    Side::Ask,
                );
                interim_wtr.serialize(recorded_data).unwrap();
            }

            trade_record = match trade_iter.next() {
                Some(r) => trade_iter.deserialize_value(&r),
                None => break,
            };
        } else if update_record.timestamp <= trade_record.timestamp
            && update_record.timestamp < previous_timestamp + aggregation_interval
        {
            // always update orderbook state with latest, no other actions needed
            orderbook.update_parse_record(&update_record);
            update_record = match update_iter.next() {
                Some(r) => update_iter.deserialize_value(&r),
                None => break,
            };
            mid_price = orderbook.get_midprice();
        }

        if update_record.timestamp >= end_timestamp {
            break;
        }

        current_timestamp = min(update_record.timestamp, trade_record.timestamp);
    }
}
