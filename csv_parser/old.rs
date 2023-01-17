fn verify_snapshot_update_match() {
    let path_snap = "/home/juuso/Documents/gradu/ADAUSDT_T_DEPTH_2021-10-30/ADAUSDT_T_DEPTH_2021-10-30_depth_snap.csv";
    let path_update = "/home/juuso/Documents/gradu/ADAUSDT_T_DEPTH_2021-10-30/ADAUSDT_T_DEPTH_2021-10-30_depth_update.csv";
    let target_path = "/home/juuso/Documents/gradu/ADAUSDT_T_DEPTH_2021-10-30/parsed.csv";
    // read all values from snapshot and save unique timestamps to vector
    let mut snap_timestamps = Vec::new();
    let mut snap_reader = csv::Reader::from_path(path_snap).unwrap();
    for result in snap_reader.deserialize() {
        let record: SnapRecord = match result {
            Ok(r) => r,
            Err(e) => {
                println!("Error: {}", e);
                continue;
            }
        };
        snap_timestamps.push(record.timestamp);
    }
    // sort vector and remove duplicates
    snap_timestamps.sort();
    snap_timestamps.dedup();

    let value_counts = 5;

    for n in 0..snap_timestamps.len() - 1 {
        let mut writer = csv::Writer::from_path(target_path).unwrap();
        let start_timestamp = snap_timestamps[n];
        let end_timestamp = snap_timestamps[n + 1];
        let (start_timestamp_parsed, mut full_orderbook_start) =
            parse_snapshot(&path_snap, start_timestamp);

        let iterate_bids = full_orderbook_start.first_n_bids(value_counts);

        full_orderbook_start = parse_updates(
            &path_update,
            start_timestamp,
            end_timestamp,
            25,
            full_orderbook_start,
            writer,
        );
        let (end_timestamp_parsed, mut full_orderbook_end) =
            parse_snapshot(&path_snap, end_timestamp);

        let iterate_bids = full_orderbook_start.first_n_bids(value_counts);
        let iterate_asks = full_orderbook_start.first_n_asks(value_counts);
        let snapshot_bids = full_orderbook_end.first_n_bids(value_counts);
        let snapshot_asks = full_orderbook_end.first_n_asks(value_counts);

        for (k, v) in &iterate_bids {
            if snapshot_bids.contains_key(k) {
                let snapshot_value = snapshot_bids.get(k).unwrap();
                if v.price != snapshot_value.price || v.size != snapshot_value.size {
                    println!("BID MISMATCH");
                    println!("{} {} {}", v.timestamp, v.price, v.size);
                    println!(
                        "{} {} {}",
                        snapshot_value.timestamp, snapshot_value.price, snapshot_value.size
                    );
                }
            } else {
                println!("BID MISSING {}", k);
            }
        }

        // replicate above for asks
        for (k, v) in &iterate_asks {
            if snapshot_asks.contains_key(k) {
                let snapshot_value = snapshot_asks.get(k).unwrap();
                if v.price != snapshot_value.price || v.size != snapshot_value.size {
                    println!("ASK MISMATCH");
                    println!("{} {} {}", k, v.price, v.size);
                    println!("{} {} {}", k, snapshot_value.price, snapshot_value.size);
                }
            } else {
                println!("ASK MISSING {}", k);
            }
        }
        print!("{} ", n);
    }
}

'''
            if record.side == Side::Ask {
                if level.size == dec!(0) {
                    orderbook.asks.remove(&record.price);
                    nth_ask = orderbook.get_nth_ask(tracked_levels).unwrap();
                    nth_ask.timestamp = record.timestamp;

                    current_update_set.insert(level.partial(), level);
                    current_update_set.insert(nth_ask.partial(), nth_ask);
                } else {
                    orderbook.asks.insert(record.price, level);
                    if record.price <= nth_ask.price {
                        current_update_set.insert(level.partial(), level);
                    }
                }
            } else {
                if level.size == dec!(0) {
                    orderbook.bids.remove(&record.price);
                    nth_bid = orderbook.get_nth_bid(tracked_levels).unwrap();
                    nth_bid.timestamp = record.timestamp;
                    current_update_set.insert(level.partial(), level);
                    current_update_set.insert(nth_bid.partial(), nth_bid);
                } else {
                    orderbook.bids.insert(record.price, level);
                    if record.price >= nth_bid.price {
                        current_update_set.insert(level.partial(), level);
                    }
                }
            }
'''