use super::parse_util::{Side, UpdateRecord};
use csv::Writer;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;

#[derive(Debug, Deserialize, Clone)]
pub struct Orderbook {
    // time, level (price), size
    pub bids: BTreeMap<Decimal, OrderbookLevel>,
    pub asks: BTreeMap<Decimal, OrderbookLevel>,
}

#[derive(Debug, Serialize)]
enum SnapRow {
    Timestamp(i64),
    Level(Decimal),
}

#[derive(Debug, Serialize)]
enum FullSnapRow {
    Timestamp(i64),
    PriceSize(Decimal),
}

impl Orderbook {
    pub fn new() -> Self {
        Orderbook {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
        }
    }

    pub fn update(&mut self, level: OrderbookLevel) {
        match level.side {
            Side::Bid => {
                if level.size == dec!(0.0) {
                    self.bids.remove(&level.price);
                } else {
                    self.bids.insert(level.price, level);
                }
            }
            Side::Ask => {
                if level.size == dec!(0.0) {
                    self.asks.remove(&level.price);
                } else {
                    self.asks.insert(level.price, level);
                }
            }
        }
    }

    pub fn update_parse_record(&mut self, record: &UpdateRecord) {
        let level = OrderbookLevel::from_update_record(record);
        self.update(level);
    }

    pub fn get_midprice(&self) -> Decimal {
        let bid = self.get_nth_bid(0).unwrap();
        let ask = self.get_nth_ask(0).unwrap();
        (bid.price + ask.price) / dec!(2.0)
    }

    pub fn get_best_bid(&self) -> Option<OrderbookLevel> {
        self.get_nth_bid(0)
    }

    pub fn get_best_ask(&self) -> Option<OrderbookLevel> {
        self.get_nth_ask(0)
    }

    pub fn first_n_bids(&self, n: i64) -> BTreeMap<Decimal, OrderbookLevel> {
        // create an n lenght btreemap of the last n bids
        let mut result = BTreeMap::new();
        let mut i = 0;
        for (price, level) in self.bids.iter().rev() {
            if i < n {
                result.insert(*price, *level);
                i += 1;
            } else {
                break;
            }
        }
        result
    }

    pub fn first_n_asks(&self, n: i64) -> BTreeMap<Decimal, OrderbookLevel> {
        let mut result = BTreeMap::new();
        let mut i = 0;
        for (price, level) in self.asks.iter() {
            if i < n {
                result.insert(*price, *level);
                i += 1;
            } else {
                break;
            }
        }
        result
    }

    pub fn get_nth_ask(&self, n: i64) -> Option<OrderbookLevel> {
        let mut i = 0;
        for (_, level) in self.asks.iter() {
            if i == n {
                return Some(*level);
            }
            i += 1;
        }
        println!("No {} th ask found", n);
        None
    }

    pub fn get_nth_bid(&self, n: i64) -> Option<OrderbookLevel> {
        let mut i = 0;
        for (_, level) in self.bids.iter().rev() {
            if i == n {
                return Some(*level);
            }
            i += 1;
        }
        println!("Could not find {} th bid", n);
        None
    }

    pub fn write_snapshot_price_size(
        &self,
        mut writer: Writer<File>,
        n: i64,
        timestamp: i64,
    ) -> Writer<File> {
        let mut to_write: Vec<FullSnapRow> = Vec::new();
        to_write.push(FullSnapRow::Timestamp(timestamp));
        let mid_price = self.get_midprice();
        to_write.push(FullSnapRow::PriceSize(mid_price));
        let bids = self.first_n_bids(n);
        let asks = self.first_n_asks(n);
        for (price, level) in bids.iter() {
            to_write.push(FullSnapRow::PriceSize(*price));
            to_write.push(FullSnapRow::PriceSize(level.size));
        }
        for (price, level) in asks.iter() {
            to_write.push(FullSnapRow::PriceSize(*price));
            to_write.push(FullSnapRow::PriceSize(level.size));
        }
        writer.serialize(to_write).unwrap();
        writer
    }

    pub fn write_snapshot_prices(
        &self,
        mut writer: Writer<File>,
        n: i64,
        timestamp: i64,
    ) -> Writer<File> {
        let bids = self.first_n_bids(n);
        let asks = self.first_n_asks(n);

        let mut to_write: Vec<SnapRow> = Vec::new();
        to_write.push(SnapRow::Timestamp(timestamp));
        for (price, _) in bids.iter() {
            to_write.push(SnapRow::Level(*price));
        }
        for (price, _) in asks.iter() {
            to_write.push(SnapRow::Level(*price));
        }
        writer.serialize(to_write).unwrap();
        writer
    }

    pub fn last_timestamp(&self) -> i64 {
        let mut last_timestamp = 0;
        for (_, level) in self.bids.iter() {
            if level.timestamp > last_timestamp {
                last_timestamp = level.timestamp;
            }
        }
        for (_, level) in self.asks.iter() {
            if level.timestamp > last_timestamp {
                last_timestamp = level.timestamp;
            }
        }
        last_timestamp
    }

    pub fn get_bid_count(&self) -> i64 {
        self.bids.len() as i64
    }

    pub fn get_ask_count(&self) -> i64 {
        self.asks.len() as i64
    }

    pub fn get_orderbook_state(&self) -> (i64, i64) {
        (self.get_bid_count(), self.get_ask_count())
    }

    pub fn get_partial_orderbook(&self, n: i64) -> Orderbook {
        let mut result = Orderbook::new();
        let mut i = 0;
        for (price, level) in self.bids.iter().rev() {
            if i < n {
                result.bids.insert(*price, *level);
                i += 1;
            } else {
                break;
            }
        }
        let mut i = 0;
        for (price, level) in self.asks.iter() {
            if i < n {
                result.asks.insert(*price, *level);
                i += 1;
            } else {
                break;
            }
        }
        result
    }

    pub fn get_worst_bid(&self) -> Option<OrderbookLevel> {
        let mut worst_bid = None;
        for (_, level) in self.bids.iter().rev() {
            if worst_bid.is_none() {
                worst_bid = Some(*level);
            } else {
                if level.price > worst_bid.unwrap().price {
                    worst_bid = Some(*level);
                }
            }
        }
        worst_bid
    }

    pub fn get_worst_ask(&self) -> Option<OrderbookLevel> {
        let mut worst_ask = None;
        for (_, level) in self.asks.iter() {
            if worst_ask.is_none() {
                worst_ask = Some(*level);
            } else {
                if level.price < worst_ask.unwrap().price {
                    worst_ask = Some(*level);
                }
            }
        }
        worst_ask
    }

    pub fn get_current_levels(&self, n: i64) -> Vec<Decimal> {
        let mut result = Vec::new();
        for i in self.first_n_bids(n) {
            result.push(i.0);
        }
        for i in self.first_n_asks(n) {
            result.push(i.0);
        }
        result
    }

    pub fn get_imbalance(&self, n: i64) -> Option<Decimal> {
        // calculate order book imbalance for first n levels of bids and asks
        let mut bid_size = Decimal::new(0, 0);
        let mut ask_size = Decimal::new(0, 0);
        let mut i = 0;
        for (_, level) in self.bids.iter().rev() {
            if i < n {
                bid_size += level.size;
                i += 1;
            } else {
                break;
            }
        }
        let mut i = 0;
        for (_, level) in self.asks.iter() {
            if i < n {
                ask_size += level.size;
                i += 1;
            } else {
                break;
            }
        }
        if bid_size + ask_size == Decimal::new(0, 0) {
            None
        } else {
            // round to 4 decimal places
            Some(((bid_size - ask_size) / (bid_size + ask_size)).round_dp(4))
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Copy, Clone, PartialEq, Eq, Hash)]
pub struct OrderbookLevel {
    pub timestamp: i64,
    pub side: Side,
    pub size: Decimal,
    pub price: Decimal,
}

impl OrderbookLevel {
    pub fn partial(&self) -> OrderbookLevelPartial {
        OrderbookLevelPartial {
            timestamp: self.timestamp,
            side: self.side,
            price: self.price,
        }
    }
    pub fn from_update_record(record: &UpdateRecord) -> OrderbookLevel {
        OrderbookLevel {
            timestamp: record.timestamp,
            side: record.side,
            size: record.qty,
            price: record.price,
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Copy, Clone, PartialEq, Eq, Hash)]
pub struct OrderbookLevelPartial {
    pub timestamp: i64,
    pub side: Side,
    pub price: Decimal,
}
