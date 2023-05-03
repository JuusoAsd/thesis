use super::parse_util::Side;
use csv::{Reader, ReaderBuilder, StringRecord, StringRecordsIntoIter};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::fs::{self, ReadDir};
use std::path::{Path, PathBuf};
use std::vec::IntoIter;
use walkdir::WalkDir;

pub struct FileHandler {
    path_iter: IntoIter<PathBuf>,
    file_iter: StringRecordsIntoIter<File>,
    pub headers: StringRecord,
    has_headers: bool,
}

impl FileHandler {
    pub fn new(files: Vec<PathBuf>, headers: StringRecord, has_headers: bool) -> Self {
        let mut path_iter = files.into_iter();

        let file_reader = if has_headers {
            Reader::from_path(path_iter.next().unwrap()).unwrap()
        } else {
            ReaderBuilder::new()
                .has_headers(false)
                .from_path(path_iter.next().unwrap())
                .unwrap()
        };
        Self {
            path_iter,
            file_iter: file_reader.into_records(),
            headers,
            has_headers,
        }
    }

    pub fn deserialize_value<'a, T>(&'a mut self, value: &'a StringRecord) -> T
    where
        T: Deserialize<'a>,
    {
        let val_deser: T = value.deserialize(Some(&self.headers)).unwrap();
        val_deser
    }
}

impl Iterator for FileHandler {
    type Item = StringRecord;
    fn next(&mut self) -> Option<StringRecord> {
        match self.file_iter.next() {
            Some(Ok(record)) => Some(record),
            Some(Err(e)) => {
                panic!("Error with next of FileHandler: {}", e);
            }
            None => {
                let next_path = match self.path_iter.next() {
                    Some(p) => p,
                    None => return None,
                };
                let file_reader = if self.has_headers {
                    Reader::from_path(next_path).unwrap()
                } else {
                    ReaderBuilder::new()
                        .has_headers(false)
                        .from_path(next_path)
                        .unwrap()
                };
                self.file_iter = file_reader.into_records();
                self.next()
            }
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ASRecord {
    timestamp: i64,
    mid_price: Decimal,
    best_bid: Decimal,
    best_ask: Decimal,
    size: Decimal,
    price: Decimal,
    side: Side,
}

impl ASRecord {
    pub fn new(
        timestamp: i64,
        mid_price: Decimal,
        best_bid: Decimal,
        best_ask: Decimal,
        size: Decimal,
        price: Decimal,
        side: Side,
    ) -> Self {
        Self {
            timestamp,
            mid_price,
            best_bid,
            best_ask,
            size,
            price,
            side,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct AggregateRecord {
    timestamp: i64,
    best_bid: Decimal,
    best_ask: Decimal,
    low_price: Decimal,
    high_price: Decimal,
    buy_volume: Decimal,
    sell_volume: Decimal,
}

impl AggregateRecord {
    pub fn new(
        timestamp: i64,
        best_bid: Decimal,
        best_ask: Decimal,
        low_price: Decimal,
        high_price: Decimal,
        buy_volume: Decimal,
        sell_volume: Decimal,
    ) -> Self {
        Self {
            timestamp,
            best_bid,
            best_ask,
            low_price,
            high_price,
            buy_volume,
            sell_volume,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct InterimRecord {
    timestamp: i64,
    mid_price: Decimal,
    size: Decimal,
    price: Decimal,
    side: Side,
}

impl InterimRecord {
    pub fn new(
        timestamp: i64,
        mid_price: Decimal,
        size: Decimal,
        price: Decimal,
        side: Side,
    ) -> Self {
        Self {
            timestamp,
            mid_price,
            size,
            price,
            side,
        }
    }
}

pub fn read_rowcount(path: &str) -> i64 {
    let mut rdr = csv::Reader::from_path(path).unwrap();
    let mut count = 0;
    for _ in rdr.records() {
        count += 1;
    }
    count
}

pub fn read_all_lines() {
    let folder_path = "/home/juuso/Documents/gradu/parsed_data/orderbook";
    let paths = fs::read_dir(folder_path).unwrap();
    let mut total_count = 0;
    let mut count = 0;
    for path in paths {
        count = read_rowcount(path.unwrap().path().to_str().unwrap());
        println!("{}", count);
        total_count += count;
    }
    println!("Total: {}", total_count);
}

pub fn get_csv_files_sorted(target_path: &PathBuf) -> Vec<PathBuf> {
    // Input is a folder, this iterates through the content and returns all csv files
    let mut path_vec = Vec::new();
    for entry in WalkDir::new(target_path) {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_file() {
            match path.extension() {
                Some(ext) => {
                    if ext == "csv" {
                        path_vec.push(path.to_path_buf());
                    }
                }
                None => continue,
            }
        }
    }
    path_vec.sort();
    path_vec
}

pub fn get_csv_files_key(target_path: &PathBuf, str_key: &str) -> Vec<PathBuf> {
    // Input is a folder, this iterates through the content and returns all csv files containing the key
    let mut path_vec = Vec::new();
    for entry in WalkDir::new(target_path) {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_file() {
            match path.extension() {
                Some(ext) => {
                    if ext == "csv" {
                        let file_name = path.file_name().unwrap().to_str().unwrap();
                        if file_name.contains(str_key) {
                            path_vec.push(path.to_path_buf());
                        }
                    }
                }
                None => continue,
            }
        }
    }
    path_vec.sort();
    path_vec
}

pub fn get_folder_update_files(folder_path: &PathBuf) -> Vec<PathBuf> {
    let csv_files = get_csv_files_sorted(folder_path);
    let mut path_vec = Vec::new();
    for file in csv_files {
        let file_name = file.file_name().unwrap().to_str().unwrap();
        if file_name.contains("_depth_update") {
            path_vec.push(file);
        }
    }
    path_vec.sort();
    path_vec
}

pub fn get_first_snapshot_file(folder_path: &PathBuf) -> Option<PathBuf> {
    let csv_files = get_csv_files_sorted(folder_path);
    let mut path_vec = Vec::new();
    for file in csv_files {
        let file_name = file.file_name().unwrap().to_str().unwrap();
        if file_name.contains("depth_snap") {
            path_vec.push(file);
        }
    }
    path_vec.sort();
    Some(path_vec.remove(0))
}

pub fn get_file_date(file_path: &Path) -> String {
    let file_str = file_path.to_str().unwrap().to_string();
    let file_date = &file_str[file_str.len() - 10..];
    file_date.to_string()
}

pub fn get_first_timestamp(file_path: &PathBuf, timestamp_column: usize) -> i64 {
    let file = File::open(file_path).unwrap();
    let mut rdr = csv::Reader::from_reader(file);
    let first_record = rdr.records().next().unwrap().unwrap();
    let first_timestamp = first_record.get(timestamp_column).unwrap();
    let first_ts = match first_timestamp.parse::<i64>() {
        Ok(ts) => ts,
        Err(_) => panic!("Could not parse timestamp for file {:?}", file_path),
    };
    first_ts
}

pub fn get_last_timestamp(file_path: &PathBuf, timestamp_column: usize) -> i64 {
    let file = File::open(file_path).unwrap();
    let mut rdr = csv::Reader::from_reader(file);
    let mut last_timestamp = 0;
    for record in rdr.records() {
        let record = record.unwrap();
        last_timestamp = record
            .get(timestamp_column)
            .unwrap()
            .parse::<i64>()
            .unwrap();
    }
    last_timestamp
}
