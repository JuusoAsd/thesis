use super::parse_util::Side;
use csv::{Reader, ReaderBuilder, StringRecord, StringRecordsIntoIter};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::vec::IntoIter;

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
// impl Iterator for FileHandler {
//     type Item = StringRecord;
//     fn next(&mut self) -> Option<StringRecord> {
//         match self.file_iter.next().unwrap() {
//             Ok(record) => Some(record),
//             Err(e) => {
//                 println!("Error with next of FileHandler: {}", e);
//                 let next_path = match self.path_iter.next() {
//                     Some(p) => p,
//                     None => return None,
//                 };
//                 let file_reader = Reader::from_path(next_path).unwrap();
//                 self.file_iter = file_reader.into_records();
//                 self.next()
//             }
//         }
//     }
// }

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

pub fn get_folder_update_files(folder_path: &PathBuf) -> Vec<PathBuf> {
    let mut path_vec = Vec::new();
    let paths = fs::read_dir(folder_path).unwrap();
    for path in paths {
        let sub_path = fs::read_dir(path.unwrap().path()).unwrap();
        for sub_path in sub_path {
            let file_path = sub_path.as_ref().unwrap().path();
            if sub_path
                .unwrap()
                .path()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .contains("depth_update.csv")
            {
                path_vec.push(file_path);
            }
        }
    }
    path_vec.sort();
    path_vec
}

pub fn get_folder_files(folder_path: &PathBuf) -> Vec<PathBuf> {
    let mut path_vec = Vec::new();
    let files = match fs::read_dir(folder_path) {
        Ok(files) => files,
        Err(_) => {
            println!("Error reading folder: {}", folder_path.to_str().unwrap());
            return path_vec;
        }
    };
    for file in files {
        let file_path = file.as_ref().unwrap().path();
        path_vec.push(file_path);
    }
    path_vec.sort();
    path_vec
}

pub fn get_first_snapshot_file(folder_path: &PathBuf) -> Option<PathBuf> {
    let sub_folders = fs::read_dir(folder_path).unwrap();
    for folder in sub_folders {
        let sub_path = fs::read_dir(folder.unwrap().path()).unwrap();
        for sub_path in sub_path {
            let file_path = sub_path.as_ref().unwrap().path();
            if sub_path
                .unwrap()
                .path()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .contains("depth_snap.csv")
            {
                return Some(file_path);
            }
        }
    }
    None
}

pub fn get_file_date(file_path: &Path) -> String {
    let file_str = file_path.to_str().unwrap().to_string();
    let file_date = &file_str[file_str.len() - 10..];
    file_date.to_string()
}
