use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use csv::Writer;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};

// use super::orderbook_util::{Orderbook, OrderbookLevel, OrderbookLevelPartial};

// pub fn find_first_timestamp(path: &str) -> i64 {
//     let mut rdr = csv::Reader::from_path(path).unwrap();
//     for result in rdr.deserialize() {
//         let record: UpdateRecord = match result {
//             Ok(r) => r,
//             Err(e) => {
//                 println!("Error reading update: {}", e);
//                 continue;
//             }
//         };
//         return record.timestamp;
//     }
//     0
// }

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
