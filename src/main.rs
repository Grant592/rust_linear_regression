use std::fs;
use std::process;
use lin_reg::*;

fn main() {
    let file = fs::read_to_string("salary_data.csv").unwrap_or_else(|err| {
        println!("Error reading file. Exiting program");
        process::exit(1);
    });

    let data = Data::new(file);
    println!("{:?}", data);
    if let Err(e) = run(&data) {
        println!("Linear Regression error: {}", e)
    };
}
