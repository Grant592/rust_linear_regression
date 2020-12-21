use std::fs;
use std::fs::File;
use std::io::Read;
use std::process;
use std::ops::Div;
use std::error::Error;

// Simple vector dot product fundtion
pub fn dot(x: &Vec<f64>, y: &Vec<f64>) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(x,y)| x * y )
        .sum()
}

// Struct to hold the data. A vec of vecs for observations and a vec for y values
#[derive(Debug)]
pub struct Data {
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    n: usize,
}


// new functions takes a two columns csv file and returns Data struct
// Needs cleaned up to be able to use other filetypes
// Will use csv library eventually
impl Data {
    pub fn new(file: String) -> Data {
        let mut x_data: Vec<Vec<f64>> = Vec::new();
        let mut y_data: Vec<f64> = Vec::new();

       // let mut salary_data = file
       //     .split("\r\n")
       //     .skip(1)
       //     .map(|line| line.split(",").collect::<Vec<&str>>())
       //     .into_iter();


        file.split("\r\n")
            .skip(1)
            .map(|line| line.split(",").collect::<Vec<&str>>())
            .map(|line| {
                for i in 0..line.len() {
                    let x = match line[i].trim().parse::<f64>() {
                        Ok(x) => x,
                        Err(e) => {
                            println!("Error parsing x-data: {}", e);
                            break
                        }
                    };
                    let y = match line[1].trim().parse::<f64>() {
                        Ok(y) =>  y,
                        Err(e) => {
                            println!("Error parsing y-data: {}", e);
                            break
                        }
                    };
                    x_data.push(vec![1.0, x]);
                    y_data.push(y);
                }
            });

       // for line in salary_data {
       //     println!("{:?}", line);

       //     let x = match line[0].trim().parse::<f64>() {
       //         Ok(x) =>  x,
       //         Err(e) => {
       //             println!("Error parsing x-data: {}", e);
       //             break
       //         }
       //     };
       //     let y = match line[1].trim().parse::<f64>() {
       //         Ok(y) =>  y,
       //         Err(e) => {
       //             println!("Error parsing y-data: {}", e);
       //             break
       //         }
       //     };

       //     x_data.push(vec![1.0, x]);
       //     y_data.push(y);
       // }

        let n = x_data.len();
        Data { x: x_data, y: y_data, n }
    }
}

// Linear Regression Struct
pub struct LinReg {
    dim: u8,
    coeffecients: Vec<f64>,
    n_iterations: u32,
    alpha: f64
}

impl LinReg {
    // Set number of dimenision by passing as arg
    pub fn new(n: u8) -> LinReg {
        LinReg {
            dim: n,
            coeffecients: vec![1.0; n as usize + 1],
            n_iterations: 1000,
            alpha: 0.01,
        }
    }

    pub fn predict(&mut self, data: &Data) -> Result<Vec<f64>, &'static str> {
        let mut predictions = Vec::new();
        for i in 0..data.n {
            let y_hat = dot(&self.coeffecients, &data.x[i]);
            predictions.push(y_hat);
        }
        Ok(predictions)
    }


    pub fn calc_loss(&mut self, data: &Data, y_hat: &Vec<f64>) -> Result<f64, &'static str> {
        let mut ms_error: f64 = data.y.iter()
            .zip(y_hat.into_iter())
            .map(|(y, y_hat)| {
                (y - y_hat).powi(2)
            }).sum();
        println!("Sum SE = {}", ms_error);
        ms_error /= (2 * data.n) as f64;
        println!("Mean SE = {}", ms_error);

        Ok(ms_error)
    }


    pub fn update_weights(&mut self, data: &Data, y_hat: &Vec<f64>) -> Result<(), &'static str> {
        let mut grad = vec![0.0, 0.0];
        for i in 0..(self.dim + 1) as usize {
            grad[i] = data.x.iter()
                .zip(data.y.iter().zip(y_hat.iter()))
                .map(|(x, (y, y_hat))| -1.0 * x[i] * (y - y_hat) )
                .sum();

            grad[i] /= data.n as f64
        }

        for i in 0..(self.dim + 1) as usize {
            self.coeffecients[i] -= self.alpha * grad[i];
        }
        Ok(())
    }
}

pub fn run(data: &Data) -> Result<(), &'static str> {

    let mut lr = LinReg::new(1);

    for _ in 0..lr.n_iterations {
        let pred = lr.predict(&data).expect("Error predicting y-values");

        let mse = lr.calc_loss(&data, &pred).expect("Error calculating loss");

        lr.update_weights(&data, &pred).expect("Error updating weights");
    }

    let final_pred = lr.predict(&data).expect("Final prediction error");
    println!("Final predictions ={:?}", final_pred);
    println!("Intercept = {}", lr.coeffecients[0]);
    println!("Weight = {}", lr.coeffecients[1]);

    Ok(())
}