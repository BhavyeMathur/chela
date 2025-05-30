use chela::*;
use std::env;

use cpu_time::ProcessTime;


const N: usize = 10000000;

type T = f32;


fn reduce0() -> u128 {
    let tensor = NdArray::<f32>::rand([N]).astype::<T>();

    let start = ProcessTime::now();
    _ = tensor.sum();
    start.elapsed().as_nanos()
}

fn reduce1() -> u128 {
    let tensor = NdArray::<f32>::rand([N]).astype::<T>();

    let start = ProcessTime::now();
    _ = tensor.product();
    start.elapsed().as_nanos()
}

fn reduce2() -> u128 {
    let tensor = NdArray::<f32>::rand([N]).astype::<T>();

    let start = ProcessTime::now();
    _ = tensor.min();
    start.elapsed().as_nanos()
}

fn reduce3() -> u128 {
    let tensor = NdArray::<f32>::rand([N]).astype::<T>();

    let start = ProcessTime::now();
    _ = tensor.max();
    start.elapsed().as_nanos()
}

fn reduce10() -> u128 {
    let tensor = NdArray::<f32>::rand([N, 2]).astype::<T>();
    let tensor = tensor.slice_along(Axis(1), 0);

    let start = ProcessTime::now();
    _ = tensor.sum();
    start.elapsed().as_nanos()
}

fn reduce11() -> u128 {
    let tensor = NdArray::<f32>::rand([N, 2]).astype::<T>();
    let tensor = tensor.slice_along(Axis(1), 0);

    let start = ProcessTime::now();
    _ = tensor.product();
    start.elapsed().as_nanos()
}

fn reduce12() -> u128 {
    let tensor = NdArray::<f32>::rand([N, 2]).astype::<T>();
    let tensor = tensor.slice_along(Axis(1), 0);

    let start = ProcessTime::now();
    _ = tensor.min();
    start.elapsed().as_nanos()
}

fn reduce13() -> u128 {
    let tensor = NdArray::<f32>::rand([N, 2]).astype::<T>();
    let tensor = tensor.slice_along(Axis(1), 0);

    let start = ProcessTime::now();
    _ = tensor.max();
    start.elapsed().as_nanos()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let id = args[1].parse::<usize>().unwrap();

    let time =
        if id == 0 { reduce0() }
        else if id == 1 { reduce1() }
        else if id == 2 { reduce2() }
        else if id == 3 { reduce3() }

        else if id == 10 { reduce10() }
        else if id == 11 { reduce11() }
        else if id == 12 { reduce12() }
        else if id == 13 { reduce13() }

        else { panic!("invalid ID") };

    println!("{}", time);
}
