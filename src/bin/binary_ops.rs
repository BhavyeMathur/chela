use chela::*;
use std::env;

use cpu_time::ProcessTime;

const N: usize = 4096;
type T = f32;


fn binary_ops0() -> u128 {
    let tensor1 = NdArray::<f32>::rand([N]).astype::<T>();
    let tensor2 = NdArray::<f32>::rand([N]).astype::<T>();

    let start = ProcessTime::now();
    _ = tensor1 - tensor2;
    start.elapsed().as_nanos()
}

fn binary_ops1() -> u128 {
    let tensor1 = NdArray::<f32>::rand([N]).astype::<T>();
    let tensor2 = NdArray::<f32>::rand([1]).astype::<T>();

    let start = ProcessTime::now();
    _ = tensor1 - tensor2;
    start.elapsed().as_nanos()
}

fn binary_ops2() -> u128 {
    let tensor1 = NdArray::<f32>::rand([N, 3]).astype::<T>();
    let tensor1 = tensor1.slice_along(Axis(-1), 0);

    let tensor2 = NdArray::<f32>::rand([N]).astype::<T>();

    let start = ProcessTime::now();
    _ = tensor1 - tensor2;
    start.elapsed().as_nanos()
}


fn binary_ops3() -> u128 {
    let tensor1 = NdArray::<f32>::rand([N, 3]).astype::<T>();
    let tensor1 = tensor1.slice_along(Axis(-1), 0);

    let tensor2 = NdArray::<f32>::rand([1]).astype::<T>();

    let start = ProcessTime::now();
    _ = tensor1 - tensor2;
    start.elapsed().as_nanos()
}

fn binary_ops4() -> u128 {
    let tensor1 = NdArray::<f32>::rand([N, 3]).astype::<T>();
    let tensor1 = tensor1.slice_along(Axis(-1), 0..2);

    let tensor2 = NdArray::<f32>::rand([N, 2]).astype::<T>();

    let start = ProcessTime::now();
    _ = tensor1 - tensor2;
    start.elapsed().as_nanos()
}

fn binary_ops5() -> u128 {
    let tensor1 = NdArray::<f32>::rand([N, 3]).astype::<T>();
    let tensor1 = tensor1.slice_along(Axis(-1), 0..2);

    let tensor2 = NdArray::<f32>::rand([1]).astype::<T>();

    let start = ProcessTime::now();
    _ = tensor1 - tensor2;
    start.elapsed().as_nanos()
}

fn binary_ops6() -> u128 {
    let tensor1 = NdArray::<f32>::rand([N, 3]).astype::<T>();
    let tensor1 = tensor1.slice_along(Axis(-1), 0..2);

    let tensor2 = NdArray::<f32>::rand([N, 2, 2]).astype::<T>();
    let tensor2 = tensor2.slice_along(Axis(-1), 0);

    let start = ProcessTime::now();
    _ = tensor1 - tensor2;
    start.elapsed().as_nanos()
}

fn binary_ops7() -> u128 {
    let tensor1 = NdArray::<f32>::rand([N, 3]).astype::<T>();
    let tensor1 = tensor1.slice_along(Axis(-1), 0..2);

    let tensor2 = NdArray::<f32>::rand([N, 3]).astype::<T>();
    let tensor2 = tensor2.slice_along(Axis(-1), 0..2);

    let start = ProcessTime::now();
    _ = tensor1 - tensor2;
    start.elapsed().as_nanos()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let id = args[1].parse::<usize>().unwrap();

    let time =
        if id == 0 { binary_ops0() }
        else if id == 1 { binary_ops1() }
        else if id == 2 { binary_ops2() }
        else if id == 3 { binary_ops3() }
        else if id == 4 { binary_ops4() }
        else if id == 5 { binary_ops5() }
        else if id == 6 { binary_ops6() }
        else if id == 7 { binary_ops7() }

        else { panic!("invalid ID") };

    println!("{}", time);
}
