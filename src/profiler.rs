use cpu_time::ProcessTime;
use std::env;


pub fn profile_func(func: impl Fn()) {
    let args: Vec<String> = env::args().collect();

    let trials = args[2].parse::<usize>().unwrap();
    let warmup = args[3].parse::<usize>().unwrap();

    for _ in 0..warmup {
        func();
    }

    for _ in 0..trials {
        let start = ProcessTime::now();
        func();
        println!("{}", start.elapsed().as_nanos());
    }
}
