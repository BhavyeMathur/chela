fn main() {
    #[cfg(use_apple_accelerate)]
    println!("cargo:rustc-link-lib=framework=Accelerate");
}
