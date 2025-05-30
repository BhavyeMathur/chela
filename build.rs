fn main() {
    #[cfg(apple_accelerate)]
    println!("cargo:rustc-link-lib=framework=Accelerate");

    #[cfg(openblas)]
    {
        use std::env;
        env::set_var("PKG_CONFIG_PATH", "/opt/homebrew/opt/openblas/lib/pkgconfig");
        pkg_config::probe_library("openblas").expect("OpenBLAS not found");
    }
}
