#![allow(non_snake_case)]

use std::ffi::{c_double, c_float, c_int};

#[cfg(use_apple_blas)]
#[link(name = "cblas")]
extern {
    pub(crate) fn catlas_sset(N: c_int, alpha: c_float, X: *const c_float, incX: c_int);

    pub(crate) fn catlas_dset(N: c_int, alpha: c_double, X: *const c_double, incX: c_int);
    
    pub(crate) fn cblas_sdot(N: c_int, X: *const c_float, incX: c_int, Y: *const c_float, incY: c_int) -> c_float;
    
    pub(crate) fn cblas_ddot(N: c_int, X: *const c_double, incX: c_int, Y: *const c_double, incY: c_int) -> c_double;
}
