#![allow(non_snake_case)]

use std::ffi::{c_double, c_float, c_int};

#[cfg(target_vendor = "apple")]
#[link(name = "cblas")]
extern {
    pub(crate) fn catlas_sset(N: c_int, alpha: c_float, X: *const c_float, incX: c_int);

    pub(crate) fn catlas_dset(N: c_int, alpha: c_double, X: *const c_double, incX: c_int);
}
