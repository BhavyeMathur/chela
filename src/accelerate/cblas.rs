#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use std::ffi::{c_double, c_float, c_int};

type CBLAS_ORDER = c_int;
type CBLAS_TRANSPOSE = c_int;
type __LAPACK_int = c_int;

pub(crate) const CBLAS_ROW_MAJOR: i32 = 101;
pub(crate) const CBLAS_COL_MAJOR: i32 = 102;

pub(crate) const CBLAS_NO_TRANS: i32 = 111;
pub(crate) const CBLAS_TRANS: i32 = 112;


#[cfg(use_apple_blas)]
#[link(name = "cblas")]
extern "C" {
    // Fill
    pub(crate) fn catlas_sset(N: c_int, alpha: c_float, X: *const c_float, incX: c_int);

    pub(crate) fn catlas_dset(N: c_int, alpha: c_double, X: *const c_double, incX: c_int);

    // Dot Product
    pub(crate) fn cblas_sdot(N: c_int, X: *const c_float, incX: c_int, Y: *const c_float, incY: c_int) -> c_float;

    pub(crate) fn cblas_ddot(N: c_int, X: *const c_double, incX: c_int, Y: *const c_double, incY: c_int) -> c_double;

    // Matrix-Vector Product
    pub(crate) fn cblas_sgemv(ORDER: CBLAS_ORDER,
                              TRANSA: CBLAS_TRANSPOSE,
                              M: __LAPACK_int,
                              N: __LAPACK_int,
                              ALPHA: c_float,
                              A: *const c_float,
                              LDA: __LAPACK_int,
                              X: *const c_float,
                              INCX: __LAPACK_int,
                              BETA: c_float,
                              Y: *mut c_float,
                              INCY: __LAPACK_int);

    pub(crate) fn cblas_dgemv(ORDER: CBLAS_ORDER,
                              TRANSA: CBLAS_TRANSPOSE,
                              M: __LAPACK_int,
                              N: __LAPACK_int,
                              ALPHA: c_double,
                              A: *const c_double,
                              LDA: __LAPACK_int,
                              X: *const c_double,
                              INCX: __LAPACK_int,
                              BETA: c_double, Y:
                              *mut c_double,
                              INCY: __LAPACK_int);

    // Matrix-Matrix Product
    pub(crate) fn cblas_sgemm(ORDER: CBLAS_ORDER,
                              TRANSA: CBLAS_TRANSPOSE,
                              TRANSB: CBLAS_TRANSPOSE,
                              M: __LAPACK_int,
                              N: __LAPACK_int,
                              K: __LAPACK_int,
                              ALPHA: c_float,
                              A: *mut c_float,
                              LDA: __LAPACK_int,
                              B: *mut c_float,
                              LDB: __LAPACK_int,
                              BETA: c_float,
                              C: *mut c_float,
                              LDC: __LAPACK_int,
    );

    pub(crate) fn cblas_dgemm(ORDER: CBLAS_ORDER,
                              TRANSA: CBLAS_TRANSPOSE,
                              TRANSB: CBLAS_TRANSPOSE,
                              M: __LAPACK_int,
                              N: __LAPACK_int,
                              K: __LAPACK_int,
                              ALPHA: c_double,
                              A: *mut c_double,
                              LDA: __LAPACK_int,
                              B: *mut c_double,
                              LDB: __LAPACK_int,
                              BETA: c_double,
                              C: *mut c_double,
                              LDC: __LAPACK_int,
    );

    // not available before macOS 15.0
    //
    // pub(crate) fn BLASSetThreading(threading: c_int) -> c_int;
    // 
    // pub(crate) fn BLASGetThreading() -> c_int;
}
