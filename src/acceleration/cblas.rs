#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use std::ffi::c_int;

type CBLAS_ORDER = c_int;
type CBLAS_TRANSPOSE = c_int;
type __LAPACK_int = c_int;

pub(crate) const CBLAS_ROW_MAJOR: i32 = 101;
pub(crate) const CBLAS_COL_MAJOR: i32 = 102;

pub(crate) const CBLAS_NO_TRANS: i32 = 111;
pub(crate) const CBLAS_TRANS: i32 = 112;


#[cfg(blas)]
#[link(name = "cblas")]
extern "C" {
    // Fill
    pub(crate) fn catlas_sset(N: c_int, alpha: f32, X: *const f32, incX: c_int);

    pub(crate) fn catlas_dset(N: c_int, alpha: f64, X: *const f64, incX: c_int);

    // Dot Product
    pub(crate) fn cblas_sdot(N: c_int, X: *const f32, incX: c_int, Y: *const f32, incY: c_int) -> f32;

    pub(crate) fn cblas_ddot(N: c_int, X: *const f64, incX: c_int, Y: *const f64, incY: c_int) -> f64;

    // Matrix-Vector Product
    pub(crate) fn cblas_sgemv(ORDER: CBLAS_ORDER,
                              TRANSA: CBLAS_TRANSPOSE,
                              M: __LAPACK_int,
                              N: __LAPACK_int,
                              ALPHA: f32,
                              A: *const f32,
                              LDA: __LAPACK_int,
                              X: *const f32,
                              INCX: __LAPACK_int,
                              BETA: f32,
                              Y: *mut f32,
                              INCY: __LAPACK_int);

    pub(crate) fn cblas_dgemv(ORDER: CBLAS_ORDER,
                              TRANSA: CBLAS_TRANSPOSE,
                              M: __LAPACK_int,
                              N: __LAPACK_int,
                              ALPHA: f64,
                              A: *const f64,
                              LDA: __LAPACK_int,
                              X: *const f64,
                              INCX: __LAPACK_int,
                              BETA: f64, Y:
                              *mut f64,
                              INCY: __LAPACK_int);

    // Matrix-Matrix Product
    pub(crate) fn cblas_sgemm(ORDER: CBLAS_ORDER,
                              TRANSA: CBLAS_TRANSPOSE,
                              TRANSB: CBLAS_TRANSPOSE,
                              M: __LAPACK_int,
                              N: __LAPACK_int,
                              K: __LAPACK_int,
                              ALPHA: f32,
                              A: *mut f32,
                              LDA: __LAPACK_int,
                              B: *mut f32,
                              LDB: __LAPACK_int,
                              BETA: f32,
                              C: *mut f32,
                              LDC: __LAPACK_int,
    );

    pub(crate) fn cblas_dgemm(ORDER: CBLAS_ORDER,
                              TRANSA: CBLAS_TRANSPOSE,
                              TRANSB: CBLAS_TRANSPOSE,
                              M: __LAPACK_int,
                              N: __LAPACK_int,
                              K: __LAPACK_int,
                              ALPHA: f64,
                              A: *mut f64,
                              LDA: __LAPACK_int,
                              B: *mut f64,
                              LDB: __LAPACK_int,
                              BETA: f64,
                              C: *mut f64,
                              LDC: __LAPACK_int,
    );

    // not available before macOS 15.0
    //
    // pub(crate) fn BLASSetThreading(threading: c_int) -> c_int;
    // 
    // pub(crate) fn BLASGetThreading() -> c_int;
}
