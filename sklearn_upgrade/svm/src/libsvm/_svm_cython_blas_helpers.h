#ifndef _SVM_CYTHON_BLAS_HELPERS_H
#define _SVM_CYTHON_BLAS_HELPERS_H

typedef enum BLAS_Order {
    RowMajor = 0,
    ColMajor = 1
} BLAS_Order;

typedef enum BLAS_Trans {
    NoTrans = 110,  
    Trans   = 116    
} BLAS_Trans;


typedef double (*dot_func)(int, const double*, int, const double*, int);

typedef void (*gemm_func)(
    BLAS_Order, BLAS_Trans, BLAS_Trans,
    int, int, int,
    double,
    const double*, int,
    const double*, int,
    double,
    double*, int 
);

typedef struct BlasFunctions {
    dot_func dot;
    gemm_func gemm;
} BlasFunctions;

#endif