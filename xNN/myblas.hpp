/*
* myblas.hpp
*
*  Created on: 2016Äê6ÔÂ5ÈÕ
*      Author: Administrator
*/

#ifndef MYBLAS_HPP_
#define MYBLAS_HPP_

#include "include\\openblas\\cblas.h"

// defs

template<typename DType>
inline void vector_hadamard_product(DType * output, const DType * vector, int l);

template<typename DType>
inline void vector_hadamard_product(DType * output, const DType * vector1, const DType * vector2, int l);

template<typename DType>
inline void vector_copy_vector(DType * output, const DType * vector, int l);

template<typename DType>
inline void vector_add_vector(DType * output, const DType * vector, int l);

template<typename DType>
inline void alpha_vector_add_vector(DType * output, const DType * vector, DType alpha, int l);

template<typename DType>
inline void alpha_vector_add_beta_vector(DType * output, const DType * vector, DType alpha, DType beta, int l);

template<typename DType>
inline void vector_mul_vector_add_matrix(DType * output, const DType * vector1, const DType * vector2, int l1, int l2);

template<typename DType>
inline void matrix_mul_vector_add_output(DType * output, const DType * matrix, const DType * vector, int h, int w);

template<typename DType>
inline void transpose_matrix_mul_vector_add_output(DType * output, const DType * matrix, const DType * vector, int h, int w);

template<typename DType>
inline void vector_mul_matrix_add_output(DType * output, const DType * vector, const DType * matrix, int h, int w);

template<typename DType>
inline void transpose_vector_mul_matrix_add_output(DType * output, const DType * vector, const DType * matrix, int h, int w);

// float
template<>
inline void vector_hadamard_product<float>(float * output, const float * vector, int l) {
	for (int i = 0; i < l; ++i) output[i] *= vector[i];
}

template<>
inline void vector_hadamard_product<float>(float * output, const float * vector1, const float * vector2, int l) {
	for (int i = 0; i < l; ++i) output[i] = vector1[i] * vector2[i];
}

template<>
inline void vector_copy_vector<float>(float * output, const float * vector, int l) {
	cblas_scopy(l, vector, 1, output, 1);
}

template<>
inline void vector_add_vector<float>(float * output, const float * vector, int l) {
	cblas_saxpy(l, 1, vector, 1, output, 1);
}

template<>
inline void alpha_vector_add_vector<float>(float * output, const float * vector, float alpha, int l) {
	cblas_saxpy(l, alpha, vector, 1, output, 1);
}

template<>
inline void alpha_vector_add_beta_vector<float>(float * output, const float * vector, float alpha, float beta, int l) {
	cblas_saxpby(l, alpha, vector, 1, beta, output, 1);
}

template<>
inline void vector_mul_vector_add_matrix(float * output, const float * vector1, const float * vector2, int l1, int l2) {
	cblas_sger(CblasRowMajor, l1, l2, 1, vector1, 1, vector2, 1, output, l2);
}

template<>
inline void matrix_mul_vector_add_output<float>(float * output, const float * matrix, const float * vector, int h, int w) {
	cblas_sgemv(CblasRowMajor, CblasNoTrans, h, w, 1, matrix, w, vector, 1, 1, output, 1);
}

template<>
inline void transpose_matrix_mul_vector_add_output<float>(float * output, const float * matrix, const float * vector, int h, int w) {
	cblas_sgemv(CblasRowMajor, CblasTrans, h, w, 1, matrix, w, vector, 1, 1, output, 1);
}

template<>
inline void vector_mul_matrix_add_output<float>(float * output, const float * vector, const float * matrix, int h, int w) {
	cblas_sgemv(CblasRowMajor, CblasTrans, h, w, 1, matrix, w, vector, 1, 1, output, 1);
}

template<>
inline void transpose_vector_mul_matrix_add_output<float>(float * output, const float * vector, const float * matrix, int h, int w) {
	cblas_sgemv(CblasRowMajor, CblasNoTrans, h, w, 1, matrix, w, vector, 1, 1, output, 1);
}

// double
template<>
inline void vector_hadamard_product<double>(double * output, const double * vector, int l) {
	for (int i = 0; i < l; ++i) output[i] *= vector[i];
}

template<>
inline void vector_hadamard_product<double>(double * output, const double * vector1, const double * vector2, int l) {
	for (int i = 0; i < l; ++i) output[i] = vector1[i] * vector2[i];
}

template<>
inline void vector_copy_vector<double>(double * output, const double * vector, int l) {
	cblas_dcopy(l, vector, 1, output, 1);
}

template<>
inline void vector_add_vector<double>(double * output, const double * vector, int l) {
	cblas_daxpy(l, 1, vector, 1, output, 1);
}

template<>
inline void alpha_vector_add_vector<double>(double * output, const double * vector, double alpha, int l) {
	cblas_daxpy(l, alpha, vector, 1, output, 1);
}

template<>
inline void alpha_vector_add_beta_vector<double>(double * output, const double * vector, double alpha, double beta, int l) {
	cblas_daxpby(l, alpha, vector, 1, beta, output, 1);
}

template<>
inline void vector_mul_vector_add_matrix(double * output, const double * vector1, const double * vector2, int l1, int l2) {
	cblas_dger(CblasRowMajor, l1, l2, 1, vector1, 1, vector2, 1, output, l2);
}

template<>
inline void matrix_mul_vector_add_output<double>(double * output, const double * matrix, const double * vector, int h, int w) {
	cblas_dgemv(CblasRowMajor, CblasNoTrans, h, w, 1, matrix, w, vector, 1, 1, output, 1);
}

template<>
inline void transpose_matrix_mul_vector_add_output<double>(double * output, const double * matrix, const double * vector, int h, int w) {
	cblas_dgemv(CblasRowMajor, CblasTrans, h, w, 1, matrix, w, vector, 1, 1, output, 1);
}

template<>
inline void vector_mul_matrix_add_output<double>(double * output, const double * vector, const double * matrix, int h, int w) {
	cblas_dgemv(CblasRowMajor, CblasTrans, h, w, 1, matrix, w, vector, 1, 1, output, 1);
}

template<>
inline void transpose_vector_mul_matrix_add_output<double>(double * output, const double * vector, const double * matrix, int h, int w) {
	cblas_dgemv(CblasRowMajor, CblasNoTrans, h, w, 1, matrix, w, vector, 1, 1, output, 1);
}

#endif