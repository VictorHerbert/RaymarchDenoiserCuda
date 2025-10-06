#ifndef VECTOR_H
#define VECTOR_H

#include <string>
#include <vector>
#include <stdio.h>

#include "cuda_runtime.h"

#include <vector>
#include <cuda_runtime.h>
#include <stdexcept>

template <typename T>
using CPUVector = std::vector<T>;

template <typename T>
struct CudaVector {
private:
    T* data_p;
    size_t size_p;

public:
    CudaVector();
    CudaVector(size_t size);
    CudaVector(T* v, size_t size);
    CudaVector(CPUVector<T>& v);

    T* data();
    const T* data() const;
    size_t size() const;

    void copy(const CPUVector<T>& v);
    void resize(size_t size);
    ~CudaVector();
};

template <typename T>
CudaVector<T>::CudaVector() : size_p(0), data_p(nullptr) {}

template <typename T>
CudaVector<T>::CudaVector(size_t size) : size_p(size) {
    cudaMalloc(&data_p, size_p * sizeof(T));
}

template <typename T>
CudaVector<T>::CudaVector(T* v, size_t size) : size_p(size) {
    cudaMalloc(&data_p, size_p * sizeof(T));
    cudaMemcpy(data_p, v, size_p * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
CudaVector<T>::CudaVector(CPUVector<T>& v) : CudaVector(v.data(), v.size()) {}

template <typename T>
T* CudaVector<T>::data() {return data_p;}

template <typename T>
const T* CudaVector<T>::data() const {return data_p;}

template <typename T>
size_t CudaVector<T>::size() const {return size_p;}

template <typename T>
void CudaVector<T>::copy(const CPUVector<T>& v) {
    if (v.size() != size_p)
        throw std::runtime_error("Size mismatch");
    cudaMemcpy(data_p, v.data(), size_p * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void CudaVector<T>::resize(size_t size) {
    cudaFree(data_p);
    size_p = size;
    cudaMalloc(&data_p, size_p * sizeof(T));
}

template <typename T>
CudaVector<T>::~CudaVector() {
    cudaFree(data_p);
}


#endif