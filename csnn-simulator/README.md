# CSNN-simulator

This is a new version of the CSNN simulator that contains 2D and 3D convolution, along with two-stream methods for video analysis


## Description

Simulator of Convolutional Spiking Neural Network

Provide implementation of experiments described in:
* __Unsupervised Visual Feature Learning with Spike-timing-dependent Plasticity: How Far are we from Traditional Feature Learning Approaches?__, P Falez, P Tirilly, IM Bilasco, P Devienne, P Boulet, Pattern Recognition.
* __Multi-layered Spiking Neural Network with Target Timestamp Threshold Adaptation and STDP__, P Falez, P Tirilly, IM Bilasco, P Devienne, P Boulet, IJCNN 2019.

## Requirement

* C++ compiler (version >= 17)
* Cmake (version >= 3.1)
* Qt4 (version >= 4.4.3)
* BLAS
* LAPACKE
* OpenCV (version >= 4.2.0)

## Installation

```bash
mkdir csnn-simulator-build
cd csnn-simulator-build
#cmake ../csnn-simulator -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake ../csnn-simulator -DCMAKE_CXX_COMPILER=/usr/local/gcc-9.5.0/bin/g++  -DCMAKE_C_COMPILER=/usr/local/gcc-9.5.0/bin/gcc
make
```

## Usage

Run MNIST Example:

    export INPUT_PATH=/path/to/mnist/
    ./Mnist