# NeoMatrix
Matrix Template for Multiplication and Transpose with optional multi-thread processing as a solution to the following challenge:

Code Sample - Matrix Multiplication and Transposition

You’ve been tasked with writing a high-performance, portable linear algebra library for a client who has an awful case of not-invented-here syndrome. This means you must only use standard libraries (and miss out on the decades of optimizations in various BLAS implementations). As a start, the client has asked for just transpose and multiplication of MxN matrices.

You should demonstrate the correctness of your solution. Your solution should be well-documented so that it can be used and maintained by the client’s engineering staff. You are free to choose the underlying matrix representation and interface.

C++

Your solution should be implemented in standard C++11 (g++ or clang++ on Linux). To keep things simple, the library may be header-only. You may include a Makefile/CMakeLists.txt, or you may specify the compiler command used to compile your source file:
g++ main.cpp -std=c++11 -lthread
