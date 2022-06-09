
#include <iostream>
#include <chrono>
#include "neomatrix.hpp"



using namespace matlib;

// if main receives ANY arguments, its assumed the user wants to 
// run matrix multiplication with threading 

// const int MAX_THREADS = 4;   // hardcoding for simplicity

int main(int argc, char *argv[]) 
{
    if (argc > 1) 
	{
	  size_t MAX_THREADS = std::thread::hardware_concurrency();

	  size_t user_num_threads = atoi(argv[1]);
	  std::cout << "The number of available processing cores: " << MAX_THREADS << std::endl;
      matlib::initMatLib(user_num_threads<=MAX_THREADS?user_num_threads:MAX_THREADS);
	  std::cout << "Running multi-threaded...\n\n";
    } else 
	  std::cout << "Running single threaded...\n\n";


    Matrix simpleA(8, 3);
    Matrix simpleB(3, 2);
	Matrix A(108, 2041);
	Matrix B(2041, 1041), C(1041, 3), D(3, 41), E(41, 3601); 
	Matrix F(3601, 5701), G(5701, 6), H(6, 1041), I(1041, 5);
	
    simpleA = 1.0;
    simpleB = 2.0;
	A = 1.00;
	B = 0.05;
	C = 1.00;
	D = -0.25;
	E = 0.005;
	F = 0.0025;
	G = -0.50;
	H = 2.00;
	I = 1.00;
	


	std::cout << "Multiplying simple matricies: \n";
    std::cout << "simpleA: \n" << simpleA << std::endl;
    std::cout << "simpleB: \n" << simpleB << std::endl;
	Matrix simple = simpleA * simpleB ;
	std::cout << "dot product result: \n" << simple << std::endl;


	std::cout << "Transpose of Dot product of matricies: [" << H.getNumRows() << "][" << H.getNumCols() << "] * ["<< I.getNumRows() << "][ " << I.getNumCols() << "]\nWhere A = 2.0, B = 1.0\n\n" ;   

	Matrix abResult = H * I ;
	std::cout << abResult.transpose();
	
	std::cout << "Dot product of matricies: [" << A.getNumRows() << "][" << A.getNumCols() << "] * ["<< B.getNumRows() << "][ " << B.getNumCols() << "] * [" << C.getNumRows() << "][" << C.getNumCols() << "] \n" ;   
	std::cout << "Where A = 1.0, B = 0.5, C = 1.0 \n\n\n";
	std::cout << A * B * C << "\n\n\n";
	A.initialize_random();
	B.initialize_random();
	C.initialize_random();
	D.initialize_random();
	E.initialize_random();
	F.initialize_random();
	G.initialize_random();
	H.initialize_random();
	I.initialize_random();
	std::cout << "Matricies: A * B * C * D * E * F * G * H * I initialized randomly [-1.0 -> 1.0]\n\n";
    auto start_time = std::chrono::high_resolution_clock::now();
	Matrix aiResult = A * B * C * D * E * F * G * H * I;
	auto processing_time = std::chrono::high_resolution_clock::now() - start_time;
	std::cout << aiResult << std::endl;
	// std::cout << aiResult.transpose();
	std::cout << "\nProcessing time: " << processing_time.count() << std::endl;

	return 0;
}
