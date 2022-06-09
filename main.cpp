// ####################################################
// Author: 	Amber Rogowicz
// File	:	main.cpp   test main for using neomatrix 
// Date:	Oct 2019

#include <iostream>
#include <chrono>
#include "neomatrix.hpp"


using namespace neolib;

static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name << "  | -h | # | [ANY GARBAGE]\n"
              << "\t-h,--help\t\tShow this help message\n"
              << "\t WITH NO PARAMETERS \t\t means run default threaded \n"
              << "\t # \t\t\t = run with the Maximum NUMBER_OF_THREADS > 1\n"
              << "\t [ANY GARBAGE]\t\t  = run with the default NUMBER_OF_THREADS = 4\n"
              << std::endl;
}

// if main receives ANY arguments, its assumed the user wants to 
// run matrix multiplication with the given amount of threads.
// NOTE: not exceeding the number of processing cores

int main(int argc, char *argv[]) 
{
    if (argc > 1) 
	{
    	std::string arg  = argv[1];
        if ((arg == "-h") || (arg == "--help")) 
		{
            // user  is asking for help to run the program
            show_usage(argv[0]);
            return(1);
    	}
		try {
	  		// How many processing cores are available?
            size_t user_num_threads = std::stoi(arg);
	  		size_t MAX_THREADS = std::thread::hardware_concurrency();
	  		std::cout << "The number of available processing cores: " << MAX_THREADS << std::endl;
	  		// Choose the number of threads based on user choice and core presence
      		neolib::initMatLib(user_num_threads<=MAX_THREADS?user_num_threads:MAX_THREADS);
	  		std::cout << "Running multi-threaded...\n\n";
		} catch(...){
        	// Number of threads in command line arguments is not parsable
            show_usage(argv[0]);
            return(1);
    	} 
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
	// Time the processing of the last compound matrix multiply statement
	auto processing_time = std::chrono::high_resolution_clock::now() - start_time;
	std::cout << aiResult << std::endl;
	// std::cout << aiResult.transpose();
	std::cout << "\nProcessing time: " << processing_time.count() << std::endl;

	return 0;
}
