// ############################################
// Author: 	Amber Rogowicz
// File	:	main.cpp   test main for running NeoMatrix 
// Date:	July 2018



#include <iostream>
#include <string>
#include "neomatrix.hpp"

using namespace std;

using namespace neolib;

// if matrix_test receives ANY arguments, it tests for help and 
// or attempts to parse the first command line parameter into an
// integer for use as the number of threads, otherwise runs with a
// default of 4
// matrix_test runs single threaded without arguments

const int MAX_THREADS = 4;   // default number of threads 

static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name << "  | -h | # | [ANY GARBAGE]\n"
              << "\t-h,--help\t\tShow this help message\n"
              << "\t WITH NO PARAMETERS \t\t means run single threaded \n"
              << "\t # \t\t\t = run with the Maximum NUMBER_OF_THREADS > 1\n"
              << "\t [ANY GARBAGE]\t\t  = run with the default NUMBER_OF_THREADS = 4\n"
              << std::endl;
}
int main(int argc, char *argv[]) 
{
    if (argc > 1) 
	{
    	int num_threads = MAX_THREADS;
    	std::string arg  = argv[1];
        if ((arg == "-h") || (arg == "--help")) 
		{
            // user  is asking for help to run the program
            show_usage(argv[0]);
            return(1);
    	}
		try {
            // string -> integer
            num_threads = std::stoi(arg);
		} catch(...){
        		// Number of threads in command line arguments is not parsable
    			num_threads = MAX_THREADS;
            	//show_usage(argv[0]);
            	//return(1);
    	} 
        neolib::initMatLib(num_threads);
	  	std::cout << "Running multi-threaded with "<< num_threads << " threads\n\n";
    } else 
	  std::cout << "Running single threaded...\n\n";


    Matrix simpleA={ { 1.0, 1.0, 1.0 } , 
					 { 1.0, 1.0, 1.0 } , 
					 { 1.0, 1.0, 1.0 } , 
					 { 1.0, 1.0, 1.0 } , 
					 { 1.0, 1.0, 1.0 } } ;
    // Matrix simpleA(5, 3);
    // simpleA = 1.0;
    Matrix simpleB(3, 2);
	Matrix A(108, 2041);
	Matrix B(2041, 1041), C(1041, 2043), D(2043, 41), E(41, 3601); 
	Matrix F(3601, 5701), G(5701, 206), H(206, 1041), I(1041, 6);
	
    simpleB = 2.0;
	A = 1.00;
	B = 0.05;
	C = 1.00;
	D = -0.25;
	E = 0.005;
	F = 0.0025;
	G = -0.20;
	H = 0.00075;
	I = 1.00;
	

	// print the intended operation and results
	//std::cout << A << "   * \n" << B << "   * \n" << C  << "   * \n" << D  << "   * \n" << E  << "   * \n"
    //				<< F  << "   * \n"<< G  << "   * \n" << H  << "   * \n" << I  << "   = \n" ;
	std::cout << "A * B * C * D * E * F * G * H * I = \n";
	Matrix aiResult = A * B * C * D * E * F * G * H * I;
	std::cout << aiResult << "\n\n";
	std::cout << "~  aiResult =\n\n " ;
	
	std::cout << aiResult.transpose() << "\n\n" ;

	std::cout << simpleA << "\n\t*\n\n" << simpleB << "\n\t= \n" ;
	Matrix simple = simpleA * simpleB ;
	std::cout << simple;

	H.initialize_random();
	I.initialize_random();
	//std::cout << H << "\n\t*\n" << I << "\n\t=\n" ;
	Matrix abResult = H * I ;
	std::cout << abResult;
	
	return 0;
}
