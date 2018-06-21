
#include <iostream>
#include "neomatrix.hpp"



using namespace matlib;

// if main receives ANY arguments, its assumed the user wants to 
// run matrix multiplication with threading 

const int MAX_THREADS = 4;   // hardcoding for simplicity

int main(int argc, char *argv[]) 
{
    if (argc > 1) 
	{
      matlib::initMatLib(MAX_THREADS);
	  std::cout << "Running multi-threaded...\n\n";
    } else 
	  std::cout << "Running single threaded...\n\n";


    Matrix simpleA(5, 3);
    Matrix simpleB(3, 2);
	Matrix A(20, 2041);
	Matrix B(2041, 1041), C(1041, 2043), D(2043, 41), E(41, 3601); 
	Matrix F(3601, 5701), G(5701, 206), H(206, 1041), I(1041, 6);
	
    simpleA = 1;
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
	

	Matrix aiResult = A * B * C * D * E * F * G * H * I;
	std::cout << aiResult.transpose();


	Matrix simple = simpleA * simpleB ;
	std::cout << simple;

	H.initialize_random();
	I.initialize_random();
	Matrix abResult = H * I ;
	std::cout << abResult.transpose();
	
	return 0;
}
