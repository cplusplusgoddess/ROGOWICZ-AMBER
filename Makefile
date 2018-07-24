
############################################
#  Author: 	Amber Rogowicz
#  File	:	Makefile  for building NeoMatrix 
#  Date:	July 2018

CC = g++
# Note: the following are OSX platform flags
# CC = clang++
# CFLAGS  = -v -Wall -std=c++0x -ggdb -fPIC

# adjust flags as necessary for your platform
CFLAGS  = -Wall -fPIC -std=c++11 

LDFLAGS =  -lpthread

all: main.o  matrix_test

matrix_test: main.o  
	$(CC) $(CFLAGS) $(LDFLAGS) main.o -o matrix_test

main.o: main.cpp neomatrix.hpp
	$(CC) $(CFLAGS) -c main.cpp -o main.o


#lib: neomatrix.o 
#	$(CC) -v -fPIC -shared neomatrix.o -o neolib.so -Wl

clean:
	rm -rf *.o  matrix_test
