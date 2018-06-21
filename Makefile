#  NeoMatrix Makefile

# Note: this was developed on the OSX platform 
CC = g++
#CC = clang++
CFLAGS  = -v -Wall -std=c++0x -ggdb -fPIC
CFLAGS  = -v -Wall -std=c++0x -ggdb 
# CFLAGS  = -Wall -fPIC -std=c++11 
LDFLAGS =  -lpthread

all: main.o  
	$(CC) $(CFLAGS) $(LDFLAGS) main.o -o matrix_test

main.o: main.cpp neomatrix.hpp
	$(CC) $(CFLAGS) -c main.cpp -o main.o


#lib: neomatrix.o 
#	$(CC) -v -fPIC -shared neomatrix.o -o libmatlib.so -Wl

clean:
	rm -rf *.o  matrix_test
