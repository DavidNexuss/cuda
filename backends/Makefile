backend.a: loader.o
	ar rcs backend.a loader.o
loader.o: loader.cpp
	nvcc -g -c loader.cpp -I ../include	
