.RECIPEPREFIX := $(.RECIPEPREFIX) 
SHELL := /bin/bash
.DELETE_ON_ERROR :

.PHONY : 

mnist.out : mnist.cpp ./header/mnist.h ./header/layer.h ./header/vector_operation.h
    g++ -O3 -o $@ $< -pthread -march=native $(CXXFLAGS)

%.out : %.cpp
    g++ -o $@ $< $(CXXFLAGS)

