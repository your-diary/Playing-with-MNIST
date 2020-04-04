.RECIPEPREFIX := $(.RECIPEPREFIX) 
SHELL := /bin/bash
.DELETE_ON_ERROR :

.PHONY : 

read_mnist.out : read_mnist.cpp read_mnist.h
    g++ -O3 -o $@ $< $(CXXFLAGS)

