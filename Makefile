.RECIPEPREFIX := $(.RECIPEPREFIX) 
SHELL := /bin/bash
.DELETE_ON_ERROR :

.PHONY : all debug

binary_name := mnist.out
source_name := mnist.cpp
header_list := header/mnist.h header/layer.h header/misc.h header/vector_operation.h
dataset := read_mnist/data/image_train.dat

all: $(binary_name)
all: override CXXFLAGS += -O3 -march=native

debug: $(binary_name)
debug: override CXXFLAGS += -g

$(binary_name) : $(source_name) $(header_list) $(dataset)
    g++ -o $@ $< -pthread $(CXXFLAGS)

$(dataset) :
    python3 read_mnist/convert.py

%.out : %.cpp
    g++ -o $@ $< $(CXXFLAGS)

