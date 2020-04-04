.RECIPEPREFIX := $(.RECIPEPREFIX) 
SHELL := /bin/bash
.DELETE_ON_ERROR :

.PHONY : all debug

binary_name := mnist.out
source_name := mnist.cpp
header_list := header/mnist.h header/layer.h header/misc.h header/vector_operation.h

all: $(binary_name)
all: override CXXFLAGS += -O3 -march=native

debug: $(binary_name)
debug: override CXXFLAGS += -g

$(binary_name) : $(source_name) $(header_list)
    g++ -o $@ $< -pthread $(CXXFLAGS)

%.out : %.cpp
    g++ -o $@ $< $(CXXFLAGS)

