# Introduction

This directory consists of scripts which convert MNIST dataset to easy-to-read binary format. Each image in MNIST dataset represents one hand-written number, has only one channel (i.e. grayscale) and has the dimension `28x28`.

| ![Example of Images in MNIST Dataset](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png) |
| :--: |
| Example of Images in MNIST Dataset (cited from [Wikipedia](https://en.wikipedia.org/wiki/MNIST_database)) |

# Data

Executing `convert.py`, the resultant datasets are created in `./data`. Since these data are normal binary data whose formats are described in the table below, you can easily write a code to read them into arbitrary containers or, in `C++`, you can just use our `read_mnist.h` to load them into `std::vector`.

| File Name | Type | Range | # of Elements | Notes |
|:--        |:--   |:--    |:--            |:--    |
| `image_train.dat` | `unsigned char` | `[0, 255]` | `28*28*60000` | By reading consecutive `28*28=784` data, you get one image for training. | 
| `label_train.dat` | `unsigned char` | `[0,9]` | `60000` | Each data represents the true label for the corresponding image in `image_train.dat`. If *N*th image in `image_train.dat` expresses the number `3`, *N*th element in `label_train.dat` is `3`. |
| `image_test.dat` | `unsigned char` | `[0, 255]` | `28*28*10000` | By reading consecutive `28*28=784` data, you get one image for testing. | 
| `label_test.dat`  | `unsigned char` | `[0,9]` | `10000` | Each data represents the true label for the corresponding image in `image_test.dat`. If *N*th image in `image_test.dat` expresses the number `3`, *N*th element in `label_test.dat` is `3`. |

# Miscellaneous Files

## For Users

- `convert.py`

This script downloads MNIST dataset and converts each data into the formats described in [Data](#data).

- `read_mnist.h`

This `C++` library implements functions to load MNIST dataset into `std::vector`. To understand the usage, read the source code or copy and paste the codes from `read_mnist.cpp`.

- `read_mnist.cpp`

This `C++` source shows an example of using `read_mnist.h`.

## For Developers

- `check.py`

This `Python 3` script outputs the same contents as those of `read_mnist.cpp`. However, it is essentially a fork of [the script written by someone](https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/ch03/mnist_show.py) and the data **before** our conversions are used. Thus, by comparing the two outputs, you can check if or not the implementation of `read_mnist.*` are correct. Currently they exactly correspond to each other. Rather than executing this script directly, you may want to use `compare.sh`.

- `compare.sh`

This script executes `read_mnist.cpp` and `check.py` to check if they have exactly the same outputs. As explained above, they should.

<!-- vim: set spell: -->

