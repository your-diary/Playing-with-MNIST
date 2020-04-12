# Playing with MNIST

<!-------------------------------------->

## Introduction

In this project, we implement a hand-written digit recognizer using C++ and MNIST dataset.

<!-------------------------------------->

## Contents

This project includes multiple contents.

### Theoretical Explanations

[`pdf/mnist.pdf`](pdf/mnist.pdf) gives theoretical explanations about machine learning, neural networks, and so on.

### MNIST Dataset Reader

The codes under `read_mnist/` implement a reader for MNIST Dataset which works in C++. You can just pick the directory to embed the reader into your application. See [`read_mnist/README.md`](read_mnist/README.md) for the detail.

### Hand-Written Digit Recognizer

`./mnist.cpp` and the headers under `./header` implement a hand-written digit recognizer. Consult [`pdf/mnist.pdf`](pdf/mnist.pdf) for the details about implementations and the code structure.

We also provide a tester `recognition_of_user_supplied_data/draw_digit.py` in which a user draws a digit using his/her mouse and the true digit the user intended is inferred. See the animation below for a demo.

| ![Demo of `draw_digit.py`.](recognition_of_user_supplied_data/demo/demo.gif) |
| :--: |
| Demo of `draw_digit.py`. |

<!-------------------------------------->

## Supported Environments

This project is cross-platform; Linux, macOS and Windows are supported.

<!-------------------------------------->

<!-- vim: set spell: -->

