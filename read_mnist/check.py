import numpy as np

import sys
sys.path.append("./original_data");
from dataset.mnist import load_mnist

from PIL import Image

import pickle

#This line loads training images, training labels (in or not in one-hot representation), testing images and testing labels (in or not in one-hot representation).
(x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize = False);
(x_train_2, t_train_one_hot), (x_test_2, t_test_one_hot) = load_mnist(flatten = True, normalize = False, one_hot_label = True);

#Used in `print_array()`.
#Converts `0.0` to `0` and `1.0` to `1`, but anything else isn't touched.
def convert_value(v):
    if (v == 0):
        return 0;
    elif (v == 1):
        return 1;
    else:
        return v;

def print_array(a):
    print("[", end = "");
    for i in range(len(a) - 1):
        print(convert_value(a[i]), ", ", sep = "", end = "");
    print(convert_value(a[len(a) - 1]), "]", sep = "");

print("image_train.size():", len(x_train));
print("image_train[0].size():", len(x_train[0]));
print("--- image_train[0] ---");
print_array(x_train[0]);
print("--- image_train[59999] ---");
print_array(x_train[59999]);

print();

print("label_train.size():", len(t_train));
print("label_train[0]:", t_train[0]);
print("label_train[59999]:", t_train[59999]);

print();

print("label_train_one_hot.size():", len(t_train_one_hot));
print("--- label_train_one_hot[0] ---")
print_array(t_train_one_hot[0]);
print("--- label_train_one_hot[59999] ---")
print_array(t_train_one_hot[59999]);

print();

print("image_test.size():", len(x_test));
print("image_test[0].size():", len(x_test[0]));
print("--- image_test[0] ---");
print_array(x_test[0]);
print("--- image_test[9999] ---");
print_array(x_test[9999]);

print();

print("label_test.size():", len(t_test));
print("label_test[0]:", t_test[0]);
print("label_test[9999]:", t_test[9999]);

print();

print("label_test_one_hot.size():", len(t_test_one_hot));
print("--- label_test_one_hot[0] ---")
print_array(t_test_one_hot[0]);
print("--- label_test_one_hot[9999] ---")
print_array(t_test_one_hot[9999]);

