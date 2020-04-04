import numpy as np

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path.append("./original_data");
from dataset.mnist import load_mnist

target_directory = "./data/";
try:
    print("Creating the output directory...");
    os.mkdir(target_directory);
except:
    pass;

#This line loads (training images, training labels) and (testing images, testing labels).
(x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize = False);

print("Preparing...");

l1 = [];
l2 = [];
l3 = [];
l4 = [];
for i in range(len(x_train)):
    l1.append(list(x_train[i]));
    l2.append(t_train[i]);
for i in range(len(x_test)):
    l3.append(list(x_test[i]));
    l4.append(t_test[i]);

from numpy import array

print("Writing to image_train.dat...");
with open(target_directory + 'image_train.dat', 'wb') as f:
    for i in range(len(l1)):
        array(l1[i], 'uint8').tofile(f);

print("Writing to label_train.dat...");
with open(target_directory + 'label_train.dat', 'wb') as f:
    for i in range(len(l2)):
        array(l2[i], 'uint8').tofile(f);

print("Writing to image_test.dat...");
with open(target_directory + 'image_test.dat', 'wb') as f:
    for i in range(len(l3)):
        array(l3[i], 'uint8').tofile(f);

print("Writing to label_test.dat...");
with open(target_directory + 'label_test.dat', 'wb') as f:
    for i in range(len(l4)):
        array(l4[i], 'uint8').tofile(f);

