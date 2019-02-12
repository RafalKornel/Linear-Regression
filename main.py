#!/usr/bin/python3

import numpy as np
import pandas as pd
import random
import os, sys, datetime
import imageio
from model import Model
from matplotlib import pylab as pl

# Parameters:

epochs = 1000
learning_rate = 0.001
visualization = True
snap_freq = 10
batch_size = 5

# Load file and assign features and lables
_file = pl.loadtxt('input_file.txt')
x = _file[:, 0]
y = _file[:, 1]

# Randomly generate slope and bias values
rand_a = random.uniform(-10, 10)
rand_b = random.uniform(-10, 10)
print("Randomly generated slope:{0} and bias:{1} ".format(rand_a, rand_b))


# Initialize model, pass arguments, start learning
test = Model(x, y, learning_rate, batch_size, visualization, epochs, bias = rand_b, slope = rand_a, frequency = snap_freq)
test.learn()

#print(test.pick_random_batch(4))
print("Result slope:{0} and bias:{1} ".format(test.slope, test.bias))

# Visualization - generating plot and gif file
f = lambda x, a, b: a * x + b
pl.plot(x, y, 'o')
_x = pl.arange(-10, 10, 0.1)
pl.plot(_x, f(_x, test.slope, test.bias))

pl.show()

if visualization == True:
    images = []
    for filename in sorted(os.listdir('shots/'), key = lambda x : int(x[0:-4:])):
        images.append(imageio.imread("shots/{0}".format(filename)))
        os.remove("shots/{0}".format(filename))
    imageio.mimsave('gifs/{0}.gif'.format(random.randrange(1, 12032145)), images, duration = 1/snap_freq**2)
