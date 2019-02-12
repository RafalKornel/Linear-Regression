import numpy as np
import pandas as pd
import random, os
from matplotlib import pylab as pl

_x = pl.arange(-10, 10, 0.1)
f = lambda x, a, b: x * a + b  # linear function

class Model():
    def __init__(self, features, labels, learning_step, batch_size = 1, visualize = True, epochs = 100, bias = 1, slope = 1, frequency = 5):
        ''' Features - series of input variables, labels - coresponding values, learning_step - determines how much
            would constants change in iteration, epochs - number of iterations, bias and slope - in y = mx + b, m is slope
            and b is bias, visualize - if true model will create pngs and gif, frequency - determines how often model 
            takes snapshot in png format (higher value means less frequent) '''
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.learning_step = learning_step
        self.epochs = epochs
        self.batch_size = batch_size
        self.bias = bias
        self.slope = slope
        self.frequency = frequency
        self.visualize = visualize

    def __str__(self):
        return "slope: {0} | bias: {1}".format(self.slope, self.bias)

    def predict(self,x):
        ''' Applies given x value to current trained linear function '''
        return self.slope * x + self.bias

    def calc_square_error(self, entry):
        ''' Calculates square error for given entry '''
        return (entry - self.predict(entry))**2

    def calc_error_on_batch(self, batch):
        ''' Calculate average error value over given batch '''
        _sum = sum( [self.calc_square_error(x) for x in batch] )
        return _sum / len(batch)

    def calc_gradient_slope(self, x, y, use_batch = False):
        ''' Calculate slope gradient, i.e. partial derivative over slope variable in given point, returned
            value is array in form: [gradient_slope_value, gradient_bias_value]. If use_batch = True function
            is going to calculate gradient over given batch'''
        if use_batch:
            _sum = [0, 0]
            for i in range(len(x)):
                _sum[0] += self.calc_gradient_slope(x[i], y[i])[0]
                _sum[1] += self.calc_gradient_slope(x[i], y[i])[1]
            return [x/2 for x in _sum]
        else:
            return [2 * x**2, 2 * x + self.bias - 2 * x * x]

    def calc_gradient_bias(self, x, y, use_batch = False):
        ''' Calculate bias gradient, i.e. partial derivative over bias variable in given point'''
        if use_batch:
            _sum = [0, 0]
            for i in range(len(x)):
                _sum[0] += self.calc_gradient_bias(x[i], y[i])[0]
                _sum[1] += self.calc_gradient_bias(x[i], y[i])[1]
            return [x/2 for x in _sum]
        else:
            return [2, 2 * self.slope * x - 2 * y]

    def pick_random_batch(self, length):
        ''' Returns random values from model's features '''
        return [np.random.randint(0, len(self.features)) for i in range(length)]
        #return np.random.choice(self.features, length)

    def deternime_next_step_slope(self, gradient_slope):
        ''' Determines how much should slope value change in this iteration '''
        return (self.slope * gradient_slope[0] + gradient_slope[1]) * self.learning_step

    def determine_next_step_bias(self, gradient_bias):
        ''' Determines how much should bias value change in this iteration '''
        return (self.bias * gradient_bias[0] + gradient_bias[1]) * self.learning_step

    def export_model_to_png(self, iteration):
        ''' Creates series of plots in png format, later converted to gif animation for visualization '''
        if not os.path.exists('shots/'): os.mkdir('shots/')
        pl.plot(_x, f(_x, self.slope, self.bias))
        pl.plot(self.features, self.labels, 'k.')
        pl.grid(True)
        axes = pl.gca()
        axes.set_xlim([-10, 10])
        axes.set_ylim([-10, 10])
        pl.savefig("shots/{0}".format(iteration), bbox_inches = 'tight')
        pl.clf()

    def learn(self):
        for epoch in range(self.epochs):

            # Picks random values from features and labels
            samp_indices = self.pick_random_batch(self.batch_size)
            samp_feature = self.features[samp_indices]
            samp_label = self.labels[samp_indices]

            # Generates next slope and bias values
            next_slope = self.deternime_next_step_slope(self.calc_gradient_slope(samp_feature, samp_label, use_batch = True))
            next_bias = self.determine_next_step_bias(self.calc_gradient_bias(samp_feature, samp_label, use_batch = True))

            # Applies generated values i.e. moves bias and slope values closer to their minimum
            self.slope -= next_slope
            self.bias -= next_bias

            if self.visualize:
                if epoch % self.frequency == 0:
                    self.export_model_to_png(iteration = epoch)
