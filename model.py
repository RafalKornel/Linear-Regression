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
        self.current_error = None


    def __str__(self):
        return "slope: {0} | bias: {1}".format(self.slope, self.bias)


    def predict(self,x):
        ''' Applies given x value to current trained linear function '''
        return self.slope * x + self.bias


    def update_current_error_value(self, error):
        ''' Updates object current_error value '''
        self.current_error = error


    def calc_square_error(self, entry_index):
        ''' Calculates square error for given entry '''
        return (self.labels[entry_index] - self.predict(self.features[entry_index]))**2


    def calc_error_on_batch(self, batch):
        ''' Calculate average error value over given batch '''
        _sum = sum( [self.calc_square_error(x) for x in batch] )
        return _sum / len(batch)


    def calc_slope_gradient(self, x, y, use_batch = False):
        ''' Calculate slope gradient, i.e. partial derivative over slope variable in given point, returned
            value is array in form: [gradient_slope_value, gradient_bias_value]. If use_batch = True function
            is going to calculate gradient over given batch'''
        if use_batch:
            _sum = 0
            for i in range(len(x)):
                _sum += self.calc_slope_gradient(x[i], y[i])
            return _sum/len(x)
        else:
            return 2 * self.slope * x**2 + 2 * x + self.bias - 2 * y * x


    def calc_bias_gradient(self, x, y, use_batch = False):
        ''' Calculate bias gradient, i.e. partial derivative over bias variable in given point'''
        if use_batch:
            _sum = 0
            for i in range(len(x)):
                _sum += self.calc_bias_gradient(x[i], y[i])
            return _sum/len(x)
        else:
            return 2 * self.slope * x + 2 * self.bias - 2 * y


    def pick_random_batch(self, length):
        ''' Returns random values from model's features '''
        return [np.random.randint(0, len(self.features)) for i in range(length)]


    def plot_model(self, iteration = 0, save_plot = False, show_plot = False):
        ''' Creates plot in matplotlib's pylab module, if save_plot = True, saves plot to shots folder, if show_plot = True, shows plot (opens pylab window with plot '''
        _x = pl.arange(min(self.features), max(self.features), 1)
        pl.plot(self.features, self.labels, 'k.')
        pl.plot(_x, f(_x, self.slope, self.bias), linewidth = 1.0)
        pl.grid(True)
        axes = pl.gca()
        axes.set_xlim([min(self.features), max(self.features)])
        axes.set_ylim([min(self.labels), max(self.labels)])

        if save_plot:
            if not os.path.exists('shots/'): os.mkdir('shots/')
            pl.savefig("shots/{0}".format(iteration), bbox_inches = 'tight')

        if show_plot:
            pl.show()

        pl.clf()  # clears plot


    def learn(self):
        ''' Main learning method, iterates ofer epochs generating batch of features and labels, calculating how much should slope and bias change and changing them. '''
        for epoch in range(self.epochs):

            # Picks random values from features and labels
            samp_indices = self.pick_random_batch(self.batch_size)
            samp_feature = self.features[samp_indices]
            samp_label = self.labels[samp_indices]

            # Generates next slope and bias values, then applies them to current slope and bias
            next_slope_step = self.calc_slope_gradient(samp_feature, samp_label, use_batch = True) * self.learning_step
            next_bias_step = self.calc_bias_gradient(samp_feature, samp_label, use_batch = True) * self.learning_step

            if self.visualize == True:
                print(next_slope_step, next_bias_step, np.sqrt(self.calc_error_on_batch(samp_indices)))

            self.slope -= next_slope_step
            self.bias -= next_bias_step

            # Creates png of plot with current weights
            if self.visualize:
                if epoch % self.frequency == 0:
                    self.plot_model(iteration = epoch, save_plot = True)#export_model_to_png(iteration = epoch)
