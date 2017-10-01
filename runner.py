# This is a script that runs linear regression using gradient descent. It's not the optimum way to 
# run linear regression but it's cool for simply exploring gradient descent (which is used a lot in
# deep learning) with pyton in a familiar context.

# import numpy in a way that all numpy functions are loaded into the local namespace so we don't
# have to define numpy.<method>. We can simply call <method> e.g. numpy.array() vs. array()
from numpy import * 
# from numpy import genfromtext

# load in the csv and separate the values by comma (if you were to take the csv in text format)
# the values across rows would be separated by commas, hence the name comma separated value
# **duhh**
def run():
  points = genfromtxt('data.csv', delimiter=',')
  
  # This is the hyperparameter alpha which determines the size of the steps we take in the gradient
  # It determines how fast our model learns. If the learning rate is too low our model will be slow
  # to converge. However if it is too high it will overshoot our global minimum and never converge.
  # Ideally we would guess and check the value of our learning rate.
  learning_rate = 0.0001

  # y = mx + b <-- The model with parameters m and b that predict y given the value of x
  # It's a standard slope formula
  # Below we are initializing the parameters of our model at zero.
  initial_b = 0
  initial_m = 0
  # We're doing 1000 iterations because the training set is so small
  num_iterations = 1000

  [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
  print(b)
  print(m)

# Here is were we structure and run our gradient descent
def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
  b = starting_b
  m = starting_m

  for i in range(num_iterations):
    b, m = step_gradient(b, m, array(points), learning_rate)
  return [b, m]

# This is where the gradient descent really happens. We will use the gradient to minimize the
# error between our datapoints and the slope
def step_gradient(b_current, m_current, points, learningRate):
  
  # We first start by initializing our parameters
  b_gradient = 0
  m_gradient = 0
  # Total number of examples
  N = float(len(points))
  # Run through each example
  for i in range(0, len(points)):
    x = points[i, 0]
    y = points[i, 1]
    # Calculate the gradient for b 
    b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
    # Calculate the gradient for m
    m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
  # This is the update rule for gradient descent 
  new_b = b_current - (learningRate * b_gradient)
  new_m = m_current - (learningRate * m_gradient)
  return [new_b, new_m]

# Here we want to take the sum of squared errors from our datapoints to the slope in a current
# timestep.
def compute_error_for_given_points(b, m, points):
  totalError = 0
  # Run through each example
  for i in range(0, len(points)): 
    # estblish x and y from our dataset
    x = points[i, 0]
    y = points[i, 1]
    # sum of squared errors equation given our value of m and b
    totalError += (y - (m * x + b)) **2
  # return the average of the squared errors
  return totalError / float(len(points))


if __name__ == '__main__':
  run()
