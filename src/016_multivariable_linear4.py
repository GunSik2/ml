import tensorflow as tf
import numpy as np


# Single variable linear regression
# Hypothesis: H(x) = W * X
# Cost Fun.: cost(w,b) = 1/m * Sum(H(x) - y)^2
# Gradient descent: W := W - alpah * 1/m * Sum((W*x - y) * x)

# training data
xy = np.loadtxt('016_train.txt', unpack=True, dtype='float32')
x = xy[0:-1]
y = xy[-1]

# Initial value (w, b)
W = tf.Variable(tf.random_uniform([1, len(x)], -1., 1.))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Hypothesis
hypothesis = tf.matmul(W, X)

# Cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Gradicent descent
a = tf.Variable(0.1)  # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)  # goal is minimize cost


# launch
# before starting, initialize the variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# fit the line
for step in range(1000):
    sess.run(train,  feed_dict={X: x, Y: y})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x, Y: y}), sess.run(W))

# learns best fit is w: [0 1 1]
print(sess.run(hypothesis, feed_dict={X: [[1], [3], [1]]}))
