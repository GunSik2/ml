import tensorflow as tf

# Multi variable linear regression
# Hypothesis: H(x) = w1 * x1 + w2 * x2 + b
# Cost Fun.: cost(w,b) = 1/m * Sum(H(x) - y)^2
# Gradient descent: W := W - alpah * 1/m * Sum((W*x - y) * x)

# training data
x1 = [1., 0., 3., 0., 5.]
x2 = [0., 2., 0., 4., 0.]
y = [1, 2, 3, 4, 5]

# Initial value (w, b)
w1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
w2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# Hypothesis
hypothesis = w1 * x1 + w2 * x2 + b

# Cost function
cost = tf.reduce_mean(tf.square(hypothesis - y))

# Gradicent descent
a = tf.Variable(0.1)  # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# launch
# before starting, initialize the variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# fit the line
for step in range(1000):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(w1), sess.run(w2), sess.run(b))

# learns best fit is w1: [1] w2: [1] b: [0]