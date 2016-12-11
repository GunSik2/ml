import tensorflow as tf

# Single variable linear regression
# Hypothesis: H(x) = Wx + b
# Cost Fun.: cost(w,b) = 1/m * Sum(H(x) - y)^2
# Gradient descent: W := W - alpah * 1/m * Sum((W*x - y) * x)

# training data
x = [1., 2., 3.]
y = [1., 2., 3.]

# Initial value (w, b)
w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# Hypothesis
hypothesis = w * x + b

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
        print(step, sess.run(cost), sess.run(w), sess.run(b))

# learns best fit is w: [1] b: [0]