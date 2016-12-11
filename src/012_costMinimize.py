import tensorflow as tf

# Single variable linear regression
# Hypothesis: H(x) = Wx + b
# Cost Fun.: cost(w,b) = 1/m * Sum(H(x) - y)^2
# Gradient descent: W := W - alpah * 1/m * Sum((W*x - y) * x)

# training data
x = [1., 2., 3., 4.]
y = [2., 4., 6., 8.]

# Initial value (w, b)
W = tf.Variable(tf.random_uniform([1], -10000., 10000.))
b = 0

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Hypothesis
hypothesis = W * X + b

# Cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Gradicent descent // manual implementation
W1 = W - tf.mul(0.1, tf.reduce_mean(tf.mul( ( tf.mul(W, X) - Y ), X ), ))
update = W.assign(W1)


# launch
# before starting, initialize the variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# fit the line
for step in range(1000):
    sess.run(update, feed_dict={X: x, Y: y})
    print(step, sess.run(cost, feed_dict={X: x, Y: y}), sess.run(W))

# learns best fit is w: [2] b: [0]
print(sess.run(hypothesis, feed_dict={X: 5}))
print(sess.run(hypothesis, feed_dict={X: 2.5}))