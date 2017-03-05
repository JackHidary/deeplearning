import numpy as np
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares

# Set up summary operations
tf.summary.scalar('loss', loss)
merged_summary_op = tf.summary.merge_all()
log_dir = "D:\\Dropbox\\github\\stuff\\tensorboard_logs"

summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())


# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  _, c, summary = sess.run([train, loss, merged_summary_op], {x:x_train, y:y_train})

  summary_writer.add_summary(summary, i)

# evaluate training accuracy
curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
