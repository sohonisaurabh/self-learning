# Solution is available in the other "solution.py" tab
import tensorflow as tf

softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

# TODO: Print cross entropy from session
with tf.Session() as sess:
    #print (sess.run(tf.mul(one_hot,tf.log(softmax)), feed_dict={softmax: softmax_data, one_hot: one_hot_data}))
    cross_entropy = sess.run(tf.reduce_sum(tf.mul(-1.0, tf.mul(one_hot, tf.log(softmax)))), feed_dict={softmax: softmax_data, one_hot: one_hot_data})

print(cross_entropy)
