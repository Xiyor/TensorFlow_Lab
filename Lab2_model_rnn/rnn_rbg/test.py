import tensorflow as tf
lstm_size = tf.placeholder(tf.int32, None)
with tf.Session() as sess:
    print(sess.run(lstm_size, feed_dict = {lstm_size: 10}))