import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


batch_size = 128
n_class = 10
learning_rate = 0.01


lstm_size = 128

# 训练样本， 第一个28表示time_step， 第二个28表示特征维度
# mnist中图像为灰度图，size为[28, 28], 通过学习行之间的序列
# 关系，所以time_step为28， 每一行的像素作为输入，所以
# input_size为28
X = tf.placeholder(tf.float32, shape = [batch_size, 28, 28])
y = tf.placeholder(tf.float32, shape = [batch_size, 10])

# 隐层至输出层之间的权重矩阵
W = tf.Variable(tf.random_normal(shape = [lstm_size, n_class]))
b = tf.Variable(tf.random_normal(shape = [n_class]))

# 定义一个LSTM cell
lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)

# outputs : [ batch_size, 1, lstm_size ]
outputs_hidden, states = tf.nn.dynamic_rnn(lstm_cell, X, dtype=tf.float32)

# 转置操作， outputs_hidden to be [1, batch_size, lstm_size]
outputs_hidden = tf.transpose(outputs_hidden, [1, 0, 2])

# outputs_hidden to be [batch_size, lstm_size]
outputs_hidden = outputs_hidden[-1]

# 输出层输出, outputs大小为[batch_size, 1]
outputs = tf.add(tf.matmul(outputs_hidden, W), b)

# 计算cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=y))

# BPTT过程
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape((batch_size, 28, 28))
        sess.run(optimizer, feed_dict = {X : batch_xs, y : batch_ys})
        print(sess.run(cost, feed_dict = {X : batch_xs, y : batch_ys}))

