import tensorflow as tf
from CNN_model import model_cnn
from Data_Prepare import Data_Prepare

class model_train:

    def __init__(self):
        self.data_parepare = Data_Prepare()
        self.model_prepare = model_cnn()
        pass

    def train_main(self):

        iter_num = 10
        batch_size = 5
        capacity = 10
        min_after_dequeue = 5

        x = tf.placeholder(tf.float32, shape=[None, 28, 28, 3])
        y = tf.placeholder(tf.float32, shape=[None, 2])
        keep_prob = tf.placeholder(tf.float32)

        image_batch, label_batch = self.data_parepare.generate_batch(batch_size, capacity, min_after_dequeue)

        y_out = self.model_prepare.forward_process(x, keep_prob)
        train_step = self.model_prepare.backpropagation_process(y_out, y)


        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for iter in range(iter_num):
                print(iter)
                image_batch_train, label_batch_train = sess.run([image_batch, label_batch])
                train_step.run(feed_dict={x: image_batch_train, y: label_batch_train, keep_prob: 0.5})

if __name__ == '__main__':
    train_obj = model_train()
    train_obj.train_main()