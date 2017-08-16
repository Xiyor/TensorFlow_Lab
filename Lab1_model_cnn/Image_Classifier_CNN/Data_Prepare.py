import tensorflow as tf
import os
import numpy as np

class Data_Prepare:
    '''
    类Data_Prepare用于数据准备工作
    '''

    def __init__(self):
        pass

    def filename_label_pipeline(self):
        '''
        生成文件名和标签列表
        :param file_path: 数据存储路径
        :return: file list 和 label list
        '''

        cur_dir_path = os.getcwd()
        data_dir = ''
        if os.path.exists(os.path.join(cur_dir_path, 'Image')):
            data_dir = os.path.join(cur_dir_path, 'Image')
        else:
            print("Cannot find data dir !!!")

        label_list = []
        filename_list = []
        label_iter = 0
        for dir in os.listdir(data_dir):
            label_vec = [0] * len(os.listdir(data_dir))
            label_vec[label_iter] = 1
            for image_file in os.listdir(os.path.join(data_dir,dir)):
                label_list.append(label_vec)
                filename_list.append(os.path.join(data_dir, dir, image_file))
            label_iter += 1

        return label_list, filename_list

        pass

    def generate_batch(self, batch_size, capacity, min_after_dequeue):
        '''
        生成用于训练的batch化的数据
        :param label_list:
        :param file_list:
        :return:
        '''

        labels, images = self.filename_label_pipeline()
        print(images)
        print(labels)
        images = tf.cast(images, tf.string)
        labels = tf.cast(labels, tf.int32)
        image_label_queue = tf.train.slice_input_producer([images, labels])
        image_queue = tf.read_file(image_label_queue[0])
        image_queue = tf.image.decode_jpeg(image_queue, channels = 3)
        image_queue = tf.image.resize_image_with_crop_or_pad(image_queue, 28, 28)
        image_queue = tf.image.per_image_standardization(image_queue)
        image_batch, label_batch = tf.train.shuffle_batch([image_queue, image_label_queue[1]], \
                                            batch_size=batch_size, capacity=capacity, \
                                            min_after_dequeue=min_after_dequeue)

        label_batch = tf.reshape(label_batch, [batch_size, 2])
        image_batch = tf.cast(image_batch, tf.float32)
        return image_batch, label_batch

if __name__ == '__main__':
    data_prepare = Data_Prepare()
    label_list, filename_list = data_prepare.filename_label_pipeline()
    print('label_list is {}'.format(label_list))
    print('filename_list is {}'.format(filename_list))

