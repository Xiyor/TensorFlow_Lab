import tensorflow as tf
import requests
from io import BytesIO
import PIL
from PIL import Image
from CNN_model import model_cnn
import numpy as np

class model_test:

    def __init__(self):
        model_cnn_obj = model_cnn()
        self.x = tf.placeholder(tf.float32, shape=[None, 28, 28, 3])
        self.keep_prob = tf.placeholder(tf.float32)

        self.y_out_label = tf.argmax(model_cnn_obj.forward_process(self.x, self.keep_prob), dimension = 1)

        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, "./Model/model_cnn.ckpt")


    def read_image_from_url(self, url):
        try:
            req = requests.get(url, timeout = 100)
        except Exception :
            print('Get image failed from url !!!')

        image_file = BytesIO(req.content)
        image = Image.open(image_file)
        image.convert('RGB')
        image = image.resize((28, 28), PIL.Image.ANTIALIAS)
        print(np.asarray(image).reshape(1, 28, 28, 3)/255)
        return np.asarray(image).reshape(1, 28, 28, 3)/255


    def test_run(self, url):

        image = self.read_image_from_url(url)
        return self.sess.run(self.y_out_label, feed_dict = {self.x : image, self.keep_prob: 0.5})


if __name__ == '__main__':
    test_obj = model_test()
    print(test_obj.test_run('https://gss0.baidu.com/-4o3dSag_xI4khGko9WTAnF6hhy/zhidao/pic/item/91ef76c6a7efce1b490b4e91a951f3deb48f654d.jpg'))