import tensorflow as tf
import requests
from io import StringIO
import PIL
from PIL import Image
from CNN_model import model_cnn

class model_test:

    def __init__(self):
        model_cnn_obj = model_cnn()
        self.x = tf.placeholder(tf.float32, shape=[None, 28, 28, 3])
        self.keep_prob = tf.placeholder(tf.float32)
        self.y_out_label = model_cnn_obj.forward_process(self.x, self.keep_prob)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, "./Model/model_cnn.ckpt")


    def read_image_from_url(self, url):
        try:
            req = requests.get(url, timeout = 10)
        except Exception :
            print('Get image failed from url !!!')

        image_file = StringIO(req.content)
        image = Image.open(image_file)
        image.convert('RGB')
        image.resize((28, 28), PIL.Image.ANTIALIAS)
        return image


    def test_run(self, url):

        image = self.read_image_from_url(url)
        self.y_out_label.run(feed_dict = {self.x : image, self.keep_prob: 0.5})

        pass