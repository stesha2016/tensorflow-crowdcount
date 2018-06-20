import tensorflow as tf
import network as nw

def MCNN(im_data, bn=False):
    with tf.variable_scope('MCNN'):
        x1 = nw.conv2d(im_data, 16, kernel_size=9, padding='same', bn=bn)
        x1 = tf.layers.max_pooling2d(x1, 2, 2)
        x1 = nw.conv2d(x1, 32, kernel_size=7, padding='same', bn=bn)
        x1 = tf.layers.max_pooling2d(x1, 2, 2)
        x1 = nw.conv2d(x1, 16, kernel_size=7, padding='same', bn=bn)
        x1 = nw.conv2d(x1, 8, kernel_size=7, padding='same', bn=bn)

        x2 = nw.conv2d(im_data, 20, kernel_size=7, padding='same', bn=bn)
        x2 = tf.layers.max_pooling2d(x2, 2, 2)
        x2 = nw.conv2d(x2, 40, kernel_size=5, padding='same', bn=bn)
        x2 = tf.layers.max_pooling2d(x2, 2, 2)
        x2 = nw.conv2d(x2, 20, kernel_size=5, padding='same', bn=bn)
        x2 = nw.conv2d(x2, 10, kernel_size=5, padding='same', bn=bn)

        x3 = nw.conv2d(im_data, 24, kernel_size=5, padding='same', bn=bn)
        x3 = tf.layers.max_pooling2d(x3, 2, 2)
        x3 = nw.conv2d(x3, 48, kernel_size=3, padding='same', bn=bn)
        x3 = tf.layers.max_pooling2d(x3, 2, 2)
        x3 = nw.conv2d(x3, 24, kernel_size=3, padding='same', bn=bn)
        x3 = nw.conv2d(x3, 12, kernel_size=3, padding='same', bn=bn)

        _ = tf.concat([x1, x2, x3], axis=-1)
        _ = nw.conv2d(_, 1, kernel_size=1, padding='same', bn=bn)
        return _