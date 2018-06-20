import tensorflow as tf

def conv2d(x, f, bn=False, relu=True, **k):
    _x = tf.layers.conv2d(x, f, **k)
    if bn:
        _x = tf.layers.batch_normalization(_x, momentum=0, epsilon=0.001)
    if relu:
        _x = tf.nn.relu(_x)
    return _x

def save_net(fname, net):
    import h5py

def load_net(fname, net):
    import h5py