import tensorflow as tf

conv_init = tf.initializers.random_normal(0, 0.01)
def conv2d(x, f, bn=False, relu=True, **k):
    _x = tf.layers.conv2d(x, f, kernel_initializer=conv_init, **k)
    if bn:
        _x = tf.layers.batch_normalization(_x, momentum=0, epsilon=0.001)
    if relu:
        _x = tf.nn.relu(_x)
    return _x

class SaveNet():
	def __init__(self):
		self.saver = tf.train.Saver()
	def save_net(self, sess, model_name, step, meta_save):
		self.saver.save(sess, model_name, global_step=step, write_meta_graph=meta_save)

def load_net(fname, net):
    import h5py