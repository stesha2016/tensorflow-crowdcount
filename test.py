import tensorflow as tf
import numpy as np
from src.network import LoadNet
from src.data_loader import ImageDataLoader

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

meta_path = './models/crowncnn-14.meta'
model_path = './models/'
val_path = '/local/share/DeepLearning/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/val'
val_gt_path = '/local/share/DeepLearning/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/val_den'

val_loader = ImageDataLoader(val_path, val_gt_path, shuffle=True, gt_downsample=True, pre_load=True)

ln = LoadNet(meta_path)
graph, sess = ln.load_net(sess, model_path)

im_data = graph.get_tensor_by_name('im_data:0')
density_op = graph.get_collection('density_op')[0]

for blob in val_loader:
	data = blob['data']
	den = blob['den']
	pred_data = sess.run(density_op, feed_dict={im_data: data})

	gt_count = np.sum(den)
	pred_count = np.sum(pred_data)
	print('gt_count is {}, pred_count is {}'.format(gt_count, pred_count))
