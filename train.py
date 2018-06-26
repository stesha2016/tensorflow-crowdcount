import tensorflow as tf
import numpy as np
import sys
from src.crowd_counter import CrowdCounter
from src.data_loader import ImageDataLoader
from src.evaluate_model import evaluate
from src.network import SaveNet, LoadNet

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

train_path = '/local/share/DeepLearning/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train'
train_gt_path = '/local/share/DeepLearning/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train_den'
val_path = '/local/share/DeepLearning/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/val'
val_gt_path = '/local/share/DeepLearning/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/val_den'
model_name = './models/crowncnn'
end_step = 1000
disp_interval = 500
base_pretrain = False
meta_path = './models/crowncnn-16.meta'
model_path = './models/'
use_tensorboard = True
log_dir = './summary'

data_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=True, pre_load=True)
val_loader = ImageDataLoader(val_path, val_gt_path, shuffle=True, gt_downsample=True, pre_load=True)

if base_pretrain:
	ln = LoadNet(meta_path)
	graph, sess = ln.load_net(sess, model_path)
	im_data = graph.get_tensor_by_name('im_data:0')
	gt_data = graph.get_tensor_by_name('gt_data:0')
	density = graph.get_collection('density_op')[0]
	optimizer = graph.get_collection('optimizer_op')[0]
	loss = graph.get_collection('loss_op')[0]
	accurary = graph.get_collection('accurary_op')[0]
else:
	im_data = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1], name='im_data')
	gt_data = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1], name='gt_data')
	net = CrowdCounter(im_data, gt_data)
	optimizer = net.get_optimizer()
	loss = net.get_loss()
	density = net.get_density()
	accurary = net.get_accuracy()
	sess.run(tf.global_variables_initializer())
	tf.add_to_collection('density_op', density)
	tf.add_to_collection('optimizer_op', optimizer)
	tf.add_to_collection('loss_op', loss)
	tf.add_to_collection('accurary_op', accurary)

train_loss = 0
best_mae = sys.maxint
sn = SaveNet()
step = -1
if use_tensorboard:
	train_writer = tf.summary.FileWriter(log_dir, sess.graph)
	tf.summary.scalar('train_loss', loss)
	tf.summary.scalar('accurary', accurary)
	merged = tf.summary.merge_all()

for epoch in range(0, end_step):
	for blob in data_loader:
		step += 1
		data = blob['data']
		den = blob['den']
		_, train_loss, pred_den, summary = sess.run([optimizer, loss, density, merged], feed_dict={im_data: data, gt_data: den})
		if use_tensorboard:
			train_writer.add_summary(summary, step)
		gt_count = np.sum(den)
		pred_count = np.sum(pred_den)
		if step % disp_interval == 0:
			print('epoch: %4d, step %4d, gt_cnt: %4.1f, et_cnt: %4.1f' % (epoch, step, gt_count, pred_count))

	if epoch % 2 == 0:
		mae, mse = evaluate(sess, density, im_data, val_loader)
		if mae < best_mae:
			best_mae = mae
			best_mse = mse
			best_epoch = epoch
			sn.save_net(sess, './models/crowncnn', epoch)
		print('EPOCH: %d, MAE: %.1f, MSE: %0.1f' % (epoch, mae, mse))
		print('BEST MAE: %0.1f, BEST MSE: %0.1f, BEST MODEL: %s' % (best_mae, best_mse, best_epoch))

if use_tensorboard:
	train_writer.close()