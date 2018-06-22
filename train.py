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
end_step = 2000
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
	MSE = graph.get_collection('MSE_op')[0]
	accurary = graph.get_collection('accurary_op')[0]
else:
	im_data = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1], name='im_data')
	gt_data = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1], name='gt_data')
	net = CrowdCounter(im_data, gt_data)
	optimizer = net.get_optimizer()
	MSE = net.get_MSE()
	density = net.get_density()
	accurary = net.get_accuracy()
	sess.run(tf.global_variables_initializer())
	tf.add_to_collection('density_op', density)
	tf.add_to_collection('optimizer_op', optimizer)
	tf.add_to_collection('MSE_op', MSE)
	tf.add_to_collection('accurary_op', accurary)

step = -1
train_loss = 0
best_mae = sys.maxint
sn = SaveNet()
if use_tensorboard:
	train_writer = tf.summary.FileWriter(log_dir, sess.graph)
	tf.summary.scalar('train_loss', MSE)
	tf.summary.scalar('accurary', accurary)
	merged = tf.summary.merge_all()

for epoch in range(0, end_step):
	for blob in data_loader:
		step += 1
		data = blob['data']
		den = blob['den']
		_, train_loss, pred_den, summary = sess.run([optimizer, MSE, density, merged], feed_dict={im_data: data, gt_data: den})
		if step % disp_interval == 0:
			gt_count = np.sum(den)
			pred_count = np.sum(pred_den)
			print('[{}]/[{}], [{}], train_loss = {}, gt_count = {}, pred_count = {}'.format(epoch, end_step, step, train_loss, gt_count, pred_count))
			if use_tensorboard:
				train_writer.add_summary(summary, step)

	if epoch % 2 == 0:
		mae, mse = evaluate(sess, density, im_data, val_loader)
		if mae < best_mae:
			best_mae = mae
			best_mse = mse
			sn.save_net(sess, './models/crowncnn', epoch)
		print('[{}]/[{}], [{}], mae/best = {}/{}, mse/best = {}/{}'.format(epoch, end_step, step, mae, best_mae, mse, best_mse))

if use_tensorboard:
	train_writer.close()