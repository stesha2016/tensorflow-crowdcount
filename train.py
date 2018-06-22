import tensorflow as tf
import numpy as np
import sys
from src.crowd_counter import CrowdCounter
from src.data_loader import ImageDataLoader
from src.evaluate_model import evaluate
from src.network import SaveNet

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

im_data = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])
gt_data = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])
net = CrowdCounter(im_data, gt_data)
optimizer = net.get_optimizer()
MSE = net.get_MSE()
density = net.get_density()

data_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=True, pre_load=True)
val_loader = ImageDataLoader(val_path, val_gt_path, shuffle=True, gt_downsample=True, pre_load=True)
sess.run(tf.global_variables_initializer())

step = -1
train_loss = 0
best_mae = sys.maxint
sn = SaveNet()
meta_save = True
for epoch in range(0, end_step):
	for blob in data_loader:
		step += 1
		data = blob['data']
		den = blob['den']
		_, train_loss, pred_den = sess.run([optimizer, MSE, density], feed_dict={im_data: data, gt_data: den})
		if step % disp_interval == 0:
			gt_count = np.sum(den)
			pred_count = np.sum(pred_den)
			print('[{}]/[{}], [{}], train_loss = {}, gt_count = {}, pred_count = {}'.format(epoch, end_step, step, train_loss, gt_count, pred_count))

	if epoch % 2 == 0:
		mae, mse = evaluate(sess, density, im_data, val_loader)
		if mae < best_mae:
			best_mae = mae
			best_mse = mse
			sn.save_net(sess, './models/crowncnn', epoch, meta_save)
			if meta_save:
				meta_save = False

		print('[{}]/[{}], [{}], mae/best = {}/{}, mse/best = {}/{}'.format(epoch, end_step, step, mae, best_mae, mse, best_mse))