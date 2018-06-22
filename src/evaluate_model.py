import numpy as np
from crowd_counter import CrowdCounter
import models

def evaluate(sess, density, im_data, data_loader):
	mae = 0.0
	mse = 0.0
	for blob in data_loader:
		data = blob['data']
		gt_den = blob['den']
		pred_den = sess.run(density, feed_dict={im_data: data})
		gt_count = np.sum(gt_den)
		pred_count = np.sum(pred_den)
		mae += abs(gt_count - pred_count)
		mse += (gt_count - pred_count) * (gt_count - pred_count)
	mae = mae / data_loader.get_num_samples()
	mse = np.sqrt(mse / data_loader.get_num_samples())
	return mae, mse
