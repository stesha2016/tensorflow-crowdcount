import tensorflow as tf
import models
import numpy as np

lr = 0.00005
class CrowdCounter():
	def __init__(self, im_data, gt_data):
		self.im_data = im_data
		self.gt_data = gt_data
		self.density = models.MCNN(im_data)

	def get_density(self):
		return self.density

	def get_MSE(self):
		return tf.losses.mean_squared_error(self.gt_data, self.density)

	def get_optimizer(self):
		return tf.train.AdamOptimizer(learning_rate=lr).minimize(self.get_MSE(), var_list=tf.trainable_variables())
