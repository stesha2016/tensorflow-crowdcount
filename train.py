import tensorflow as tf
from src.crowd_counter import CrowdCounter
from src.data_loader import ImageDataLoader

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

train_path = '/local/share/DeepLearning/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train'
train_gt_path = '/local/share/DeepLearning/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train_den'

im_data = tf.placeholder(dtype=tf.float32, shape=[None, 1024, 768, 1])
gt_data = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])
net = CrowdCounter(im_data, gt_data)
optimizer = net.get_optimizer()

data_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=False, pre_load=True)
for blob in data_loader:
	im_data = blob['data']
	gt_data = blob['den']
	print(im_data.shape)
	print(gt_data.shape)
	break;
