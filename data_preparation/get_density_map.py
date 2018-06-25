import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pandas as pd

data_path = '../data/Annotations/'
image_path = '../data/Images/'
output_path = '../data/ground_truth/'

if not os.path.exists(output_path):
	os.makedirs(output_path)

def parse_xml(file):
	f = open(file, 'r')
	tree = ET.parse(f)
	root = tree.getroot()
	heads = []

	size = root.find('size')
	w = int(size.find('width').text)
	h = int(size.find('height').text)

	for obj in root.iter('object'):
		xmlbox = obj.find('bndbox')
		head = [int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
				int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text)]
		heads.append(head)

	return (h, w), heads

def generate_density_map(heads, imsize):
	densmap = np.zeros(imsize, np.float32)
	sigma = 4.0
	ksize = (15, 15)
	for head in heads:
		x0 = int((head[0] + head[2]) / 2)
		y0 = int((head[1] + head[3]) / 2)
		dens = np.zeros_like(densmap)
		dens[y0, x0] = 1.0
		dens = cv2.GaussianBlur(dens, ksize, sigma)
		densmap += dens
	return densmap

data_files = os.listdir(data_path)
for file in data_files:
	imsize, heads = parse_xml(os.path.join(data_path, file))
	densmap = generate_density_map(heads, imsize)
	base_name = '.'.join(file.split('.')[:-1])
	file_name = os.path.join(output_path, base_name + '.csv')
	df = pd.DataFrame(densmap)
	df.to_csv(file_name, header=False, index=False)