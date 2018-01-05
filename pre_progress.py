# -*-coding:utf-8-*-
import cv2
import os
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
show = False
def deout(image):
	"""去掉外边框
	Args:image type:np.array
	return:np.array image
	"""
	h,w = image.shape
	all_avg = np.mean(image)
	flag = 0
	new_h = 0
	new_w = 0
	new_h_end = h
	new_w_end = w
	for i in range(h):
		if np.mean(image[i,:])>all_avg:
			flag += 1
		if flag == 10:
			new_h = i
			break
	flag = 0
	for j in range(w):
		if np.mean(image[:,j])>all_avg:
			flag += 1
		if flag == 10:
			new_w = j
			break
			
	flag = 0
	for i in range(h-1,0, -1):
		if np.mean(image[i,:])>all_avg:
			flag += 1
		if flag == 10:
			new_h_end = i
			break
	flag = 0
	for j in range(w-1, 0, -1):
		if np.mean(image[:,j])>all_avg:
			flag += 1
		if flag == 10:
			new_w_end = j
			break
	return image[new_h:new_h_end,new_w:new_w_end]
	
def get_in(image):
	"""获取仪表盘读数"""
	kernel = np.ones((3,3),np.uint8)
	temp = cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel)
	h,w = temp.shape
	all_avg = np.mean(temp)
	print(all_avg)
	flag = 0
	new_h = 0
	new_w = 0
	new_h_end = h
	new_w_end = w
	for i in range(h):
		if np.mean(temp[i,:])<all_avg:
			flag += 1
		if flag == 20:
			new_h = i
			break
	flag = 0
	for j in range(w):
		if np.mean(temp[:,j])<all_avg:
			flag += 1
		if flag == 15:
			new_w = j
			break
			
	flag = 0
	for i in range(h-1,0, -1):
		if np.mean(temp[i,:])<all_avg:
			flag += 1
		if flag == 20:
			new_h_end = i
			break
	flag = 0
	for j in range(w-1, 0, -1):
		if np.mean(temp[:,j])<all_avg:
			flag += 1
		if flag == 15:
			new_w_end = j
			break
	return image[new_h:new_h_end,new_w:new_w_end]
	
def get_num(image):
	h,w = image.shape
	temp = []
	d = int(w/6)
	for j in range(1,7):
		index_x = int(d/2)
		index_y_0 = int(d/2)
		index_y_d = int(d/2)
		img = image[:,(j-1)*d:j*d]
		while d-1 != index_x and sum(img[:,index_x]) > 0:
			index_x += 1
		while d-1 != index_y_d and sum(img[index_y_d,:]) > 0:
			index_y_d += 1
		while 0 != index_y_0 and sum(img[index_y_0,:]) > 0:
			index_y_0 -= 1
		img = cv2.resize(img[index_y_0:index_y_d+1,:index_x],(28,28),interpolation=cv2.INTER_CUBIC)
		ret,img = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)
		temp.append(img)
	return temp

def go_noise(image):
	# kernel = np.array([[0.5,0.5],[0.5,0.5]])
	# temp = cv2.erode(image,kernel,1)
	kernel = np.ones((2,2),np.uint8)
	temp = cv2.erode(image,kernel,1)
	# temp = cv2.dilate(image,kernel,1)
	temp = cv2.morphologyEx(temp,cv2.MORPH_OPEN,kernel)
	plt.figure('1') 
	plt.imshow(temp,cmap='gray')  
	return image - temp
	
	
def pre_progress(path):
	image = cv2.imread(path,0)
	h,w = image.shape
	image = image[:int(h/2),:w]
	ret,image = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY)
	image_deout = deout(image)
	image_getin = get_in(image_deout)
	image_getin = go_noise(image_getin)
	img = get_num(image_getin)
	
	if show:
		# plt.subplot(321)
		# plt.imshow(img[0],cmap='gray')	
		# plt.subplot(322)
		# plt.imshow(img[1],cmap='gray')
		# plt.subplot(323)
		# plt.imshow(img[2],cmap='gray')	
		# plt.subplot(324)
		# plt.imshow(img[3],cmap='gray')
		# plt.subplot(325)
		# plt.imshow(img[4],cmap='gray')	
		# plt.subplot(326)
		# plt.imshow(img[5],cmap='gray')

		plt.figure('去除外边框')
		plt.imshow(image_deout,cmap='gray')
		plt.figure('去除内边框')
		plt.imshow(image_getin,cmap='gray')
		plt.show()
	return img
	
if __name__ == '__main__':
	show = True
	path = 'E:\\li\\hjl\\data\\1.jpg'
	pre_progress(path)