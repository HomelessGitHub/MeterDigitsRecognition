# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 21:43:27 2017
some function for run net
@author: liyang
"""
import os
import tensorflow as tf
    
# save model  
def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')   
    
# read data and return batch
def datareader(data_path, batch_size):
    """
    Args:
        name_txt:path of txt(image name and label name)
        data_path:path of data
        batch_size:num of mini batch
    return: image_batch and label_batch
    """
    image_name = []
    label_name = []
    # file = open(name_txt, 'r')
    # lines = file.readlines()
    lines = os.listdir(data_path)
    for line in lines:
        # line = line.strip("\n")
        # name_image, name_label = line.split(" ")
        image_name.append(os.path.join(data_path, line))
        label_name.append(int(line[:1]))
    image_name = tf.convert_to_tensor(image_name, dtype = tf.string)
    label_name = tf.convert_to_tensor(label_name, dtype = tf.int32)
    queue = tf.train.slice_input_producer([image_name, label_name])
    image_contents = tf.read_file(queue[0])
    label_contents = queue[1]
    img = tf.image.decode_png(image_contents, channels = 1)
    img = tf.cast(img,tf.float32)
    img = tf.reshape(img,[28,28,1])
    label_contents = tf.reshape(label_contents,[1])
    image_batch, label_batch = tf.train.batch([img, label_contents], batch_size)
    return image_batch, label_batch
    
# write results of evaluation into txt
def print_result(input_name, output, file_path):
    """
    Args:
        input_name:name of input
        output:result of evaluation dtype=string
        file_path:path of txt
    """
    file = open(file_path, 'a')
    file.write(input_name + ":" + output)
