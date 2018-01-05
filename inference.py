# -*- coding: utf-8 -*-
"""
train Lenet use our dataset
"""
import os
import tensorflow as tf
import numpy as np
import time
from model import net
from utils import datareader,save

restore_from = "E:/li/hjl/Lenet-5/model/model.ckpt"

def load(sess, saver, ckpt_path):
    saver.restore(sess, ckpt_path)

def inference(data):
    result = ''
    image = tf.convert_to_tensor(np.array(data),tf.float32)
    image = tf.reshape(image,[6,28,28,1])
    outp = net(image,True,None)
    output = tf.nn.softmax(outp)

    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=100)
    
    # Load variables if the checkpoint is provided.
    if restore_from is not None:
        load(sess,saver, restore_from)

    start_time = time.time()
    out = sess.run(output)
    print(out)
    duration = time.time() - start_time
    for i in range(6):
        temp = out[i].tolist()
        result += str(temp.index(max(temp)))
    sess.close()
    return result