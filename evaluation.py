# -*- coding: utf-8 -*-
"""
train Lenet use our dataset
"""
import os
import tensorflow as tf
import numpy as np
import time
from model import net
from utils import datareader

image_path = "E:\\li\\hjl\\data\\eva\\"
txt_path = ""
batch_size = 50
regularizer_rate = 0.0005
learning_rate = 6.19e-3
restore_from = "E:\\li\\hjl\\Lenet-5\\snapshot\\"
num_steps = 300
snapshot_dir = "E:\\li\\hjl\\Lenet-5\\eval\\"

# load model
def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path)) 
    
# save model  
def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')   

def main():
    coord = tf.train.Coordinator()
    
    image_batch, label_batch = datareader(image_path,batch_size)
    label_batch = tf.reshape(label_batch,[batch_size])
    regularizer = tf.contrib.layers.l2_regularizer(regularizer_rate)
    
    output = net(image_batch,True,regularizer)
    
    output = tf.cast(tf.argmax(tf.nn.softmax(output),1),tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(output, label_batch),tf.float32))
    tf.summary.scalar('acc', acc)
    summary_writer = tf.summary.FileWriter(snapshot_dir,graph=tf.get_default_graph())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=400)
    loader = tf.train.Saver(var_list=tf.global_variables())
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    merged = tf.summary.merge_all()
    for step in range(num_steps):
        start_time = time.time()
        ckpt_path = restore_from + "model.ckpt-" + str(step)
        load(loader, sess, ckpt_path)
        
        loss_value, summary = sess.run([acc, merged])
        summary_writer.add_summary(summary, int(step))
        # save(saver, sess, snapshot_dir, int(step/save_pred_every))
        duration = time.time() - start_time
        print('step {:d} \t acc = {:3f}, ({:3f} sec/step)'.format(step, loss_value,duration))
    coord.request_stop()
    coord.join(threads)

main()