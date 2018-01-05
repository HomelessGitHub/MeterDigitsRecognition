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

image_path = "E:\\li\\hjl\\data\\image\\"
txt_path = ""
batch_size = 10
regularizer_rate = 0.0005
learning_rate = 6.19e-4
restore_from = None
num_steps = 530*30
save_pred_every = 53
snapshot_dir = "E:\\li\\hjl\\Lenet-5\\snapshot\\"

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
    label_batch = tf.reshape(label_batch,[10])
    regularizer = tf.contrib.layers.l2_regularizer(regularizer_rate)
    
    output = net(image_batch,True,regularizer)
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batch, logits=output)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    tf.summary.scalar('loss', loss)
    summary_writer = tf.summary.FileWriter(snapshot_dir,graph=tf.get_default_graph())
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss,aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=400)
    
    # Load variables if the checkpoint is provided.
    if restore_from is not None:
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, restore_from)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    merged = tf.summary.merge_all()
    for step in range(num_steps):
        start_time = time.time()
        loss_value = 0
        if step % save_pred_every == 0:
            loss_value, summary, _ = sess.run([loss, merged, train_op])
            summary_writer.add_summary(summary, int(step/save_pred_every))
            save(saver, sess, snapshot_dir, int(step/save_pred_every))
        else:
            loss_value, _ = sess.run([loss, train_op])
        duration = time.time() - start_time
        print('step {:d} iter {:d} \t loss = {:3f}, ({:3f} sec/step)'.format(int(step / save_pred_every),
                                                                               int(step % save_pred_every), loss_value,
                                                                               duration))
    coord.request_stop()
    coord.join(threads)

main()