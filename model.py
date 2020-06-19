#coding=utf-8
from __future__ import print_function

import os
import time
# import random

from PIL import Image
import tensorflow as tf
import numpy as np
import numpy.random as random
from utils import *
from scipy.ndimage import maximum_filter
def concat(layers):
    return tf.concat(layers, axis=3)

def DecomNet(input_im, layer_num, channel=64, kernel_size=3,is_training=True):
    input_max = tf.reduce_max(input_im, axis=3, keepdims=True)
    input_im = concat([input_max, input_im])
    with tf.variable_scope('DecomNet', reuse=tf.AUTO_REUSE):
        conv_0 = tf.layers.conv2d(input_im, channel/2, kernel_size, padding='same', activation=tf.nn.relu, name="first_layer")

        conv = tf.layers.conv2d(input_im, channel, kernel_size * 3, padding='same', activation=None, name="shallow_feature_extraction")

        conv1 = tf.layers.conv2d(conv, channel, kernel_size, padding='same', activation=tf.nn.relu, name='activated_layer_1')
        # conv1_ba = tf.layers.batch_normalization(conv1,training=is_training)

        conv2 = tf.layers.conv2d(conv1, channel*2, kernel_size,strides=2, padding='same', activation=tf.nn.relu, name='activated_layer_2')
        # conv2_ba = tf.layers.batch_normalization(conv2,training=is_training)

        conv3 = tf.layers.conv2d(conv2, channel*2, kernel_size, padding='same', activation=tf.nn.relu, name='activated_layer_3')
        # conv3_ba = tf.layers.batch_normalization(conv3,training=is_training)

        conv4 = tf.layers.conv2d_transpose(conv3, channel, kernel_size,strides=2, padding='same', activation=tf.nn.relu, name='activated_layer_4')
        # conv4_ba = tf.layers.batch_normalization(conv4,training=is_training)
        conv4_ba2=concat([conv4,conv1])

        conv5 = tf.layers.conv2d(conv4_ba2, channel, kernel_size,padding='same', activation=tf.nn.relu, name='activated_layer_5')

        conv6=concat([conv5,conv_0])
        # conv7=concat([conv6,conv])

        conv7 = tf.layers.conv2d(conv6, channel, kernel_size,padding='same', activation=None, name='activated_layer_7')

        conv8 = tf.layers.conv2d(conv7, 4, kernel_size, padding='same', activation=None, name='recon_layer')

        #conv9 = tf.layers.conv2d(conv7, 3, kernel_size,padding='same', activation=None, name='noise_layer_9') # pre noise # in futrue
 
    R = tf.sigmoid(conv8[:,:,:,0:3])
    L = tf.sigmoid(conv8[:,:,:,3:4]) # L can be higher than 1,lower can promise image enhancement in low light area


    return R, L


class lowlight_enhance(object):
    def __init__(self, sess):
        self.sess = sess
        self.DecomNet_layer_num = 5

        # build the model
        self.input_low = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low')
        self.input_high = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high')

        self.input_low_eq = tf.placeholder(tf.float32, [None, None, None, 1], name='input_low_eq')

        self.input_low_eq_guide = tf.placeholder(tf.float32, [None, None, None, 1], name='input_low_eq_guide') # tried weighted loss, 
        self.input_low_eq_guide_weight = tf.placeholder(tf.float32, [None, None, None, 1], name='input_low_eq_guide_weight')

        [R_low, I_low] = DecomNet(self.input_low, layer_num=self.DecomNet_layer_num)

        I_low_3 = concat([I_low, I_low, I_low])

        self.output_R_low = R_low # R
        self.output_I_low = I_low_3 # I
        self.output_S_low_zy = (R_low * I_low_3) # check training


        # loss

        self.recon_loss_low = tf.reduce_mean(tf.abs(R_low * I_low_3  -  self.input_high)) #keep color and denoise

        R_low_max = tf.reduce_max(R_low, axis=3, keepdims=True)
        self.recon_loss_low_eq = tf.reduce_mean(tf.abs(R_low_max -  self.input_low_eq))# improve contrast and light

        self.R_low_loss_smooth= tf.reduce_mean(tf.abs(self.gradient(tf.image.rgb_to_grayscale(R_low), "x"))+tf.abs(self.gradient(tf.image.rgb_to_grayscale(R_low), "y"))) # denoise

        self.Ismooth_loss_low = self.smooth(I_low, R_low)# smooth


        self.loss_Decom_zhangyu= self.recon_loss_low + 0.1 * self.Ismooth_loss_low + 0.1 * self.recon_loss_low_eq + 0.01*self.R_low_loss_smooth#+ 0.0*self.N_low_loss


        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')

        self.var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]

        self.train_op_Decom = optimizer.minimize(self.loss_Decom_zhangyu, var_list = self.var_Decom)

        self.sess.run(tf.global_variables_initializer())

        self.saver_Decom = tf.train.Saver(var_list = self.var_Decom)

        print("[*] Initialize model successfully...")

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
        self.smooth_kernel_y = tf.transpose(self.smooth_kernel_x, [1, 0, 2, 3])

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        return tf.abs(tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))

    def ave_gradient(self, input_tensor, direction):
        return tf.layers.average_pooling2d(self.gradient(input_tensor, direction), pool_size=3, strides=1, padding='SAME')

    def smooth(self, input_I, input_R):
        input_R = tf.image.rgb_to_grayscale(input_R)
        # another_=tf.reduce_mean(self.gradient(input_I, "x")/tf.maximum(self.gradient(input_I, "x"),0.01)+self.gradient(input_I, "y")/tf.maximum(self.gradient(input_I, "y"),0.01))
        return tf.reduce_mean(self.gradient(input_I, "x") * tf.exp(-10 * self.gradient(input_R, "x")) + self.gradient(input_I, "y") * tf.exp(-10 * self.gradient(input_R, "y")))#+another_

    def evaluate(self, epoch_num, eval_low_data, sample_dir, train_phase):
        print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))

        for idx in range(len(eval_low_data)):
            input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)

            if train_phase == "Decom":
                result_1, result_2 = self.sess.run([self.output_R_low, self.output_I_low], feed_dict={self.input_low: input_low_eval})

            save_images(os.path.join(sample_dir, 'eval_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_1, result_2)

    def train(self, train_low_data, train_low_data_eq,eval_low_data,train_high_data, batch_size, patch_size, epoch, lr, sample_dir, ckpt_dir, eval_every_epoch, train_phase):
        # assert len(train_low_data) == len(train_high_data)
        numBatch = len(train_low_data) // int(batch_size)

        # load pretrained model
        if train_phase == "Decom":
            train_op = self.train_op_Decom
            train_loss = self.loss_Decom_zhangyu
            saver = self.saver_Decom

        load_model_status, global_step = self.load(saver, ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")

        print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id = 0
        boolflag = True
        loss = 0
        for epoch in range(start_epoch, epoch):
            boolflag = True
            for batch_id in range(start_step, numBatch):
                # generate data for a batch
                batch_input_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                batch_input_low_eq = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")


                for patch_id in range(batch_size):
                    h, w, _ = train_low_data[image_id].shape
                    x = np.random.randint(0, h - patch_size)
                    y = np.random.randint(0, w - patch_size)

                    rand_mode = np.random.randint(0, 7)

                    batch_input_low[patch_id, :, :, :] = data_augmentation(train_low_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    batch_input_high[patch_id, :, :, :] = data_augmentation(train_high_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    batch_input_low_eq[patch_id, :, :, :] = data_augmentation(train_low_data_eq[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)



                    image_id = (image_id + 1) % len(train_low_data)
                    if image_id == 0:
                        tmp = list(zip(train_low_data, train_low_data))
                        np.random.shuffle(list(tmp))
                        train_low_data, _  = zip(*tmp)

                # train
                if not boolflag:
                    _ = self.sess.run([train_op], feed_dict={self.input_low: batch_input_low, \
                                                                               self.input_high: batch_input_high, \
                                                                               self.input_low_eq: batch_input_low_eq, \
                                                                               self.lr: lr[epoch]})
                else:
                    boolflag=False
                    _, loss = self.sess.run([train_op,train_loss], feed_dict={self.input_low: batch_input_low, \
                                                                               self.input_high: batch_input_high, \
                                                                               self.input_low_eq: batch_input_low_eq, \
                                                                               self.lr: lr[epoch]})

                print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                      % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1

            # evalutate the model and save a checkpoint file for it
            start_step = 0
            if (epoch + 1) % eval_every_epoch == 0:
                self.evaluate(epoch + 1, eval_low_data, sample_dir=sample_dir, train_phase=train_phase)
                self.save(saver, iter_num, ckpt_dir, "RetinexNet-%s" % train_phase)

        print("[*] Finish training for phase %s." % train_phase)

    def save(self, saver, iter_num, ckpt_dir, model_name):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        print("[*] Saving model %s" % model_name)
        saver.save(self.sess, \
                   os.path.join(ckpt_dir, model_name), \
                   global_step=iter_num)

    def load(self, saver, ckpt_dir):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(ckpt_dir)
            try:
                global_step = int(full_path.split('/')[-1].split('-')[-1])
            except ValueError:
                global_step = None
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            print("[*] Failed to load model from %s" % ckpt_dir)
            return False, 0

    def test(self, test_low_data, test_high_data, test_low_data_names, save_dir, decom_flag):
        tf.global_variables_initializer().run()

        print("[*] Reading checkpoint...")
        load_model_status_Decom, _ = self.load(self.saver_Decom, './checkpoint/Decom')
        if load_model_status_Decom:
            print("[*] Load weights successfully...")
        
        print("[*] Testing...")
        total_run_time = 0.0
        for idx in range(len(test_low_data)):
            print(test_low_data_names[idx])
            [_, name] = os.path.split(test_low_data_names[idx])
            suffix = name[name.find('.') + 1:]
            name = name[:name.find('.')]

            input_low_test = np.expand_dims(test_low_data[idx], axis=0)

            start_time = time.time()
            R_low, I_low,output_S_low_zy = self.sess.run([self.output_R_low, self.output_I_low,self.output_S_low_zy], feed_dict={self.input_low: input_low_test})

            if(idx!=0):
                total_run_time += time.time() - start_time
            if decom_flag == decom_flag:
                save_images(os.path.join(save_dir, name + "_R_low." + suffix), R_low)
                save_images(os.path.join(save_dir, name + "_I_low." + suffix), I_low)

        ave_run_time = total_run_time / (float(len(test_low_data))-1)
        print("[*] Average run time: %.4f" % ave_run_time)
