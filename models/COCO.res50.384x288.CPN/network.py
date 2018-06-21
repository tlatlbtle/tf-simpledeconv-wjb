import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys, os
import argparse
import numpy as np
from functools import partial

from config import cfg
from tfflat.base import ModelDesc, Trainer
from tfflat.utils import mem_info

from nets.basemodel import resnet50, resnet_arg_scope, resnet_v1
resnet_arg_scope = partial(resnet_arg_scope, bn_trainable=cfg.bn_train)

def create_deconv_net(blocks, is_training, trainable=True):
    initializer = tf.contrib.layers.xavier_initializer()
    deconv_times=3
    upsample=blocks
    for i in range(deconv_times):
        upsample = slim.conv2d_transpose(upsample, 256, [4, 4],stride=2,trainable=trainable, 
                                         weights_initializer=initializer,padding='SAME', 
                                         normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu,
                                         scope='upsample/res{}'.format(5-i))
        
    out = slim.conv2d(upsample, cfg.nr_skeleton, [1, 1],
                trainable=trainable, weights_initializer=initializer,
                padding='SAME', activation_fn=None,scope='heatmap_out')
    return out


class Network(ModelDesc):
    def make_data(self):
        from COCOAllJoints import COCOJoints
        from dataset import Preprocessing

        d = COCOJoints()
        train_data, _ = d.load_data(cfg.min_kps)
#         print(train_data)
        from tfflat.data_provider import DataFromList, MultiProcessMapDataZMQ, BatchData, MapData
        dp = DataFromList(train_data)
        if cfg.dpflow_enable:
            dp = MultiProcessMapDataZMQ(dp, cfg.nr_dpflows, Preprocessing)
        else:
            dp = MapData(dp, Preprocessing)
        dp = BatchData(dp, cfg.batch_size // cfg.nr_aug)
        dp.reset_state()
        dataiter = dp.get_data()

        return dataiter

    def make_network(self, is_train):
        if is_train:
            image = tf.placeholder(tf.float32, shape=[cfg.batch_size, *cfg.data_shape, 3])
            label15 = tf.placeholder(tf.float32, shape=[cfg.batch_size, *cfg.output_shape, cfg.nr_skeleton])
            label11 = tf.placeholder(tf.float32, shape=[cfg.batch_size, *cfg.output_shape, cfg.nr_skeleton])
            label9 = tf.placeholder(tf.float32, shape=[cfg.batch_size, *cfg.output_shape, cfg.nr_skeleton])
            label7 = tf.placeholder(tf.float32, shape=[cfg.batch_size, *cfg.output_shape, cfg.nr_skeleton])
            valids = tf.placeholder(tf.float32, shape=[cfg.batch_size, cfg.nr_skeleton])
            labels = [label15, label11, label9, label7]
            # labels.reverse() # The original labels are reversed. For reproduction of our pre-trained model, I'll keep it same.
            self.set_inputs(image, label15, label11, label9, label7, valids)
        else:
            image = tf.placeholder(tf.float32, shape=[None, *cfg.data_shape, 3])
            # labels.reverse() # The original labels are reversed. For reproduction of our pre-trained model, I'll keep it same.
            self.set_inputs(image)

        resnet_fms = resnet50(image, is_train, bn_trainable=True)
        out = create_deconv_net(resnet_fms[3], is_train)
        def ohkm(loss, top_k):
            ohkm_loss = 0.
            for i in range(cfg.batch_size):
                sub_loss = loss[i]
                topk_val, topk_idx = tf.nn.top_k(sub_loss, k=top_k, sorted=False, name='ohkm{}'.format(i))
                tmp_loss = tf.gather(sub_loss, topk_idx, name='ohkm_loss{}'.format(i)) # can be ignore ???
                ohkm_loss += tf.reduce_sum(tmp_loss) / top_k
            ohkm_loss /= cfg.batch_size
            return ohkm_loss
        # make loss
        if is_train:
#             print(out.shape,label7.shape)(24, 96, 72, 17)
            total_loss = tf.reduce_mean(tf.square(out - label7), (1,2)) * tf.to_float((tf.greater(valids, 0.1)))
#             print(total_loss.shape)
            total_loss_value=tf.reduce_sum(total_loss) / cfg.batch_size
            self.add_tower_summary('loss', total_loss_value)
            self.set_loss(total_loss_value)
        else:
            self.set_outputs(out)

if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu', '-d', type=str, dest='gpu_ids')
        parser.add_argument('--continue', '-c', dest='continue_train', action='store_true')
        parser.add_argument('--debug', dest='debug', action='store_true')
        args = parser.parse_args()

        if not args.gpu_ids:
            args.gpu_ids = str(np.argmin(mem_info()))

        if '-' in args.gpu_ids:
            gpus = args.gpu_ids.split('-')
            gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
            gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
            args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

        return args
    args = parse_args()

    cfg.set_args(args.gpu_ids, args.continue_train)
    trainer = Trainer(Network(), cfg)
    trainer.train()

