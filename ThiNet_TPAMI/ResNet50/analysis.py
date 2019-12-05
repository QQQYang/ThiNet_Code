#coding=utf-8
'''
This file is used for analysing the filters and activations of a network, which inspire us of new ideas about network pruning

Author: yqian@aibee.com
'''
# import ptvsd
# ptvsd.enable_attach(address = ('0.0.0.0', 5689))
# ptvsd.wait_for_attach()

import caffe
import numpy as np
from PIL import Image
from net_generator import solver_and_prototxt
import random
import time
import os
import argparse
import json

def im_resize(im, height=224, width=224):
    d_type = im.dtype
    im = Image.fromarray(im)
    im = im.resize([height, width], Image.BICUBIC)
    im = np.array(im, d_type)
    return im


def convert2rgb(im):
    if len(im.shape) == 2:
        im = im.reshape((im.shape[0], im.shape[1], 1))
        im = np.concatenate((im, im, im), axis=2)
    if im.shape[2] == 4:
        im = np.array(Image.fromarray(im).convert('RGB'))
    return im

def collect_activation(selected_layer, selected_block):
    model_def = '/ssd/yqian/prune/model/ResNet50/' + '2a' + '_' + '0' + '/deploy.prototxt'
    model_weights = '/ssd/yqian/prune/model/ResNet50/2a_0/snapshot/_iter_64000.caffemodel.h5'
    # model_def = '/ssd/yqian/prune/model/reid/deploy_tmp.prototxt'
    # model_weights = '/ssd/yqian/prune/model/reid/npair_res50_thinet/res50_baseline.caffemodel'

    # load net
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    mean_value = np.array([128, 128, 128], dtype=np.float32)
    mean_value = mean_value.reshape([3, 1, 1])

    sample_num = 100
    act_mean = {}
    layers = ['2a', '2b', '2c', '3a', '3b', '3c', '3d', '4a', '4b', '4c', '4d', '4e', '4f', '5a', '5b', '5c']
    for foldername in os.listdir('../dataset/cifar-10-batches-py/img'):
        img_list = os.listdir('../dataset/cifar-10-batches-py/img/' + foldername)
        img_index = random.sample(range(len(img_list)), sample_num)
        for file_index in img_index:
            time_start = time.time()

            file_path = '../dataset/cifar-10-batches-py/img/' + foldername + '/' + img_list[file_index]
            im = Image.open(file_path)
            im = convert2rgb(np.array(im))
            im = im_resize(im, 32, 32)
            im = np.array(im, np.float64)
            # im = im[:, :, ::-1]  # convert RGB to BGR
            im = im.transpose((2, 0, 1))  # convert to 3x256x256
            im -= mean_value

            # shape for input (data blob is N x C x H x W), set data
            # center crop
            # im = im[:, 16:240, 16:240]
            net.blobs['data'].reshape(1, *im.shape)
            net.blobs['data'].data[...] = im
            # run net and take argmax for prediction
            net.forward()

            for i in range(len(selected_layer)):
                for j in range(len(selected_block)):
                    if selected_block[j] == 1:
                        output_layer = 'res' + layers[selected_layer[i]] + '_branch2b'
                    else:
                        output_layer = 'res' + layers[selected_layer[i]] + '_branch2c'
                    activation = net.blobs[output_layer].data
                    if output_layer not in act_mean:
                        act_mean[output_layer] = [np.mean(activation, axis=(0, 2, 3)).tolist()]
                    else:
                        act_mean[output_layer].append(np.mean(activation, axis=(0, 2, 3)).tolist())
    with open('act_mean.json','w') as f:
        json.dump(act_mean, f)    

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_layer", type=int, nargs='+', default = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    parser.add_argument("--selected_block", type=int, nargs='+', default = [1,2], help='range from 1 to 2')
    parser.add_argument("--gpu", type=int, default = 4)

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    while True:
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
        memory_max = max(memory_gpu)
        if memory_max>5000:
            gpu = np.argmax(memory_gpu)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu))
            os.system('rm tmp')
            print('Find vacant GPU: %d' % gpu)
            break

    opt = get_opt()
    collect_activation(opt.selected_layer, opt.selected_block)