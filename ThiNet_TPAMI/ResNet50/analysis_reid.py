#coding=utf-8
'''
This file is used for analysing the filters and activations of a network, which inspire us of new ideas about network pruning

Author: yqian@aibee.com
'''
# import ptvsd
# ptvsd.enable_attach(address = ('0.0.0.0', 5678))
# ptvsd.wait_for_attach()

import caffe
import numpy as np
from PIL import Image
import cv2
from net_generator import solver_and_prototxt
import random
import time
import os
import argparse
import json

def cal_corrcoef(act):
    act_sum = np.sum(act)
    act = np.sort(act)[::-1]
    y = [sum(act[:i+1])/act_sum for i in range(len(act))]
    x = [float(i+1)/len(act) for i in range(len(act))]
    coef = np.corrcoef(np.array([x, y]))
    return coef[0, 1]

def resize_image_with_padding(im, new_dims, interp_order=1):
    """
    Resize an image array with interpolation.

    Parameters
    ----------
    im : (H x W x K) ndarray
    new_dims : (height, width) tuple of new dimensions.
    interp_order : interpolation order, default is linear.

    Returns
    -------
    im : resized ndarray with shape (new_dims[0], new_dims[1], K)
    """
    ret = np.empty((new_dims[0], new_dims[1], im.shape[-1]),
                   dtype=np.float32)
    ret.fill(0)
    target_as = new_dims[1]  / float(new_dims[0])
    aspect_ratio = im.shape[1] / float(im.shape[0])
    if target_as < aspect_ratio:
        scale = new_dims[1] / float(im.shape[1])
        scaled_width = int(new_dims[1])
        scaled_height = min(int(new_dims[0]), int(scale* im.shape[0]))
        resized_img = cv2.resize(im, (scaled_width, scaled_height))
        start_x = 0
        start_y = 0
        ret[start_x: start_x  + scaled_height, start_y: start_y + scaled_width, :] = resized_img
    else:
        scale = new_dims[0] / float(im.shape[0])
        scaled_width = min(int(new_dims[1]), int(scale* im.shape[1]))
        scaled_height = int(new_dims[0])
        resized_img = cv2.resize(im, (scaled_width, scaled_height))
        start_x = 0
        start_y = int((new_dims[1] - scaled_width) / 2)
        ret[start_x: start_x  + scaled_height, start_y: start_y + scaled_width, :] = resized_img
    return ret.astype(np.float32)

def collect_activation(selected_layer, selected_block):
    model_def = '/ssd/yqian/prune/model/reid/deploy_baseline.prototxt'
    model_weights = '/ssd/yqian/prune/model/body_reid_general_npair_caffe_cpu_ctf_20190925_v010002/npair_may_to_aug_ctf_all_stores_finetune_full_year_iter_44000.caffemodel'

    # load net
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    mean_value = np.array([104, 117, 123], dtype=float)

    sample_num = 2000
    act_mean = {}
    layers = ['2a', '2b', '2c', '3a', '3b', '3c', '3d', '4a', '4b', '4c', '4d', '4e', '4f', '5a', '5b', '5c']
    data_list = np.loadtxt('/ssd/yqian/prune/dataset/data/test_data/eval_CTF_beijing_xhm_20181207_label_finish_revision.txt', dtype=str)
    img_index = random.sample(range(len(data_list)), sample_num)
    # f = open('/ssd/yqian/prune/dataset/data/train_all_new.txt')
    for file_index in img_index:
        # offset = random.randrange(2e7)
        # f.seek(offset, 0)
        # line = f.readline()
        # time_start = time.time()
        # while len(line) < 2:
        #     offset = random.randrange(2e7)
        #     f.seek(offset, 0)
        #     line = f.readline()
        # try:
        #     file_path = '/ssd/yqian/prune/dataset/data/' + line.split()[0]
        # except IndexError:
        #     print('error: ', len(line))
        # im = cv2.imread(file_path)
        # while im is None:
        #     offset = random.randrange(2e7)
        #     f.seek(offset, 0)
        #     line = f.readline()
        #     while len(line) < 2:
        #         offset = random.randrange(2e7)
        #         f.seek(offset, 0)
        #         line = f.readline()
        #     try:
        #         file_path = '/ssd/yqian/prune/dataset/data/' + line.split()[0]
        #     except IndexError:
        #         print('error: ', len(line))
        #     im = cv2.imread(file_path)
        # print(line.split()[0])
        file_path = '/ssd/yqian/prune/dataset/data/test_data/all/' + data_list[file_index][0]
        im = cv2.imread(file_path)
        im = resize_image_with_padding(im, (384, 128))
        im -= mean_value
        im = np.transpose(im, (2,0,1)) # HWC -> CHW
        im = np.reshape(im, (1, 3, 384, 128)) #CHW ->NCHW

        # shape for input (data blob is N x C x H x W), set data
        # center crop
        # im = im[:, 16:240, 16:240]
        net.blobs['data'].reshape(*im.shape)
        net.blobs['data'].data[...] = im
        # run net and take argmax for prediction
        net.forward()

        for i in range(len(selected_layer)):
            for j in range(len(selected_block)):
                if selected_block[j] == 1:
                    output_layer = 'res' + layers[selected_layer[i]] + '_branch2a'
                else:
                    output_layer = 'res' + layers[selected_layer[i]] + '_branch2b'
                activation = net.blobs[output_layer].data
                if output_layer not in act_mean:
                    act_mean[output_layer] = [np.mean(activation, axis=(0, 2, 3)).tolist()]
                else:
                    act_mean[output_layer].append(np.mean(activation, axis=(0, 2, 3)).tolist())
    for key in act_mean:
        layer_act = act_mean[key]
        act_mean[key] = np.sum(np.abs(np.array(layer_act)), axis=0).tolist()
        act_mean[key] = float(cal_corrcoef(act_mean[key]))
    print(act_mean)
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