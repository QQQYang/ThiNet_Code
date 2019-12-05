# coding=utf-8
# import ptvsd
# ptvsd.enable_attach(address = ('0.0.0.0', 5679))
# ptvsd.wait_for_attach()
import sys
# caffe_root = '/home/luojh2/Software/caffe-master/'
# sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
from PIL import Image
from net_generator import solver_and_prototxt
import random
import time
import os
import argparse


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


def get_index(compress_rate, compress_layer, gpu, compress_block, method):
    # fix
    if len(compress_block) > 1:
        model_dir = '-'.join([str(i) for i in compress_layer])+'_'+str(compress_block[0])+'_'+str(compress_rate[0])
        model_def = os.path.join('/ssd/yqian/prune/model/ResNet50', model_dir, 'deploy.prototxt')
        model_weights = os.path.join('/ssd/yqian/prune/model/ResNet50', model_dir, 'snapshot/_iter_10000.caffemodel.h5')
        compress_block = [compress_block[-1]]
        compress_rate = [compress_rate[-1]]
    else:
        model_def = '/ssd/yqian/prune/model/ResNet50/' + '2a' + '_' + '0' + '/deploy.prototxt'
        model_weights = '/ssd/yqian/prune/model/ResNet50/2a_0/snapshot/_iter_64000.caffemodel.h5'
    # set parameters
    sample_num = 10  # 1000 categories sample sample_num images
    channel_num = 10  # sample channel number

    layers = ['2a', '2b', '2c', '3a', '3b', '3c', '3d', '4a', '4b', '4c', '4d', '4e', '4f', '5a', '5b', '5c']

    # load net
    caffe.set_device(0)
    caffe.set_mode_gpu()
    # model_def = '/ssd/yqian/prune/model/ResNet50/'+'2a' + '_' + '0' +'/deploy.prototxt'
    # model_weights = '/ssd/yqian/prune/model/ResNet50/2a_0/snapshot/_iter_64000.caffemodel.h5'
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    mean_value = np.array([128, 128, 128], dtype=np.float32)
    mean_value = mean_value.reshape([3, 1, 1])

    # if compress_block == 1:
    #     input_layer = 'bn' + layers[compress_layer] + '_branch2a'
    #     output_layer = 'res' + layers[compress_layer] + '_branch2b'
    #     kernel_size = 3
    #     padding = 1
    # else:
    #     input_layer = 'res' + layers[compress_layer]
    #     output_layer = 'res' + layers[compress_layer] + '_branch2a'
    #     kernel_size = 1
    #     padding = 0

    # extract feature
    Xs, Ys = [], []
    count = []
    if method == 'single':
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

                for i in range(len(compress_layer)):
                    for j in range(len(compress_block)):
                        if compress_block[j] == 1:
                            input_layer = 'bn' + layers[compress_layer[i]] + '_branch2a'
                            output_layer = 'res' + layers[compress_layer[i]] + '_branch2b'
                            kernel_size = 3
                            padding = 1
                        else:
                            input_layer = 'bn' + layers[compress_layer[i]] + '_branch2b'
                            output_layer = 'res' + layers[compress_layer[i]] + '_branch2c'
                            kernel_size = 1
                            padding = 0

                        Activation = net.blobs[output_layer].data  # res2a_branch2c
                        Input = net.blobs[input_layer].data  # bn2a_branch2b
                        Filters = net.params[output_layer][0].data  # res2a_branch2c

                        if padding > 0:
                            padded = np.zeros((Input.shape[0], Input.shape[1], Input.shape[
                                2] + 2 * padding, Input.shape[3] + 2 * padding), dtype=np.float32)
                            padded[:, :, padding:-padding, padding:-padding] = Input
                            Input = padded

                        # if count == 0:
                        #     X = np.zeros([channel_num * 1000 * sample_num, Filters.shape[1]])
                        #     Y = np.zeros([channel_num * 1000 * sample_num, 1])
                        if len(Xs) < i+1:
                            X = np.zeros([channel_num * 1000 * sample_num, Filters.shape[1]])
                            Y = np.zeros([channel_num * 1000 * sample_num, 1])
                            Xs.append([X])
                            Ys.append([Y])
                            count.append([0])
                        elif len(Xs[i]) < j+1:
                            X = np.zeros([channel_num * 1000 * sample_num, Filters.shape[1]])
                            Y = np.zeros([channel_num * 1000 * sample_num, 1])
                            Xs[i].append(X)
                            Ys[i].append(Y)
                            count[i].append(0)

                        for tmp in range(channel_num):
                            filter_num = random.randint(0, Filters.shape[0] - 1)
                            r = random.randint(0, Input.shape[2] - kernel_size)
                            c = random.randint(0, Input.shape[3] - kernel_size)
                            In_ = Input[:, :, r:r + kernel_size, c:c + kernel_size]
                            In_ = In_.reshape([In_.shape[1], -1])
                            F_ = Filters[filter_num, :, :, :]
                            F_ = F_.reshape([F_.shape[0], -1])
                            Out_ = Activation[0, filter_num, r, c]
                            # X[count, :] = np.reshape(np.sum(F_ * In_, axis=1), [1, -1])
                            # Y[count, 0] = np.reshape(Out_, [1, -1])
                            Xs[i][j][count[i][j], :] = np.reshape(np.sum(F_ * In_, axis=1), [1, -1])
                            Ys[i][j][count[i][j], 0] = np.reshape(Out_, [1, -1])
                            count[i][j] = count[i][j] + 1

                time_end = time.time()
                # print 'Done! use %f second, %d image' % (time_end - time_start, count / channel_num)
        # sort index
        index, w = [], []
        for i in range(len(compress_layer)):
            layer_index, layer_w = [], []
            for j in range(len(compress_block)):
                single_index, single_w = value_sum(Xs[i][j], Ys[i][j], compress_rate[j])
                layer_index.append(single_index)
                layer_w.append(single_w)
            index.append(layer_index)
            w.append(layer_w)
        return index, w
    else:
        for j in range(len(compress_block)):
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

                    for i in range(len(compress_layer)):
                        if compress_block[j] == 1:
                            input_layer = 'bn' + layers[compress_layer[i]] + '_branch2a'
                            output_layer = 'res' + layers[compress_layer[i]] + '_branch2b'
                            kernel_size = 3
                            padding = 1
                        else:
                            input_layer = 'bn' + layers[compress_layer[i]] + '_branch2b'
                            output_layer = 'res' + layers[compress_layer[i]] + '_branch2c'
                            kernel_size = 1
                            padding = 0

                        Activation = net.blobs[output_layer].data  # res2a_branch2c
                        Input = net.blobs[input_layer].data  # bn2a_branch2b
                        Filters = net.params[output_layer][0].data  # res2a_branch2c

                        if padding > 0:
                            padded = np.zeros((Input.shape[0], Input.shape[1], Input.shape[
                                2] + 2 * padding, Input.shape[3] + 2 * padding), dtype=np.float32)
                            padded[:, :, padding:-padding, padding:-padding] = Input
                            Input = padded

                        # if count == 0:
                        #     X = np.zeros([channel_num * 1000 * sample_num, Filters.shape[1]])
                        #     Y = np.zeros([channel_num * 1000 * sample_num, 1])
                        if len(Xs) < i+1:
                            X = np.zeros([channel_num * 1000 * sample_num, Filters.shape[1]])
                            Y = np.zeros([channel_num * 1000 * sample_num, 1])
                            Xs.append([X])
                            Ys.append([Y])
                            count.append([0])
                        elif len(Xs[i]) < j+1:
                            X = np.zeros([channel_num * 1000 * sample_num, Filters.shape[1]])
                            Y = np.zeros([channel_num * 1000 * sample_num, 1])
                            Xs[i].append(X)
                            Ys[i].append(Y)
                            count[i].append(0)

                        for tmp in range(channel_num):
                            filter_num = random.randint(0, Filters.shape[0] - 1)
                            r = random.randint(0, Input.shape[2] - kernel_size)
                            c = random.randint(0, Input.shape[3] - kernel_size)
                            In_ = Input[:, :, r:r + kernel_size, c:c + kernel_size]
                            In_ = In_.reshape([In_.shape[1], -1])
                            F_ = Filters[filter_num, :, :, :]
                            F_ = F_.reshape([F_.shape[0], -1])
                            Out_ = Activation[0, filter_num, r, c]
                            # X[count, :] = np.reshape(np.sum(F_ * In_, axis=1), [1, -1])
                            # Y[count, 0] = np.reshape(Out_, [1, -1])
                            Xs[i][j][count[i][j], :] = np.reshape(np.sum(F_ * In_, axis=1), [1, -1])
                            Ys[i][j][count[i][j], 0] = np.reshape(Out_, [1, -1])
                            count[i][j] = count[i][j] + 1

                    time_end = time.time()
                    # print 'Done! use %f second, %d image' % (time_end - time_start, count / channel_num)
            # sort index
            index, w = [], []
            for i in range(len(compress_layer)):
                layer_index, layer_w = [], []
                single_index, single_w = value_sum(Xs[i][j], Ys[i][j], compress_rate[j])
                layer_index.append(single_index)
                layer_w.append(single_w)
                index.append(layer_index)
                w.append(layer_w)
            return index, w


# use greedy method to select index
# x:N*64 matrix, N is the instance number, 64 is channel number
def value_sum(x, y, compress_rate):
    # 1. set parameters
    x = np.mat(x)
    y = np.mat(y)
    goal_num = int(x.shape[1] * compress_rate)
    index = []

    # 2. select
    y_tmp = y

    ## cluster
    # ws = []
    # for j in range(x.shape[1]):
    #     tmp_w = (x[:, j].T*y_tmp)[0, 0]/(x[:,j].T*x[:,j])[0,0]
    #     tmp_value = np.linalg.norm(y_tmp-tmp_w*x[:,j])
    #     ws.append(tmp_value)
    # np.savetxt('error.txt', ws)

    ## greedy
    for i in range(goal_num):
        min_value = float("inf")
        s = time.time()
        for j in range(x.shape[1]):
            if j not in index:
                tmp_w = (x[:, j].T*y_tmp)[0, 0]/((x[:,j].T*x[:,j])[0,0]+1e-4)
                tmp_value = np.linalg.norm(y_tmp-tmp_w*x[:,j])
                if tmp_value < min_value:
                    min_value = tmp_value
                    min_index = j
        index.append(min_index)
        selected_x = x[:, index]
        w = np.linalg.pinv(selected_x.T * selected_x) * selected_x.T * y
        y_tmp = y - selected_x * w
        print('goal num={0}, channel num={1}, i={2}, loss={3:.3f}, time={4:.3f}'.format(goal_num, x.shape[1], i,
                                                                                        min_value, time.time() - s))

    # 3. return index
    index = np.array(list(index))
    index = np.sort(index)

    # 4.least square
    selected_x = x[:, index]
    w = (selected_x.T * selected_x + 1e-5*np.mat(np.identity(selected_x.shape[-1], dtype=np.float))).I * (selected_x.T * y)
    w = np.array(w)

    loss = np.linalg.norm(y - selected_x * w)
    print('loss with w={0:.3f}'.format(loss))
    return index, w


def copy_net(net_new, net, layer_name):
    net_new.params[
        'res' + layer_name][0].data[...] = net.params['res' + layer_name][0].data
    net_new.params[
        'bn' + layer_name][0].data[...] = net.params['bn' + layer_name][0].data
    net_new.params[
        'bn' + layer_name][1].data[...] = net.params['bn' + layer_name][1].data
    net_new.params['bn' + layer_name][2] = net.params['bn' + layer_name][2]
    net_new.params[
        'scale' + layer_name][0].data[...] = net.params['scale' + layer_name][0].data
    net_new.params[
        'scale' + layer_name][1].data[...] = net.params['scale' + layer_name][1].data
    return net_new


def compress_2c(net_new, net, layer_name, index, w):
    weight = net.params['res' + layer_name][0].data
    weight = weight[:, index, :, :]
    # important !!!!!!!!!!!!
    for i in range(weight.shape[1]):
        weight[:, i, :, :] *= w[i]
    #
    net_new.params['res' + layer_name][0].data[...] = weight

    weight = net.params['bn' + layer_name][0].data
    bias = net.params['bn' + layer_name][1].data
    net_new.params['bn' + layer_name][0].data[...] = weight
    net_new.params['bn' + layer_name][1].data[...] = bias
    net_new.params['bn' + layer_name][2] = net.params['bn' + layer_name][2]

    weight = net.params['scale' + layer_name][0].data
    bias = net.params['scale' + layer_name][1].data
    net_new.params['scale' + layer_name][0].data[...] = weight
    net_new.params['scale' + layer_name][1].data[...] = bias
    return net_new


def compress_2b(net_new, net, layer_name, index):
    weight = net.params['res' + layer_name][0].data
    net_new.params['res' + layer_name][0].data[...] = weight[index, :, :, :]

    weight = net.params['bn' + layer_name][0].data
    bias = net.params['bn' + layer_name][1].data
    net_new.params['bn' + layer_name][0].data[...] = weight[index]
    net_new.params['bn' + layer_name][1].data[...] = bias[index]
    net_new.params['bn' + layer_name][2] = net.params['bn' + layer_name][2]

    weight = net.params['scale' + layer_name][0].data
    bias = net.params['scale' + layer_name][1].data
    net_new.params['scale' + layer_name][0].data[...] = weight[index]
    net_new.params['scale' + layer_name][1].data[...] = bias[index]
    return net_new


def compress_net(index, w, compress_layer, compress_block, compress_rate):
    # fix
    new_compress_block = list(compress_block)
    net_new = caffe.Net('/ssd/yqian/prune/model/ResNet50/' + '-'.join([str(i) for i in compress_layer])+'_'+'-'.join([str(i) for i in compress_block])+'_'+'-'.join([str(i) for i in compress_rate])+'/'+'trainval.prototxt', caffe.TEST)
    if len(compress_block) > 1:
        model_dir = '-'.join([str(i) for i in compress_layer])+'_'+str(compress_block[0])+'_'+str(compress_rate[0])
        model_def = os.path.join('/ssd/yqian/prune/model/ResNet50', model_dir, 'deploy.prototxt')
        model_weights = os.path.join('/ssd/yqian/prune/model/ResNet50', model_dir, 'snapshot/_iter_10000.caffemodel.h5')
        compress_block = [compress_block[-1]]
    else:
        model_def = '/ssd/yqian/prune/model/ResNet50/' + '2a' + '_' + '0' + '/deploy.prototxt'
        model_weights = '/ssd/yqian/prune/model/ResNet50/2a_0/snapshot/_iter_64000.caffemodel.h5'
    # other layers
    layers = ['2a', '2b', '2c', '3a', '3b', '3c', '3d',
              '4a', '4b', '4c', '4d', '4e', '4f', '5a', '5b', '5c']

    # model_def = '/ssd/yqian/prune/model/ResNet50/' + '2a' + '_' + '0' + '/deploy.prototxt'
    # model_weights = '/ssd/yqian/prune/model/ResNet50/2a_0/snapshot/_iter_64000.caffemodel.h5'    
    net = caffe.Net(model_def,  # defines the structure of the matrix
                    model_weights,  # contains the trained weights
                    caffe.TEST)  # use test mode (e.g., don't perform dropout)

    # use test mode (e.g., don't perform dropout)

    # layer 0
    layer_name = 'conv1'
    net_new.params[layer_name][0].data[...] = net.params[layer_name][0].data
    net_new.params[layer_name][1].data[...] = net.params[layer_name][1].data
    net_new.params[
        'bn_' + layer_name][0].data[...] = net.params['bn_' + layer_name][0].data
    net_new.params[
        'bn_' + layer_name][1].data[...] = net.params['bn_' + layer_name][1].data
    net_new.params['bn_' + layer_name][2] = net.params['bn_' + layer_name][2]
    net_new.params[
        'scale_' + layer_name][0].data[...] = net.params['scale_' + layer_name][0].data
    net_new.params[
        'scale_' + layer_name][1].data[...] = net.params['scale_' + layer_name][1].data

    # media
    # compress_block += 1
    for i in range(0, 16):
        layer_name = layers[i]
        if i not in compress_layer:
            # copy block
            net_new = copy_net(net_new, net, layer_name + '_branch2a')
            net_new = copy_net(net_new, net, layer_name + '_branch2b')
            net_new = copy_net(net_new, net, layer_name + '_branch2c')
    for j in range(len(compress_layer)):
        layer_name = layers[compress_layer[j]]
        for k in range(len(compress_block)):
            if compress_block[k] == 1:  # important
                net_new = compress_2b(net_new, net, layer_name + '_branch2a', index[j][k])
                net_new = compress_2c(net_new, net, layer_name + '_branch2b', index[j][k], w[j][k])
                net_new = copy_net(net_new, net, layer_name + '_branch2c')
            else:
                net_new = copy_net(net_new, net, layer_name + '_branch2a')
                net_new = compress_2b(net_new, net, layer_name + '_branch2b', index[j][k])
                net_new = compress_2c(net_new, net, layer_name + '_branch2c', index[j][k], w[j][k])

    # branch 1
    candidate = ['2a', '3a', '4a', '5a']
    for i in range(0, 4):
        layer_name = candidate[i]
        net_new = copy_net(net_new, net, layer_name + '_branch1')

    # fc
    net_new.params['fc1000'][0].data[...] = net.params['fc1000'][0].data
    net_new.params['fc1000'][1].data[...] = net.params['fc1000'][1].data
    net_new_name = '/ssd/yqian/prune/model/ResNet50/'+'-'.join([str(i) for i in compress_layer])+'_'+'-'.join([str(i) for i in new_compress_block])+'_'+'-'.join([str(i) for i in compress_rate])+'/'+'prune.caffemodel'
    net_new.save(net_new_name)
    print 'OK!'


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compress_layer", type=list, default = [0,1,2])
    parser.add_argument("--compress_block", type=list, default = [1,2], help='range from 0 to 1')
    parser.add_argument("--compress_rate", type=float, default = 0.9)
    parser.add_argument("--gpu", type=int, default = 4)

    opt = parser.parse_args()
    return opt

def pipeline(opt):
    solver_and_prototxt('/ssd/yqian/prune/model/ResNet50', opt.compress_layer, opt.compress_rate, opt.compress_block)
    index, w = get_index(list(opt.compress_rate), opt.compress_layer, opt.gpu, list(opt.compress_block), opt.method)

    compress_net(index, w, opt.compress_layer, list(opt.compress_block), list(opt.compress_rate))

if __name__ == '__main__':
    # compress_layer = int(sys.argv[1])
    # compress_block = int(sys.argv[2])
    # compress_rate = float(sys.argv[3])
    # gpu = int(sys.argv[4])

    opt = get_opt()

    solver_and_prototxt('/ssd/yqian/prune/model/ResNet50', opt.compress_layer, opt.compress_rate, opt.compress_block)
    index, w = get_index(opt.compress_rate, opt.compress_layer, opt.gpu, list(opt.compress_block))

    compress_net(index, w, opt.compress_layer, list(opt.compress_block), opt.compress_rate)
