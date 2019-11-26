# import ptvsd
# ptvsd.enable_attach(address = ('0.0.0.0', 5689))
# ptvsd.wait_for_attach()
import os
import compress_model as cm
import argparse
import numpy as np

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compress_layer", type=int, nargs='+', default = [0,1,2,3,4])
    parser.add_argument("--compress_block", type=list, default = [1,2], help='range from 1 to 2')
    parser.add_argument("--compress_rate", type=float, default = 0.9)
    parser.add_argument("--gpu", type=int, default = 4)

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    gpu = 0
    while True:
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
        memory_max = max(memory_gpu)
        if memory_max>10000:
            gpu = np.argmax(memory_gpu)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu))
            os.system('rm tmp')
            print('Find vacant GPU: %d' % gpu)
            break
    opt = get_opt()
    opt.gpu = int(gpu)
    cm.pipeline(opt)

    model_dir = os.path.join('/ssd/yqian/prune/model/ResNet50', '-'.join([str(i) for i in opt.compress_layer])+'_'+'-'.join([str(i) for i in opt.compress_block])+'_'+str(opt.compress_rate))
    os.system('/root/caffe/build/tools/caffe_parallel train --solver %s/solver.prototxt --weights=%s/prune.caffemodel' % (model_dir, model_dir))
    os.system('/root/caffe/build/tools/caffe_parallel test --model %s/trainval.prototxt --weights=%s/snapshot/_iter_10000.caffemodel.h5' % (model_dir, model_dir))
    os.system('python ThiNet_Code/ToolKit/FLOPs_and_size.py %s/trainval.prototxt' % model_dir)
    