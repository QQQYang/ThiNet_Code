# import ptvsd
# ptvsd.enable_attach(address = ('0.0.0.0', 5678))
# ptvsd.wait_for_attach()
import os
import compress_model_reid as cm
import argparse
import numpy as np

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compress_layer", type=int, nargs='+', default = [0,1,2,3,4])
    parser.add_argument("--compress_block", type=int, nargs='+', default = [1,2], help='range from 1 to 2')
    parser.add_argument("--compress_rate", type=float, nargs='+', default = [0.5, 0.5])
    parser.add_argument("--gpu", type=int, default = 4)
    parser.add_argument('--method', type=str, default='single')

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    gpu = 0
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
    opt.gpu = int(gpu)
    cm.pipeline(opt)

    # model_dir = os.path.join('/ssd/yqian/prune/model/reid', '-'.join([str(i) for i in opt.compress_layer])+'_'+'-'.join([str(i) for i in opt.compress_block])+'_'+'-'.join([str(i) for i in opt.compress_rate]))
    # os.system('mpiexec --allow-run-as-root -np 8 /root/caffe/build/tools/caffe_parallel train --solver %s/solver.prototxt --weights=%s/prune.caffemodel 2>&1 | tee %s_train.log' % (model_dir, model_dir, model_dir))
    # os.system('python ThiNet_Code/ToolKit/FLOPs_and_size.py %s/trainval.prototxt' % model_dir)
    