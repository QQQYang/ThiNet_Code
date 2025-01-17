'''
usage: python data_utils.py cifar10_dir/
'''
# import ptvsd
# ptvsd.enable_attach(address = ('0.0.0.0', 5678))
# ptvsd.wait_for_attach()
import sys
import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread, imsave

import numpy as np
import lmdb
import caffe
import cv2

def load_CIFAR_batch(filename, pad=True):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).astype(np.uint8)
    padded = np.zeros((10000, 3, 40, 40), dtype=np.uint8)
    padded[:,:,:,:] = 128
    padded[:,:,4:-4, 4:-4] = X
    Y = np.array(Y, dtype=np.int64) 
    if not pad:
      return X, Y
    return padded, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f, False)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  idx = np.arange(len(Ytr))
  np.random.shuffle(idx)
  print 'shuffle training data', len(idx)
  Xtr = Xtr[idx]
  Ytr = Ytr[idx]
  print idx
  print 'tr label',Ytr.min(), Ytr.max()
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'), pad=False)
  print 'te label',Yte.min(), Yte.max()
  print Xtr.shape
  print Ytr.shape
  print Xte.shape
  print Yte.shape
  return Xtr, Ytr, Xte, Yte

def py2lmdb(X, y, save_path):
  # Let's pretend this is interesting data

  assert X.dtype == np.uint8
  N = X.shape[0]
  assert N == y.shape[0], str(N) + ' ' + str(y.shape)
  
  
  # We need to prepare the database for the size. We'll set it 10 times
  # greater than what we theoretically need. There is little drawback to
  # setting this too big. If you still run into problem after raising
  # this, you might want to try saving fewer entries in a single
  # transaction.
  map_size = X.nbytes * 10
  
  env = lmdb.open(save_path, map_size=map_size)
  
  with env.begin(write=True) as txn:
      # txn is a Transaction object
      for i in range(N):
          datum = caffe.proto.caffe_pb2.Datum()
          datum.channels = X.shape[1]
          datum.height = X.shape[2]
          datum.width = X.shape[3]
          datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
          datum.label = int(y[i])
          str_id = '{:08}'.format(i)
  
          # The encode is only essential in Python 3
          txn.put(str_id.encode('ascii'), datum.SerializeToString())

def save_imgs(X, Y, root_dir):
  assert X.dtype == np.uint8
  for i in range(X.shape[0]):
    if not os.path.exists(os.path.join(root_dir, 'img', str(Y[i]))):
      os.makedirs(os.path.join(root_dir, 'img', str(Y[i])))
    imsave(os.path.join(root_dir, 'img', str(Y[i]), str(i)+'.jpg'), X[i].transpose((1,2,0)))
    # cv2.imwrite(os.path.join(root_dir, str(Y[i]), str(i)+'.jpg'), X[i].transpose((1,2,0)))


if __name__ == '__main__':
  root = sys.argv[1]
  Xtr, Ytr, Xte, Yte = load_CIFAR10(root)
  save_imgs(Xtr, Ytr, root)
  # paths = [ os.path.join(root, i) for i in ['train', 'test']]
  # py2lmdb(Xtr, Ytr, paths[0])
  # py2lmdb(Xte, Yte, paths[1])
  # for i in paths:
  #   print 'saved to', i
  
