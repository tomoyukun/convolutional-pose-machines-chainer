from model import convolutional_pose_machine
from chainer import Variable
from chainer import optimizers
from chainer import cuda
from chainer import serializers
from tqdm import tqdm

import numpy as np
import Traindataset
import Testdataset
import cmd_options
import log
import sys
import csv
import os
pairs = ((12,13),(11,14),(10,15),(2,3),(1,4), (0,5))

if __name__ == '__main__' :
    args = cmd_options.get_arguments()

    #prepare dataset and logger
    data = Traindataset.Train(args.traindata_path, args.im_size,
                         args.rotate_range, args.scale_range, pairs)
    eva =  Testdataset.Test(args.testdata_path, args.im_size, args.pck_rate)
    log = log.Log(args.dir)

    #prepare model
    cpm = convolutional_pose_machine.CPM(args.n_point, args.n_stage)
    if args.init_model:
        serializers.load_npz(args.init_model, cpm)
    optimizer = optimizers.Adam(alpha=args.lr)
    optimizer.setup(cpm)

    cuda.get_device(args.gpu).use()
    cpm.to_gpu()
    xp = cuda.cupy

    #trainloop
    for epoch in xrange(args.n_epoch):
        print 'epoch {}'.format(epoch+1)
        perm = np.random.permutation(len(data))
        trainloss = 0
        testloss = 0

        #train
        print 'training...'
        for i in tqdm(xrange(0, len(data), args.batchsize)):
            imgs = []
            b_maps = []
            c_maps = []
            miniperm = perm[i:i+args.batchsize]

            for j in xrange(len(miniperm)):
                img, b_map, c_map = data.generate(miniperm[j])
                imgs += [img]
                b_maps += [b_map]
                c_maps += [[c_map]]
            imgs = Variable(xp.asarray((np.transpose(np.array(imgs).astype(np.float32), (0,3,1,2)))))
            b_maps = Variable(xp.asarray(np.array(b_maps).astype(np.float32)))
            c_maps = Variable(xp.asarray(np.array(c_maps).astype(np.float32)))
            h, loss = cpm(imgs, c_maps, b_maps)

            #update paramater
            cpm.cleargrads()
            loss.backward()
            optimizer.update()

            trainloss += cuda.to_cpu(loss.data)*len(miniperm)
        print 'trainloss = {}'.format(trainloss / len(data))

        #test
        print 'testing'
        results = []
        for i in tqdm(xrange(len(eva))):
            img, b_map, c_map = eva.generate(i)
            img = img[np.newaxis,:,:,:]
            b_map = b_map[np.newaxis,:,:,:]
            c_map = c_map[np.newaxis,np.newaxis,:,:]
            img = Variable(xp.asarray((np.transpose(np.array(img).astype(np.float32), (0,3,1,2)))), volatile='on')
            b_map = Variable(xp.asarray(np.array(b_map).astype(np.float32)),volatile='on')
            c_map = Variable(xp.asarray(np.array(c_map).astype(np.float32)),volatile='on')
            h, loss = cpm(img, c_map, b_map)
            testloss += loss.data
            h = cuda.to_cpu(h.data[0])
            result = eva.evaluate(i, h)
            results +=[result]

        results = np.array(results)
        print 'testloss = {}'.format(testloss / len(eva))
        print 'testacc  = {}'.format(results.mean())
        # log and save
        log(epoch+1, results.mean(), trainloss/len(data), testloss/len(eva))
        serializers.save_npz(os.path.join(args.dir, 'trained_model','cpm_epoch{}'.format(epoch+1)), cpm)
