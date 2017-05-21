from model import convolutional_pose_machine
from chainer import Variable
from chainer import optimizers
from chainer import cuda
from chainer import serializers

import matplotlib as mpl
mpl.use('Agg')

import argparse
import copy
import cv2 as cv
import numpy as np
import Traindataset
import Testdataset
import matplotlib.pyplot as plt
import log
import sys
import csv
import os

pose = ('r-ankle','r-knee','r-hip', 'l-hip', 'l-knee', 'l-ankle', 'pelvis', 'thorax',
        'upperneck', 'head top', 'r-wrist', 'r-elbow', 'r-shoulder', 'l-shoulder', 'l-elbow', 'l-wrist')
color = ((18,0,230),(0,152,243),(0,241,225,),(31,195,143),(68,153,0),(150,158,0),(223,160,0),
         (183,104,0),(136,32,29),(131,7,146),(127,0,228),(79,0,229),(255,255,255),(141,149,0),
         (226,188,163),(105,168,105))

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0,
                        help='Set GPU device numbers with comma saparated. '
                        'Default is 0.')
    parser.add_argument('--n_stage', type=int, default=4,
                        help='Set the number of stages of your network '
                        'Default is 4 stages.')
    parser.add_argument('--init_model', type=str, default='trained_model.model',
                        help='Path to chainer model to load before trainig.')
    parser.add_argument('--n_point', type=int, default=16,
                        help='Set the number of joint points')
    parser.add_argument('--im_size', type=int, default=368,
                        help='Set input size of your network')
    parser.add_argument('--imagename', type=str, default='',
                        help='Set the name of image')
    parser.add_argument('--resultname', type=str, default='',
                        help='Set the name of result image')
    parser.add_argument('--c_deviation', type=float, default=2.0,
                        help='Set deviation of center map')

    args = parser.parse_args()

    return args


def makeGaussian(size, xd, yd, center, order=1000):
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    x0 = center[0]
    y0 = center[1]
    return order*np.exp(-0.5*(((x-x0)**2)/xd**2 + ((y-y0)**2)/yd**2) - np.log(2*(np.pi)*xd*yd))

def gcn_image(image):
    image = image.astype(np.float)
    image -= image.reshape(-1,3).mean(axis=0)
    image /= image.reshape(-1,3).std(axis=0) + 1e-5
    return image

def write_skelton(image, co):
    l_weight = int(image.shape[0]/100.0)
    cv.line(image, co[0], co[1], color[0], l_weight)
    cv.line(image, co[1], co[2], color[1], l_weight)
    cv.line(image, co[3], co[4], color[2], l_weight)
    cv.line(image, co[4], co[5], color[3], l_weight)
    cv.line(image, co[10], co[11], color[4], l_weight)
    cv.line(image, co[11], co[12], color[5], l_weight)
    cv.line(image, co[13], co[14], color[6], l_weight)
    cv.line(image, co[14], co[15], color[7], l_weight)
    cv.line(image, co[6], co[7], color[8], l_weight)
    cv.line(image, co[7], co[8], color[9], l_weight)
    cv.line(image, co[8], co[9], color[10], l_weight)

    return image



if __name__ == '__main__' :
    args = get_arguments()
    #prepare directory
    if not os.path.exists(os.path.join('demo_images', args.imagename)):
        print 'demo image does not exist'
        sys.exit()
    if not os.path.exists('demo_results'):
        os.mkdir('demo_results')

    #prepare model
    cpm = convolutional_pose_machine.CPM(args.n_point, args.n_stage)
    if args.init_model:
        serializers.load_npz(os.path.join('init_models',args.init_model), cpm)
    cuda.get_device(args.gpu).use()
    cpm.to_gpu()
    xp = cuda.cupy
    cpm.train = False

    #prepare image
    image = cv.imread(os.path.join(os.path.join('demo_images', args.imagename)))
    width = image.shape[1]
    height = image.shape[0]
    img = cv.resize(image, (args.im_size, args.im_size))
    img = gcn_image(img)
    c_map = makeGaussian(args.im_size, args.c_deviation*8, args.c_deviation*8, (args.im_size/2, args.im_size/2))
    img = img[np.newaxis,:,:,:]
    c_map = c_map[np.newaxis,np.newaxis,:,:]
    img = Variable(xp.asarray((np.transpose(np.array(img).astype(np.float32), (0,3,1,2)))), volatile='on')
    c_map = Variable(xp.asarray(np.array(c_map).astype(np.float32)), volatile = 'on')

    #start to demonstration
    h = cpm.predict(img, c_map)
    h = cuda.to_cpu(h.data[0])

    #visualization
    b_maps = plt.figure(figsize=(10,10), dpi=1000)
    re_image = copy.copy(image)
    co_s = []
    for i in xrange(args.n_point):
        ax = b_maps.add_subplot(4,4,i+1)
        b_map = h[i]
        ax.imshow(image, extent=(0,width,0,height))
        ax.imshow(b_map, extent=(0,width,0,height), alpha=0.5)
        ax.set_title('${}$'.format(pose[i]))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        co = np.unravel_index(b_map.argmax(), b_map.shape)
        co_x = int(co[1]*width/46.0)
        co_y = int(co[0]*height/46.0)
        cv.circle(re_image, (co_x, co_y), int(height/100), color[i], -1)
        co_s += [(co_x, co_y)]
    re_image = write_skelton(re_image, co_s)

    #save image
    if args.resultname:
        result_imname = args.resultname
        result_bename = 'belief_maps_{}'.format(args.resultname)
    else:
        result_imname = 'result_{}'.format(args.imagename)
        result_bename = 'belief_maps_{}'.format(args.imagename)
    b_maps.savefig(os.path.join('demo_results',result_bename))
    cv.imwrite(os.path.join('demo_results',result_imname), re_image)
    print 'completed'
