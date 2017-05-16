from skimage import transform
from tqdm import tqdm

import csv
import cv2 as cv
import numpy as np
import os

class Test(object):

    def __init__(self, raw_data, imsize, pck_rate=0.2, c_deviation=2):
        for key, val in locals().items():
            setattr(self, key, val)
        self.load_data(self.raw_data)

    def __len__(self):
        return len(self.images)

    def load_data(self, raw_data):
        self.images = []
        self.joint_x = []
        self.joint_y = []
        self.cropsize = []
        print 'preparing test dataset...'
        with open(raw_data,"rb") as f:
            reader = csv.reader(f)
            for row in tqdm(reader):
                path = row[0]
                image = cv.imread("dataset/mpii/images/{}".format(path))
                joint_x = np.array(map(float, row[1::2]))
                joint_y = np.array(map(float, row[2::2]))
                bbox = [(max(joint_x) + min(joint_x)) / 2.0,
                        (max(joint_y) + min(joint_y)) / 2.0,
                        (max(joint_x) - min(joint_x)) * 1.2,
                        (max(joint_y) - min(joint_y)) * 1.2]
                image, joint_x, joint_y, bbox = self.crop_image(image, joint_x, joint_y, bbox)
                self.cropsize += [[image.shape[1], image.shape[0]]]
                image, joint_x, joint_y = self.resize_image(image, joint_x, joint_y)
                image = self.gcn_image(image)
                self.images += [image]
                self.joint_x += [joint_x]
                self.joint_y += [joint_y]

        print 'ready!'

    def generate(self, i):
        image = self.images[i]
        joint_x = np.array(self.joint_x[i])
        joint_y = np.array(self.joint_y[i])
        t = []
        h_map = np.zeros((self.imsize/8, self.imsize/8))
        for x, y in zip(joint_x, joint_y):
            b_map = self.makeGaussian(self.imsize/8, self.c_deviation, self.c_deviation, (x/8, y/8))
            h_map = h_map - b_map
            t += [b_map]
        t += [h_map]
        c_map = self.makeGaussian(self.imsize, self.c_deviation*8, self.c_deviation*8, (self.imsize/2, self.imsize/2))
        t = np.array(t)
        return image, t, c_map

    def evaluate(self, i, h):
        joint_x = self.joint_x[i]
        joint_y = self.joint_y[i]
        is_correct = []
        for j in xrange(len(joint_x)):
            b_map = h[j]
            co = np.unravel_index(b_map.argmax(), b_map.shape)

            if ((joint_x[j] - co[1]*8)/float(self.imsize))**2 + ((joint_y[j] - co[0]*8)/float(self.imsize))**2 < (self.pck_rate)**2:
                is_correct += [1]
            else:
                is_correct += [0]
        return is_correct

    def resize_image(self, image, joint_x, joint_y):
        joint_x = joint_x*self.imsize / image.shape[1]
        joint_y = joint_y*self.imsize / image.shape[0]
        image = cv.resize(image, (self.imsize, self.imsize))
        return image, joint_x, joint_y


    def crop_image(self, image, joint_x, joint_y, bbox):
        scale = 1.0
        bbox[2] = bbox[2]*scale
        bbox[3] = bbox[3]*scale
        bb_x1=int(bbox[0]-bbox[2]/2)
        bb_y1=int(bbox[1]-bbox[3]/2)
        bb_x2=int(bbox[0]+bbox[2]/2)
        bb_y2=int(bbox[1]+bbox[3]/2)
        if bb_x1<0 or bb_x2>image.shape[1] or bb_y1<0 or bb_y2 > image.shape[0]:
            pad = int(max(-(bb_x1), bb_x2-image.shape[1], -(bb_y1), bb_y2-image.shape[0]))
            image = np.pad(image, ((pad,pad),(pad,pad),(0,0)), 'constant')
        else:
            pad = 0
        image = image[bb_y1+pad:bb_y2+pad, bb_x1+pad:bb_x2+pad]
        joint_x = joint_x - bb_x1
        joint_y = joint_y - bb_y1
        bbox[0] = bbox[0] - bb_x1
        bbox[1] = bbox[1] - bb_y1
        return image, joint_x, joint_y, bbox

    def gcn_image(self, image):
        image = image.astype(np.float)
        image -= image.reshape(-1,3).mean(axis=0)
        image /= image.reshape(-1,3).std(axis=0) + 1e-5
        return image


    def makeGaussian(self, size, xd, yd, center, order=1000):
        x = np.arange(0, size, 1, float)
        y = x[:,np.newaxis]
        x0 = center[0]
        y0 = center[1]
        return order*np.exp(-0.5*(((x-x0)**2)/xd**2 + ((y-y0)**2)/yd**2) - np.log(2*(np.pi)*xd*yd))
