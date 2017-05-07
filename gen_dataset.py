from scipy.io import loadmat
import csv
import numpy as np
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rate', type=float, default=0.1,
                        help='Set test/train rate of dataset. '
                        'Default is 0.1.')
    arg = parser.parse_args()

    return arg


def gen_annotation(per):
    mat = loadmat('dataset/mpii/mpii_human_pose_v1_u12_1.mat')
    Data = []

    #conversion data
    for i, (anno, train_flag) in enumerate(
        zip(mat['RELEASE']['annolist'][0, 0][0],
            mat['RELEASE']['img_train'][0, 0][0])):

        img_fn = str(anno['image']['name'][0, 0][0])
        train_flag = int(train_flag)

        if 'annopoints' in str(anno['annorect'].dtype):
            annopoints = anno['annorect']['annopoints'][0]
            for annopoint in annopoints:
                if len(annopoint) != 0:

                    annopoint = annopoint['point'][0, 0]
                    j_id = [str(j_i[0, 0]) for j_i in annopoint['id'][0]]
                    x = [x[0, 0] for x in annopoint['x'][0]]
                    y = [y[0, 0] for y in annopoint['y'][0]]
                    joint_pos = {}
                    for _j_id, (_x, _y) in zip(j_id, zip(x, y)):
                        joint_pos[str(_j_id)] = [float(_x), float(_y)]

                    joints = []
                    data = [img_fn]
                    if len(joint_pos) == 16:
                        for i in xrange(16):
                                for key in joint_pos.keys():
                                    if int(key) == i:
                                        joints += [joint_pos[key][0]]
                                        joints += [joint_pos[key][1]]
                        data.extend(joints)
                        Data += [data]
    #split data
    N = len(Data)
    N_test = int(N*per)
    perm = np.random.permutation(N)
    test_id = perm[:N_test]
    train_id = perm[N_test:]

    f = open('dataset/mpii/test_joints.csv', 'w')
    csvWriter = csv.writer(f)
    for i in test_id:
        csvWriter.writerow(Data[i])
    f.close

    f = open('dataset/mpii/train_joints.csv', 'w')
    csvWriter = csv.writer(f)
    for i in train_id:
        csvWriter.writerow(Data[i])
    f.close

if __name__ == '__main__' :
    arg = get_arguments()
    gen_annotation(arg.rate)
