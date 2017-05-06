import argparse
import time
import os
import logging
#import os

def create_log(args):
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        filename=os.path.join(args.dir, 'options.txt'), level=logging.DEBUG)
    logging.info(args)


def create_result_dir(dire):
    if not os.path.exists('results'):
        os.mkdir('results')

    if dire:
        result_dir = os.path.join('results', dire)
    else:
        result_dir = os.path.join(
                'results', time.strftime('%Y-%m-%d_%H-%M-%S'))

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
        os.mkdir(os.path.join(result_dir, 'trained_model'))

    return result_dir


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0,
                        help='Set GPU device numbers with comma saparated. '
                        'Default is 0.')
    parser.add_argument('--n_epoch', type=int, default=50,
                        help='Set training epoch '
                        'Default is 50 epoch.')
    parser.add_argument('--batchsize', type=int, default=8,
                        help='Default is 8.')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Initial learning rate. Default is 0.0005')
    parser.add_argument('--n_stage', type=int, default=4,
                        help='Set the number of stages of your network '
                        'Default is 4 stages.')
    parser.add_argument('--init_model', type=str, default='',
                        help='Path to chainer model to load before trainig.')
    parser.add_argument('--n_point', type=int, default=16,
                        help='Set the number of joint points')
    parser.add_argument('--im_size', type=int, default=368,
                        help='Set input size of your network')
    parser.add_argument('--traindata_path', type=str, default='dataset/mpii/train_joints.csv',
                        help='Set training datafile')
    parser.add_argument('--testdata_path', type=str, default='dataset/mpii/test_joints.csv',
                        help='Set testing datafile')
    parser.add_argument('--scale_range', type=str, default='0.9, 1.3',
                        help='Set range of scaling for date augmentation')
    parser.add_argument('--rotate_range', type=int, default=30,
                        help='Set range of rotating for date augmentation')
    parser.add_argument('--dir', type=str, default='',
                        help='Directory name to save logs.')
    parser.add_argument('--pck_rate', type=float, default=0.2,
                        help='Set the rate for evalutating model')

    args = parser.parse_args()

    args.scale_range = map(float,args.scale_range.split(','))
    args.dir = create_result_dir(args.dir)

    create_log(args)

    return args
