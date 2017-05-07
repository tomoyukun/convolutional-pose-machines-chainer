# Convolutional Pose Machines implementation by Chainer
Experimental implementation. This is not official.  
Original paper is [Convolutional Pose Machines](http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Wei_Convolutional_Pose_Machines_CVPR_2016_paper.html), and official implementarion by caffe is [convolutional-pose-machines-release](https://github.com/CMU-Perceptual-Computing-Lab/convolutional-pose-machines-release).  
 I will add more information soon.
# Requirements
- python 2.7
- Chainer 1.17.0+
# Start to demo
## Download trained model
    cd init_models
    wget https://www.dropbox.com/s/k3ykwmt290iceuo/trained_model.model
note : This trained model is 4stages-model so you should set n_stage of next command to 4.
## Use demo.py
    python demo.py --imagename sample.jpg --gpu 0 --n_stage 4
note : This implementation for only turning on GPU.  

# Start to training
## Prepare MPII dataset
    bash dataset.sh
    python gen_dataset.py --rate 0.1
## Use train.py
For example,   

    python train.py --gpu 0 --n_epoch 100 --batchsize 8 --lr 0.0005 --n_stage 6

If you want to check possible options,

    python train.py --help
# examples of demo.py
![sample1](https://github.com/tomoyukun/convolutional-pose-machine-chainer/blob/image/output6.png)
![bliefmap](https://github.com/tomoyukun/convolutional-pose-machine-chainer/blob/image/belief_maps2.png)
![sample2](https://github.com/tomoyukun/convolutional-pose-machine-chainer/blob/image/output3.jpg)
![sample3](https://github.com/tomoyukun/convolutional-pose-machine-chainer/blob/image/output8.jpg)




