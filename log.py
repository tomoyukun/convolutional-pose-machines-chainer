import csv
import os

class Log(object):

    def __init__(self, dir_path):
        self.log_path = os.path.join(dir_path, 'log.csv')
        with open(self.log_path, 'w') as f:
            f.write('epoch, testacc, trainloss, testloss\n')
            
    def __call__(self, epoch, testacc, trainloss, testloss):
        with open(self.log_path, 'a') as f:
            f.write('{},{},{},{}\n'.format(epoch, testacc, trainloss, testloss))

        
