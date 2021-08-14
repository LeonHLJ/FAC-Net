import argparse

parser = argparse.ArgumentParser(description='WSTAL')

# basic setting
parser.add_argument('--gpus', type=int, default=[0], nargs='+', help='used gpu')
parser.add_argument('--run-type', type=int, default=0,
                    help='train rgb (0) or train flow (1) or evaluate rgb (2) or evaluate flow (3)')
parser.add_argument('--model-id', type=int, default=1, help='model id for saving model')

# loading model
parser.add_argument('--pretrained', default=False, help='is pretrained model')
parser.add_argument('--load-epoch', type=int, default=None, help='epoch of loaded model')

# storing parameters
parser.add_argument('--save-interval', type=int, default=5, help='interval for storing model')

# dataset patameters
parser.add_argument('--dataset-root', default='your_dataset_path', help='dataset root path')
parser.add_argument('--dataset-name', default='Thumos14reduced', help='dataset to train on')

# model settings
parser.add_argument('--feature-type', type=str, default='I3D', help='type of feature to be used (default: I3D)')
parser.add_argument('--inp-feat-num', type=int, default=1024, help='size of input feature (default: 1024)')
parser.add_argument('--out-feat-num', type=int, default=1024, help='size of output feature (default: 1024)')
parser.add_argument('--class-num', type=int, default=20, help='number of classes (default: )')
parser.add_argument('--scale-factor', type=float, default=5.0, help='temperature factors')
parser.add_argument('--temperature', type=float, default=[1.0, 2.0, 5.0], help='temperature factors')

# training paramaters
parser.add_argument('--batch-size', type=int, default=10, help='number of instances in a batch of data (default: 10)')
parser.add_argument('--optimizer', type=str, default='Adam', help='used optimizer')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument('--weight-decay', type=float, default=0.005, help='weight deacy (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
parser.add_argument('--dropout', default=0.5, help='dropout value (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--max-epoch', type=int, default=100, help='maximum iteration to train (default: 50000)')

parser.add_argument('--lambda-cw', default=1.0, help='balancing hyper-parameter of cw branch')
parser.add_argument('--lambda-ca', default=0.1, help='balancing hyper-parameter of ca branch')
parser.add_argument('--lambda-mil', default=0.1, help='balancing hyper-parameter of mil branch')

# testing paramaters
parser.add_argument('--class-threshold', type=float, default=0.1, help='class threshold for rejection')
parser.add_argument('--start-threshold', type=float, default=0.03, help='start threshold for action localization')
parser.add_argument('--end-threshold', type=float, default=0.055, help='end threshold for action localization')
parser.add_argument('--threshold-interval', type=float, default=0.005, help='threshold interval for action localization')

# Learning Rate Decay
parser.add_argument('--decay-type', type=int, default=0,
                    help='weight decay type (0 for None, 1 for step decay, 2 for cosine decay)')
parser.add_argument('--changeLR_list', type=int, default=[40, 1000], help='change lr step')
