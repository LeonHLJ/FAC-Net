import argparse

parser = argparse.ArgumentParser(description='WTALC')


# basic setting
parser.add_argument('--gpus', type=int, default=[0], nargs='+', help='used gpu')
parser.add_argument('--rgb-model-id', type=int, default=1, help='model id for saving model (first stream)')
parser.add_argument('--flow-model-id', type=int, default=2, help='model id for saving model (second stream)')

# loading model
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--pretrained', default=True, help='is pretrained model')
parser.add_argument('--rgb-load-epoch', type=int, default=100, help='epoch of loaded model')
parser.add_argument('--flow-load-epoch', type=int, default=100, help='epoch of loaded model')

# dataset patameters
parser.add_argument('--dataset-root', default='your_dataset_path', help='dataset root path')
parser.add_argument('--dataset-name', default='Thumos14reduced', help='dataset to train on')

# input paramaters
parser.add_argument('--feature-type', type=str, default='I3D', help='type of feature to be used (default: I3D)')
parser.add_argument('--inp-feat-num', type=int, default=1024, help='size of input feature (default: 1024)')
parser.add_argument('--out-feat-num', type=int, default=1024, help='size of output feature (default: 1024)')
parser.add_argument('--class-num', type=int, default=20, help='number of classes (default: )')
parser.add_argument('--scale-factor', type=float, default=5.0, help='temperature factors')
parser.add_argument('--temperature', type=float, default=[1.0, 2.0, 5.0], help='temperature factors')

# model parameters
parser.add_argument('--dropout', default=0.5, help='dropout value (default: 0.5)')

# testing paramaters
parser.add_argument('--class-threshold', type=float, default=0.1, help='class threshold for rejection')
parser.add_argument('--start-threshold', type=float, default=0.032, help='start threshold for action localization')
parser.add_argument('--end-threshold', type=float, default=0.055, help='end threshold for action localization')
parser.add_argument('--threshold-interval', type=float, default=0.004, help='threshold interval for action localization')
