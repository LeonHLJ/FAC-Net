from __future__ import print_function
import os
import torch
import torch.nn as nn
from evaluation.eval import ts_eval
from model.main_branch import WSTAL
from utils.video_dataloader import VideoDataset
from tensorboard_logger import Logger


class Processor():
    def __init__(self, args):
        # parameters
        self.args = args
        # create logger
        log_dir = './logs/' + self.args.dataset_name + '/eval'
        self.logger = Logger(log_dir)
        # device
        self.device = torch.device(
            'cuda:' + str(self.args.gpu_ids[0]) if torch.cuda.is_available() and len(self.args.gpu_ids) > 0 else 'cpu')

        # dataloader
        if self.args.dataset_name in ['Thumos14', 'Thumos14reduced']:
            self.data_loader = torch.utils.data.DataLoader(VideoDataset(self.args, 'eval'), batch_size=1,
                                                           shuffle=False, drop_last=False)
        else:
            raise ValueError('Do Not Exist This Dataset')

        # Model Setting
        self.rgb_model = WSTAL(self.args).to(self.device)
        self.flow_model = WSTAL(self.args).to(self.device)

        # Model Parallel Setting
        if len(self.args.gpu_ids) > 1:
            self.rgb_model = nn.DataParallel(self.rgb_model, device_ids=self.args.gpu_ids)
            self.rgb_model_module = self.rgb_model.module
            self.flow_model = nn.DataParallel(self.flow_model, device_ids=self.args.gpu_ids)
            self.flow_model_module = self.flow_model.module
        else:
            self.rgb_model_module = self.rgb_model
            self.flow_model_module = self.flow_model

        # Loading Pretrained Model
        if self.args.pretrained:
            rgb_model_dir = './ckpt/' + self.args.dataset_name + '/' + str(self.args.rgb_model_id) + '/' + str(
                self.args.rgb_load_epoch) + '.pkl'
            if os.path.isfile(rgb_model_dir):
                self.rgb_model_module.load_state_dict(torch.load(rgb_model_dir))
            else:
                raise ValueError('Do Not Exist This RGB Pretrained File')
            flow_model_dir = './ckpt/' + self.args.dataset_name + '/' + str(self.args.flow_model_id) + '/' + str(
                self.args.flow_load_epoch) + '.pkl'
            if os.path.isfile(rgb_model_dir):
                self.flow_model_module.load_state_dict(torch.load(flow_model_dir))
            else:
                raise ValueError('Do Not Exist This Flow Pretrained File')

    def processing(self):
        self.val()

    def val(self):
        print('Start testing!')
        self.rgb_model_module.eval()
        self.flow_model_module.eval()
        ts_eval(self.data_loader, self.args, self.logger, self.rgb_model_module, self.flow_model_module, self.device)
        print('Finish testing!')
