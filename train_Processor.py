from __future__ import print_function
import os
import torch
import torch.nn as nn
import numpy as np
from evaluation.eval import ss_eval
from model.main_branch import WSTAL
from model.losses import NormalizedCrossEntropy
from utils.video_dataloader import VideoDataset
from tensorboard_logger import Logger


class Processor():
    def __init__(self, args):
        # parameters
        self.args = args
        # create logger
        log_dir = './logs/' + self.args.dataset_name + '/' + str(self.args.model_id)
        self.logger = Logger(log_dir)
        # device
        self.device = torch.device(
            'cuda:' + str(self.args.gpu_ids[0]) if torch.cuda.is_available() and len(self.args.gpu_ids) > 0 else 'cpu')

        # dataloader
        if self.args.dataset_name in ['Thumos14', 'Thumos14reduced']:
            if self.args.run_type == 0:
                self.train_data_loader = torch.utils.data.DataLoader(VideoDataset(self.args, 'rgb_train'),
                                                                     batch_size=1,
                                                                     shuffle=True,
                                                                     num_workers=2 * len(self.args.gpu_ids),
                                                                     drop_last=False)
                self.test_data_loader = torch.utils.data.DataLoader(VideoDataset(self.args, 'rgb_test'), batch_size=1,
                                                                    shuffle=False, drop_last=False)
            elif self.args.run_type == 1:
                self.train_data_loader = torch.utils.data.DataLoader(VideoDataset(self.args, 'flow_train'),
                                                                     batch_size=1,
                                                                     shuffle=True,
                                                                     num_workers=2 * len(self.args.gpu_ids),
                                                                     drop_last=False)
                self.test_data_loader = torch.utils.data.DataLoader(VideoDataset(self.args, 'flow_test'), batch_size=1,
                                                                    shuffle=False, drop_last=False)
            elif self.args.run_type == 2:
                self.test_data_loader = torch.utils.data.DataLoader(VideoDataset(self.args, 'rgb_test'), batch_size=1,
                                                                    shuffle=False, drop_last=False)
            elif self.args.run_type == 3:
                self.test_data_loader = torch.utils.data.DataLoader(VideoDataset(self.args, 'flow_test'), batch_size=1,
                                                                    shuffle=False, drop_last=False)
        else:
            raise ValueError('Do Not Exist This Dataset')

        # Loss Function Setting
        self.loss_nce = NormalizedCrossEntropy()

        # Model Setting
        self.model = WSTAL(self.args).to(self.device)

        # Model Parallel Setting
        if len(self.args.gpu_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model_module = self.model.module
        else:
            self.model_module = self.model

        # Loading Pretrained Model
        if self.args.pretrained:
            model_dir = './ckpt/' + self.args.dataset_name + '/' + str(self.args.model_id) + '/' + str(
                self.args.load_epoch) + '.pkl'
            if os.path.isfile(model_dir):
                self.model_module.load_state_dict(torch.load(model_dir))
            else:
                raise ValueError('Do Not Exist This Pretrained File')

        # Optimizer Setting
        if self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=[0.9, 0.99],
                                              weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                             weight_decay=self.args.weight_decay, nesterov=True)
        else:
            raise ValueError('Do Not Exist This Optimizer')

        # Optimizer Parallel Setting
        if len(self.args.gpu_ids) > 1:
            self.optimizer = nn.DataParallel(self.optimizer, device_ids=self.args.gpu_ids)
            self.optimizer_module = self.optimizer.module
        else:
            self.optimizer_module = self.optimizer

    def processing(self):
        if self.args.run_type == 0 or self.args.run_type == 1:
            self.train()
        elif self.args.run_type == 2 or self.args.run_type == 3:
            self.val(self.args.load_epoch)
        else:
            raise ValueError('Do not Exist This Processing')

    def train(self):
        print('Start training!')
        self.model_module.train(mode=True)

        if self.args.pretrained:
            epoch_range = range(self.args.load_epoch, self.args.max_epoch)
        else:
            epoch_range = range(self.args.max_epoch)

        iter = 0
        step = 0
        current_lr = self.args.lr
        loss_recorder = {
            'cw': 0,
            'ca': 0,
            'mil': 0,
            'sum': 0,
        }
        for epoch in epoch_range:
            for num, sample in enumerate(self.train_data_loader):
                if self.args.decay_type == 0:
                    for param_group in self.optimizer_module.param_groups:
                        param_group['lr'] = current_lr
                elif self.args.decay_type == 1:
                    if num == 0:
                        current_lr = self.Step_decay_lr(epoch)
                        for param_group in self.optimizer_module.param_groups:
                            param_group['lr'] = current_lr
                elif self.args.decay_type == 2:
                    current_lr = self.Cosine_decay_lr(epoch, num)
                    for param_group in self.optimizer_module.param_groups:
                        param_group['lr'] = current_lr

                iter = iter + 1
                features = sample['data'].numpy()
                labels = sample['labels'].numpy()

                labels = torch.from_numpy(labels).float().to(self.device)
                features = torch.from_numpy(features).float().to(self.device)

                ab_labels = torch.cat([labels, torch.ones(labels.size(0), 1).to(self.device)], -1)
                awb_labels = torch.cat([labels, torch.zeros(labels.size(0), 1).to(self.device)], -1)

                cw_pred, ca_pred, mil_pred, frm_scrs = self.model(features)

                cls_cw_loss = self.loss_nce(cw_pred, awb_labels)
                cls_ca_loss = self.loss_nce(ca_pred, awb_labels)
                cls_mil_loss = self.loss_nce(mil_pred, ab_labels)

                total_loss = cls_cw_loss * self.args.lambda_cw + cls_ca_loss * self.args.lambda_ca \
                            + cls_mil_loss * self.args.lambda_mil

                loss_recorder['cw'] += cls_cw_loss.item()
                loss_recorder['ca'] += cls_ca_loss.item()
                loss_recorder['mil'] += cls_mil_loss.item()
                loss_recorder['sum'] += total_loss.item()

                total_loss.backward()

                if iter % self.args.batch_size == 0:
                    step += 1
                    print('Epoch: {}/{}, Iter: {:02d}, Lr: {:.6f}'.format(
                        epoch + 1,
                        self.args.max_epoch,
                        step,
                        current_lr), end=' ')
                    for k, v in loss_recorder.items():
                        print('Loss_{}: {:.4f}'.format(k, v / self.args.batch_size), end=' ')
                        loss_recorder[k] = 0

                    print()

                    self.optimizer_module.step()
                    self.optimizer_module.zero_grad()

            if (epoch + 1) % self.args.save_interval == 0:
                out_dir = './ckpt/' + self.args.dataset_name + '/' + str(self.args.model_id) + '/' + str(
                    epoch + 1) + '.pkl'
                torch.save(self.model_module.state_dict(), out_dir)
                self.model_module.eval()
                ss_eval(epoch + 1, self.test_data_loader, self.args, self.logger, self.model_module, self.device)
                self.model_module.train()

    def val(self, epoch):
        print('Start testing!')
        self.model_module.eval()
        ss_eval(epoch, self.test_data_loader, self.args, self.logger, self.model_module, self.device)
        print('Finish testing!')

    def Step_decay_lr(self, epoch):
        lr_list = []
        current_epoch = epoch + 1
        for i in range(0, len(self.args.changeLR_list) + 1):
            lr_list.append(self.args.lr * (2.0 ** i))

        lr_range = self.args.changeLR_list.copy()
        lr_range.insert(0, 0)
        lr_range.append(self.args.max_epoch + 1)

        if len(self.args.changeLR_list) != 0:
            for i in range(0, len(lr_range) - 1):
                if lr_range[i + 1] >= current_epoch > lr_range[i]:
                    lr_step = i
                    break

        current_lr = lr_list[lr_step]
        return current_lr

    def Cosine_decay_lr(self, epoch, batch):
        if self.args.warmup:
            max_epoch = self.args.max_epoch - self.args.warmup_epoch
            current_epoch = epoch + 1 - self.args.warmup_epoch
        else:
            max_epoch = self.args.max_epoch
            current_epoch = epoch + 1

        current_lr = 1 / 2.0 * (1.0 + np.cos(
            (current_epoch * self.args.batch_num + batch) / (max_epoch * self.args.batch_num) * np.pi)) * self.args.lr

        return current_lr
