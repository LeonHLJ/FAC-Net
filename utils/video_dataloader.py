import torch
import numpy as np
import utils.utils as utils
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, args, run_type):
        self.args = args
        self.run_type = run_type
        self.dataset_name = args.dataset_name
        self.num_class = args.class_num
        self.feature_size = args.inp_feat_num
        self.path_to_features = args.dataset_root + '%s/%s-%s-JOINTFeatures.npy' % (
            args.dataset_name, args.dataset_name, args.feature_type)
        self.path_to_annotations = args.dataset_root + self.dataset_name + '/'
        self.features = np.load(self.path_to_features, encoding='bytes')
        self.segments = np.load(self.path_to_annotations + 'segments.npy')
        self.labels = np.load(self.path_to_annotations + 'labels_all.npy')  # Specific to Thumos14
        self.classlist = np.load(self.path_to_annotations + 'classlist.npy')
        self.subset = np.load(self.path_to_annotations + 'subset.npy')
        self.trainidx = []
        self.testidx = []
        self.classwiseidx = []
        self.currenttestidx = 0
        self.labels_multihot = [utils.strlist2multihot(labs, self.classlist) for labs in self.labels]

        self.train_test_idx()
        self.classwise_feature_mapping()

    def __len__(self):
        if self.run_type == 'rgb_train' or self.run_type == 'flow_train':
            return int(len(self.trainidx))
        else:
            return int(len(self.testidx))

    def train_test_idx(self):
        for i, s in enumerate(self.subset):
            if s.decode('utf-8') == 'validation':  # Specific to Thumos14
                self.trainidx.append(i)
            else:
                self.testidx.append(i)

    def classwise_feature_mapping(self):
        for category in self.classlist:
            idx = []
            for i in self.trainidx:
                for label in self.labels[i]:
                    if label == category.decode('utf-8'):
                        idx.append(i)
                        break
            self.classwiseidx.append(idx)

    def __getitem__(self, idx):
        sample = dict()
        if self.run_type == 'rgb_train':
            labs = self.labels_multihot[self.trainidx[idx]]
            feat = self.features[self.trainidx[idx]][:, 0:1024]
            sample['data'] = feat
            sample['labels'] = labs
        elif self.run_type == 'rgb_test':
            labs = self.labels_multihot[self.testidx[idx]]
            feat = self.features[self.testidx[idx]][:, 0:1024]
            sample['vid_len'] = feat.shape[0]
            sample['data'] = feat
            sample['labels'] = labs
        elif self.run_type == 'flow_train':
            labs = self.labels_multihot[self.trainidx[idx]]
            feat = self.features[self.trainidx[idx]][:, 1024:]
            sample['data'] = feat
            sample['labels'] = labs
        elif self.run_type == 'flow_test':
            labs = self.labels_multihot[self.testidx[idx]]
            feat = self.features[self.testidx[idx]][:, 1024:]
            sample['vid_len'] = feat.shape[0]
            sample['data'] = feat
            sample['labels'] = labs
        elif self.run_type == 'eval':
            labs = self.labels_multihot[self.testidx[idx]]
            rgb_feat = self.features[self.testidx[idx]][:, 0:1024]
            flow_feat = self.features[self.testidx[idx]][:, 1024:]
            assert (rgb_feat.shape[0] == flow_feat.shape[0])
            sample['vid_len'] = rgb_feat.shape[0]
            sample['rgb_data'] = rgb_feat
            sample['flow_data'] = flow_feat
            sample['labels'] = labs

        return sample
