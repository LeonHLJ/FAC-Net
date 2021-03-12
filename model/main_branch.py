import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.init as torch_init
from torch.autograd import Variable
import pdb


def weights_init_random(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


def calculate_l1_norm(f):
    f_norm = torch.norm(f, p=2, dim=-1, keepdim=True)
    f = torch.div(f, f_norm)
    return f


class WSTAL(nn.Module):
    def __init__(self, args):
        super().__init__()
        # read parameters
        self.n_in = args.inp_feat_num
        self.n_out = args.out_feat_num
        self.n_class = args.class_num
        # hyper-parameters
        self.dropout = args.dropout
        self.scale_factor = args.scale_factor
        self.temperature = args.temperature

        # feature embedding
        self.feature_embedding = FeatureEmbedding(self.n_in, self.n_out, self.dropout)
        # action features
        self.ac_center = nn.Parameter(torch.zeros(self.n_class + 1, self.n_out))
        torch_init.xavier_uniform_(self.ac_center)
        # foreground feature
        self.fg_center = nn.Parameter(-1.0 * self.ac_center[-1, ...][None, ...])

        self.apply(weights_init_random)

    def forward(self, x):
        # feature embedding
        x_emb = self.feature_embedding(x)

        # normalization
        norms_emb = calculate_l1_norm(x_emb)
        norms_ac = calculate_l1_norm(self.ac_center)
        norms_fg = calculate_l1_norm(self.fg_center)

        # generate foeground and action scores
        frm_scrs = torch.einsum('ntd,cd->ntc', [norms_emb, norms_ac]) * self.scale_factor
        frm_fg_scrs = torch.einsum('ntd,kd->ntk', [norms_emb, norms_fg]).squeeze(-1) * self.scale_factor

        # attention
        class_wise_atts = [F.softmax(frm_scrs * t, 1) for t in self.temperature]
        class_agno_atts = [F.softmax(frm_fg_scrs * t, 1) for t in self.temperature]

        # class-wise foreground classification branch
        # class-wise feature aggregation
        cw_vid_feats = [torch.einsum('ntd,ntc->ncd', [x_emb, att]) for att in class_wise_atts]
        # normalization
        norms_cw_vid_feats = [calculate_l1_norm(f) for f in cw_vid_feats]
        # calculate score
        cw_vid_scrs = [torch.einsum('ncd,kd->nck', [f, norms_fg]).squeeze(-1) for f in norms_cw_vid_feats]
        cw_vid_scr = torch.stack(cw_vid_scrs, -1).mean(-1) * self.scale_factor * 2.0  # here we keep the scale of cosine similarity as 10.0
        cw_vid_pred = F.softmax(cw_vid_scr, -1)

        # class-agnostic branch
        # foreground feature aggregation
        ca_vid_feats = [torch.einsum('ntd,nt->nd', [x_emb, att]) for att in class_agno_atts]
        # normalization
        norms_ca_vid_feats = [calculate_l1_norm(f) for f in ca_vid_feats]
        # calculate score
        ca_vid_scrs = [torch.einsum('nd,cd->nc', [f, norms_ac]).squeeze(-1) for f in norms_ca_vid_feats]
        ca_vid_scr = torch.stack(ca_vid_scrs, -1).mean(-1) * self.scale_factor * 2.0
        ca_vid_pred = F.softmax(ca_vid_scr, -1)

        # multiple instance learning branch
        # temporal score aggregation
        mid_vid_scrs = [torch.einsum('ntc,ntc->nc', [frm_scrs, att]) for att in class_wise_atts]
        mil_vid_scr = torch.stack(mid_vid_scrs, -1).mean(-1) * 2.0 # frm_scrs have been multiplied by the scale factor
        mil_vid_pred = F.softmax(mil_vid_scr, -1)

        return cw_vid_pred, ca_vid_pred, mil_vid_pred, frm_scrs


class FeatureEmbedding(nn.Module):
    def __init__(self, n_in, n_out, dropout):
        super().__init__()
        self.process = nn.Sequential(
            nn.Linear(n_in, n_in),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(n_in, n_out),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.process(x)
        return x





