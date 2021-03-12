import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class NormalizedCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, labels):
        new_labels = labels / (torch.sum(labels, dim=1, keepdim=True) + 1e-8)
        loss = -1.0 * torch.mean(torch.sum(Variable(new_labels) * torch.log(pred), dim=1), dim=0)
        return loss


class MILL_B(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, labels, device):
        new_labels = torch.cat([labels, torch.ones(pred.size(0), 1).to(device)], -1)
        new_labels = new_labels / (torch.sum(new_labels, dim=1, keepdim=True) + 1e-8)
        loss = -1.0 * torch.mean(torch.sum(Variable(new_labels) * torch.log(pred), dim=1), dim=0)
        return loss


class MILL_WB(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, labels, device):
        new_labels = torch.cat([labels, torch.zeros(pred.size(0), 1).to(device)], -1)
        new_labels = new_labels / (torch.sum(new_labels, dim=1, keepdim=True) + 1e-8)
        loss = -1.0 * torch.mean(torch.sum(Variable(new_labels) * torch.log(pred), dim=1), dim=0)
        return loss


class EquivalentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, frm_scores, labels, device):
        act_frm_scores = frm_scores[..., :-1]
        bg_frm_scores = frm_scores[..., -1]
        n, t, c = act_frm_scores.size()
        labels = labels / torch.sum(labels, dim=1, keepdim=True)

        fg_class_wise_att1 = F.softmax(act_frm_scores, dim=1)
        fg_class_wise_att2 = F.softmax(act_frm_scores * 5.0, dim=1)
        bg_class_wise_att1 = (1 - fg_class_wise_att1) / (t - 1)
        bg_class_wise_att2 = F.softmax(bg_frm_scores, dim=1)

        fg_bot_score = torch.einsum('ntc,ntc->nc', [frm_scores, fg_class_wise_att1])
        fg_top_score = torch.einsum('ntc,ntc->nc', [frm_scores, fg_class_wise_att2])
        bg_bot_score = torch.einsum('ntc,ntc->nc', [frm_scores, bg_class_wise_att1])
        bg_top_score = torch.einsum('ntc,nt->nc', [frm_scores, bg_class_wise_att2])

        #import pdb; pdb.set_trace()
        loss = torch.abs(fg_bot_score - fg_top_score) * labels + torch.abs(bg_bot_score - bg_top_score) * labels 
        loss = loss.sum(-1).mean(-1)

        return loss