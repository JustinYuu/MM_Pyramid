import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from nets.MPAM import MultimodalPyramidAttentionalModule as MMP
from nets.Transformer import *


class AVVPNet(nn.Module):
    def __init__(self, head, hidden_size, ffn_dim, n_channels):
        super(AVVPNet, self).__init__()

        self.fc_prob = nn.Linear(512, 25)
        self.fc_frame_att = nn.Linear(512, 25)
        self.fc_av_att = nn.Linear(512, 25)
        self.fc_a = nn.Linear(128, 512)
        self.fc_v = nn.Linear(2048, 512)
        self.fc_st = nn.Linear(512, 512)
        self.fc_fusion = nn.Linear(1024, 512)
        self.MMP = MMP(hidden_size, ffn_dim, n_channels, nhead=head)

    def forward(self, audio, visual, visual_st):
        f_a = self.fc_a(audio)
        # 2d and 3d visual feature
        vid_s = self.fc_v(visual).permute(0, 2, 1).unsqueeze(-1)
        vid_s = F.avg_pool2d(vid_s, (8, 1)).squeeze(-1).permute(0, 2, 1)
        vid_st = self.fc_st(visual_st)
        f_v = torch.cat((vid_s, vid_st), dim=-1)
        f_v = self.fc_fusion(f_v)

        v_o, a_o = self.MMP(f_v, f_a)
        # x = x.view(x.size(0), x.size(1), 2, -1)
        # prediction
        x = torch.cat([v_o.unsqueeze(-2), a_o.unsqueeze(-2)], dim=-2)
        frame_prob = torch.sigmoid(self.fc_prob(x))

        # attentive MMIL pooling
        frame_att = torch.softmax(self.fc_frame_att(x), dim=1)
        av_att = torch.softmax(self.fc_av_att(x), dim=2)
        temporal_prob = (frame_att * frame_prob)
        global_prob = (temporal_prob * av_att).sum(dim=2).sum(dim=1)

        a_prob = temporal_prob[:, :, 0, :].sum(dim=1)
        v_prob = temporal_prob[:, :, 1, :].sum(dim=1)

        return global_prob, a_prob, v_prob, frame_prob