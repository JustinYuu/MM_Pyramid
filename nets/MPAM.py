import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import pdb
from nets.Transformer import *


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=dilation, dilation=dilation)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, 1)

        self.net = nn.Sequential(self.conv1, self.relu1, self.conv2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class SemanticFusionModule(nn.Module):
    def __init__(self, num_inputs, ffn_dim, dropout=0.2, nhead=8):
        super(SemanticFusionModule, self).__init__()
        c = copy.deepcopy
        self.num_inputs = num_inputs
        self.multiheadattn = MultiHeadAttention(nhead, num_inputs)
        self.feedforward = PositionwiseFeedForward(num_inputs, ffn_dim, dropout=dropout)
        self.vadaptiveinteraction = TransformerLayer(num_inputs, c(self.multiheadattn), c(self.feedforward), dropout)
        self.aadaptiveinteraction = TransformerLayer(num_inputs, c(self.multiheadattn), c(self.feedforward), dropout)
        self.vselectfusion = nn.Linear(num_inputs, 1)
        self.aselectfusion = nn.Linear(num_inputs, 1)

    def forward(self, v_stage, a_stage):
        v_interact = self.vadaptiveinteraction(v_stage, v_stage, v_stage)  # [nbatch*10, stage_num, 512]
        a_interact = self.aadaptiveinteraction(a_stage, a_stage, a_stage)  # [nbatch*10, stage_num, 512]
        v_weight = torch.sigmoid(self.vselectfusion(v_interact))
        a_weight = torch.sigmoid(self.aselectfusion(a_interact))
        v_interact = v_interact.permute(0, 2, 1).contiguous()
        a_interact = a_interact.permute(0, 2, 1).contiguous()
        v_out = torch.bmm(v_interact, v_weight).view(-1, 10, self.num_inputs)
        a_out = torch.bmm(a_interact, a_weight).view(-1, 10, self.num_inputs)
        # v_out = torch.mean(v_stage, dim=1, keepdim=False).view(-1, 10, self.num_inputs)
        # a_out = torch.mean(a_stage, dim=1, keepdim=False).view(-1, 10, self.num_inputs)
        return v_out, a_out


class SemanticCaptureModule(nn.Module):
    def __init__(self, num_inputs, ffn_dim, num_channels, kernel_size=3, dropout=0.2, nhead=8):
        super(SemanticCaptureModule, self).__init__()
        v_layers = []
        a_layers = []
        msa_layers = []
        num_levels = len(num_channels)
        c = copy.deepcopy
        self.multiheadattn = MultiHeadAttention(nhead, num_inputs)
        self.feedforward = PositionwiseFeedForward(num_inputs, ffn_dim, dropout=dropout)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            v_layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size)]
            a_layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size)]

            msa_layers += [IntegrateAttentionBlock(TransformerLayer(num_inputs,
                                                                    MultiHeadAttention(nhead, num_inputs,
                                                                                       masksize=dilation_size * 2),
                                                                    c(self.feedforward), dropout),
                                                   TransformerLayer(num_inputs,
                                                                    MultiHeadAttention(nhead, num_inputs,
                                                                                       masksize=dilation_size * 2),
                                                                    c(self.feedforward), dropout),
                                                   TransformerLayer(num_inputs,
                                                                    MultiHeadAttention(nhead, num_inputs,
                                                                                       masksize=dilation_size * 2),
                                                                    c(self.feedforward), dropout), num_inputs
                                                   )]
        self.vnetwork = nn.Sequential(*v_layers)
        self.anetwork = nn.Sequential(*a_layers)
        self.msa = nn.Sequential(*msa_layers)

    def forward(self, v, a):
        # v: [nbatch, 10, 512] a: [nbatch, 10, 512]
        v_stage_list = []
        a_stage_list = []
        v_stage = v
        a_stage = a
        for i in range(len(self.vnetwork)):
            v_msa, a_msa = self.msa[i](v_stage, a_stage)
            v_msa = v_msa.permute(0, 2, 1).contiguous()  # [nbatch, 512, 10]
            a_msa = a_msa.permute(0, 2, 1).contiguous()  # [nbatch, 512, 10]
            v_tcn = self.vnetwork[i](v_msa)
            a_tcn = self.anetwork[i](a_msa)
            v_stage = v_tcn.permute(0, 2, 1).contiguous()  # [nbatch, 10, 512]
            a_stage = a_tcn.permute(0, 2, 1).contiguous()  # [nbatch, 10, 512]
            v_stage_list.append(v_stage)
            a_stage_list.append(a_stage)
        v_stage = torch.stack(v_stage_list, dim=2)  # [nbatch, 10, stage_num, 512]
        a_stage = torch.stack(a_stage_list, dim=2)  # [nbatch, 10, stage_num, 512]]
        v_stage = v_stage.view(-1, v_stage.size(2), v_stage.size(3))
        a_stage = a_stage.view(-1, a_stage.size(2), a_stage.size(3))
        return v_stage, a_stage


class MultimodalPyramidAttentionalModule(nn.Module):
    def __init__(self, num_inputs, ffn_dim, num_channels, kernel_size=3, dropout=0.2, nhead=8):
        super(MultimodalPyramidAttentionalModule, self).__init__()
        self.capture = SemanticCaptureModule(num_inputs, ffn_dim, num_channels, kernel_size, dropout, nhead)
        self.fusion = SemanticFusionModule(num_inputs, ffn_dim, dropout, nhead)

    def forward(self, video, audio):
        v_cap, a_cap = self.capture(video, audio)
        v_out, a_out = self.fusion(v_cap, a_cap)
        return v_out, a_out
