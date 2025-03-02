import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale, build_activation_layer, build_norm_layer
import torch.nn.functional as F
from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead
from mmengine.model import BaseModule, ModuleList, Sequential
from typing import Dict, List
from torch import Tensor
from mmseg.utils import OptConfigType, SampleList
from typing import Optional, Tuple, Union
from mmseg.models.losses import accuracy

class BFEM(nn.Module):
    def __init__(self, channels, conv_cfg, norm_cfg, act_cfg):
        super(BFEM, self).__init__()
        self.conv1 = ConvModule(channels,
                channels,
                3,
                padding = 1,
                conv_cfg = conv_cfg,
                norm_cfg = norm_cfg,
                act_cfg = act_cfg,
                inplace = False)

        self.flow_make = nn.Conv2d(channels * 2, 2, kernel_size=3, padding=1, bias=False)


    def forward(self, p4, q1):
        size = q1.size()[2:]
        p4 = self.conv1(p4)
        p4 = resize(p4, size, mode='bilinear', align_corners=False)
        flow = self.flow_make(torch.cat([q1, p4], dim=1))

        seg_flow_warp = self.flow_warp(q1, flow, size)
        seg_edge = q1 - seg_flow_warp

        return seg_edge

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output

class FFM(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(FFM, self).__init__()

        self.conv = nn.Sequential(
            BatchNorm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False)
        )

    def forward(self, f1, f2, bd):
        edge_att = torch.sigmoid(bd)
        return self.conv(edge_att * f2 + (1 - edge_att) * f1)

class ASCM_stage1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn


class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = ASCM_stage1(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class DAPPM(BaseModule):

    def __init__(self,
                 in_channels: int,
                 branch_channels: int,
                 out_channels: int,
                 num_scales: int,
                 kernel_sizes: List[int] = [5, 9, 17],
                 strides: List[int] = [2, 4, 8],
                 paddings: List[int] = [2, 4, 8],
                 norm_cfg: Dict = dict(type='BN', momentum=0.1),
                 act_cfg: Dict = dict(type='ReLU', inplace=True),
                 conv_cfg: Dict = dict(
                     order=('norm', 'act', 'conv'), bias=False),
                 upsample_mode: str = 'bilinear'):
        super().__init__()

        self.num_scales = num_scales
        self.unsample_mode = upsample_mode
        self.in_channels = in_channels
        self.branch_channels = branch_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.conv_cfg = conv_cfg

        self.scales = ModuleList([
            ConvModule(
                in_channels,
                branch_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                **conv_cfg)
        ])
        for i in range(1, num_scales - 1):
            self.scales.append(
                Sequential(*[
                    nn.AvgPool2d(
                        kernel_size=kernel_sizes[i - 1],
                        stride=strides[i - 1],
                        padding=paddings[i - 1]),
                    ConvModule(
                        in_channels,
                        branch_channels,
                        kernel_size=1,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        **conv_cfg)
                ]))
        self.scales.append(
            Sequential(*[
                nn.AdaptiveAvgPool2d((1, 1)),
                ConvModule(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **conv_cfg)
            ]))
        self.processes = ModuleList()
        for i in range(num_scales - 1):
            self.processes.append(
                ConvModule(
                    branch_channels,
                    branch_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **conv_cfg))

        self.compression = ConvModule(
            branch_channels * num_scales,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **conv_cfg)

        self.shortcut = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **conv_cfg)

    def forward(self, inputs: Tensor):
        feats = []
        feats.append(self.scales[0](inputs))

        for i in range(1, self.num_scales):
            feat_up = F.interpolate(
                self.scales[i](inputs),
                size=inputs.shape[2:],
                mode=self.unsample_mode)
            feats.append(self.processes[i - 1](feat_up + feats[i - 1]))

        return self.compression(torch.cat(feats,
                                          dim=1)) + self.shortcut(inputs)


class ChannelAtt(nn.Module):
    def __init__(self, in_channels, out_channels, conv_cfg, norm_cfg, act_cfg):
        super(ChannelAtt, self).__init__()
        self.conv_bn_relu = ConvModule(in_channels, out_channels, 3, stride=1, padding=1, conv_cfg=conv_cfg,
                                       norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv_1x1 = ConvModule(out_channels, out_channels, 1, stride=1, padding=0, conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x, fre=False):
        """Forward function."""
        feat = self.conv_bn_relu(x)
        if fre:
            h, w = feat.size()[2:]
            h_tv = torch.pow(feat[..., 1:, :] - feat[..., :h - 1, :], 2)
            w_tv = torch.pow(feat[..., 1:] - feat[..., :w - 1], 2)
            atten = torch.mean(h_tv, dim=(2, 3), keepdim=True) + torch.mean(w_tv, dim=(2, 3), keepdim=True)
        else:
            atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv_1x1(atten)
        return feat, atten


class SSRM(nn.Module):
    def __init__(self, channels, conv_cfg, norm_cfg, act_cfg, ext=1, r=16):
        super(SSRM, self).__init__()
        self.r = r
        self.g1 = nn.Parameter(torch.zeros(1))
        self.g2 = nn.Parameter(torch.zeros(1))
        #
        self.spatial_mlp = nn.Sequential(nn.Linear(channels , channels), nn.ReLU(), nn.Linear(channels, channels))
        self.spatial_att = ChannelAtt(channels * ext, channels, conv_cfg, norm_cfg, act_cfg)
        #
        self.context_mlp = nn.Sequential(*[nn.Linear(channels , channels), nn.ReLU(), nn.Linear(channels, channels)])
        self.context_att = ChannelAtt(channels, channels, conv_cfg, norm_cfg, act_cfg)
        self.context_head = ConvModule(channels, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                       act_cfg=act_cfg)
        self.smooth = ConvModule(channels, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                 act_cfg=None)

    def forward(self, sp_feat, co_feat):
        # **_att: B x C x 1 x 1
        s_feat, s_att = self.spatial_att(sp_feat)
        c_feat, c_att = self.context_att(co_feat)
        b, c, h, w = s_att.size()
        s_att_split = s_att.view(b, self.r, c // self.r)
        c_att_split = c_att.view(b, self.r, c // self.r)
        chl_affinity = torch.bmm(s_att_split, c_att_split.permute(0, 2, 1))
        chl_affinity = chl_affinity.view(b, -1)
        sp_mlp_out = F.relu(self.spatial_mlp(chl_affinity))
        co_mlp_out = F.relu(self.context_mlp(chl_affinity))
        re_s_att = torch.sigmoid(s_att + self.g1 * sp_mlp_out.unsqueeze(-1).unsqueeze(-1))
        re_c_att = torch.sigmoid(c_att + self.g2 * co_mlp_out.unsqueeze(-1).unsqueeze(-1))
        c_feat = torch.mul(c_feat, re_c_att)
        s_feat = torch.mul(s_feat, re_s_att)
        c_feat = F.interpolate(c_feat, s_feat.size()[2:], mode='bilinear', align_corners=False)
        c_feat = self.context_head(c_feat)
        out = self.smooth(s_feat + c_feat)
        return out


class ASCM_stage2(nn.Module):
    def __init__(self, channels, r=4):
        super(ASCM_stage2, self).__init__()
        inter_channels = int(channels // r)

        # bottleneck
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # GAP + bottleneck
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg

        # attention map
        wei = self.sigmoid(xlg)

        xo = 2* x * wei + 2 * residual * (1 - wei)
        return xo


@MODELS.register_module()
class MSCSCHead(BaseDecodeHead):

    def __init__(self,  **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        # PSP Module
        self.psp_modules = DAPPM(
            self.in_channels[-1],
            self.channels,
            self.channels,
            num_scales=5)

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        # self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)

            self.f1_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.f2_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.bd_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)

        # sk module
        self.SK0 = ASCM_stage1(self.channels)
        self.SK1 = ASCM_stage1(self.channels)
        self.SK2 = ASCM_stage1(self.channels)
        # self.SK3 = ASCM_stage1(self.channels)

        # edge module
        self.edge_module = BFEM(self.channels, self.conv_cfg, self.norm_cfg, self.act_cfg)

        # lateral feature fusion
        self.LFusion = []
        self.LFusion0 = ASCM_stage2(self.channels)
        self.LFusion.append(self.LFusion0)

        self.LFusion1 = ASCM_stage2(self.channels)
        self.LFusion.append(self.LFusion1)

        self.LFusion2 = ASCM_stage2(self.channels)
        self.LFusion.append(self.LFusion2)

        # multi-scale feature fusion
        self.MSFuse0 = SSRM(self.channels, self.conv_cfg, self.norm_cfg, self.act_cfg)
        self.MSFuse1 = SSRM(self.channels, self.conv_cfg, self.norm_cfg, self.act_cfg)
        self.MSFuse2 = SSRM(self.channels, self.conv_cfg, self.norm_cfg, self.act_cfg)

        # double FPN
        self.extra_l_conv = ConvModule(
                self.in_channels[-1],
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)

        # FPN Fusion
        self.fpn_fusion = FFM(self.channels, self.channels)

        self.sem_cls_seg = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)
        self.bd_cls_seg = nn.Conv2d(self.channels, 1, kernel_size=1)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        output = self.psp_modules(x)

        return output

    def forward(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)


        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals.append(self.psp_forward(inputs))
        laterals[0] = self.SK0(laterals[0])
        laterals[1] = self.SK1(laterals[1])
        laterals[2] = self.SK2(laterals[2])
        # laterals[3] = self.SK3(laterals[3])

        # rebuild top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = self.LFusion[i - 1](laterals[i - 1],
                                                  resize(laterals[i],
                                                         size=prev_shape,
                                                         mode='bilinear',
                                                         align_corners=self.align_corners))

        # feature_p = []
        feature_p1 = self.extra_l_conv(inputs[-1])
        feature_p2 = self.MSFuse0(laterals[2], feature_p1)
        feature_p3 = self.MSFuse1(laterals[1], feature_p2)
        feature_p4 = self.MSFuse2(laterals[0], feature_p3)
        # edge feature

        feature_bd = self.bd_conv(
            self.edge_module(laterals[-1], feature_p4)
        )

        feats1 = self.f1_conv(laterals[0])
        feats2 = self.f2_conv(feature_p4)

        feats = self.fpn_fusion(feats1, feats2, feature_bd)
        output = self.cls_seg(feats)
        fp1_output = self.sem_cls_seg(feats1)
        bd_output = self.bd_cls_seg(feature_bd)
        if self.training:
            return output, fp1_output, bd_output
        else:
            return output

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tuple[Tensor]:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        gt_edge_segs = [
            data_sample.gt_edge_map.data for data_sample in batch_data_samples
        ]
        gt_sem_segs = torch.stack(gt_semantic_segs, dim=0)
        gt_edge_segs = torch.stack(gt_edge_segs, dim=0)
        return gt_sem_segs, gt_edge_segs

    def loss_by_feat(self, seg_logits: Tuple[Tensor],
                     batch_data_samples: SampleList) -> dict:
        loss = dict()
        out_logit, fp1_logit, bd_logit = seg_logits
        sem_label, bd_label = self._stack_batch_gt(batch_data_samples)
        out_logit = resize(
            input=out_logit,
            size=sem_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        fp1_logit = resize(
            input=fp1_logit,
            size=sem_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        bd_logit = resize(
            input=bd_logit,
            size=bd_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        sem_label = sem_label.squeeze(1)
        bd_label = bd_label.squeeze(1)

        loss['loss_sem'] = self.loss_decode[0](out_logit, sem_label)
        loss['loss_sem_fp1'] = self.loss_decode[1](fp1_logit, sem_label, ignore_index=self.ignore_index)
        loss['loss_bd'] = self.loss_decode[2](bd_logit, bd_label)

        filler = torch.ones_like(sem_label) * self.ignore_index
        sem_bd_label = torch.where(
            torch.sigmoid(bd_logit[:, 0, :, :]) > 0.8, sem_label, filler)
        loss['loss_sem_bd'] = self.loss_decode[3](out_logit, sem_bd_label)

        loss['acc_seg'] = accuracy(out_logit, sem_label, ignore_index=self.ignore_index)

        return loss
