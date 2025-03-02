# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import os.path as osp
from mmseg.models.utils import resize
from mmseg.models import builder
from mmseg.models import build_segmentor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from mmseg.models.segmentors.base import BaseSegmentor

@MODELS.register_module()
class EncoderDecoder_LOCAL8x8(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 fuse_head = None,
                 train_cfg=None,
                 test_cfg=None,
                 global_cfg=None,
                 pretrained=None):
        super(EncoderDecoder_LOCAL8x8, self).__init__()
        self.global_cfg = global_cfg
        self.global_model = build_segmentor(global_cfg.model, train_cfg=global_cfg.train_cfg, test_cfg=global_cfg.test_cfg)

        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        self._init_fuse_head(fuse_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

        self.global_model.eval()

        for k,v in self.global_model.named_parameters():
            v.requires_grad = False

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def _init_fuse_head(self, fuse_head):
        """Initialize ``fuse_head``"""
        self.fuse_head = builder.build_head(fuse_head)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        super(EncoderDecoder_LOCAL8x8, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

        print('Loading Global Model=======> '+self.global_cfg.global_model_path)
        if not osp.isfile(self.global_cfg.global_model_path):
            raise RuntimeError("========> no checkpoint found at '{}'".format(self.global_cfg.global_model_path))
        global_model_dict = torch.load(self.global_cfg.global_model_path, map_location='cpu')
        self.global_model.load_state_dict(global_model_dict['state_dict'])

    def extract_feat(self, img):   # 使用局部网络backbone提取特征，返回的x为(p6, p12, p18, p24) shape=(1,256,32,32)共4个
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone into a tuple list."""
        x = self.extract_feat(img)
        return x

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        # 使用局部网络的解码头，返回的是解码头损失与局部特征x,x是(p6, p12, p18, p24)经过降为到128再cat在降为，x的shape=(1,128,512,512)
        """Run forward function and calculate loss for decode head in training.
           Generate the LOCAL FEATURE
        """
        losses = dict()
        loss_decode, local_features = self.decode_head.forward_train_with_local_features(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses, local_features

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits, local_feature = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits,local_feature

    def _fuse_head_forward_test(self, local_features, global_features):
        """Run forward function and calculate loss for fuse head in
                inference."""
        concat_features = torch.cat((local_features, global_features), dim=1)
        fuse_logits, _ = self.fuse_head.fuse_forward_test(concat_features)
        return fuse_logits

    def _fuse_features_forward_train(self, local_features, global_features):
        """Run forward function and calculate loss for fuse head in
                inference."""
        concat_features = torch.cat((local_features, global_features), dim=1)
        _, fuse_features = self.fuse_head.fuse_forward_test(concat_features)
        return fuse_features

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def _fuse_head_forward_train(self, local_features, global_features, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_fuse = self.fuse_head.fuse_forward_train(local_features, global_features, gt_semantic_seg)

        losses.update(add_prefix(loss_fuse, 'fuse_edge'))
        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img, img_metas, gt_semantic_seg, num_classes):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        self.global_model.eval()
        with torch.no_grad():
            global_features = self.global_model.inference_global_feature(img).detach()
            #获得全局网络backbone和decode_head提取的特征，shape=(1,128,512,512)
        batch_size, _, h_img, w_img = img.size()
        h_encode = w_encode = int (20 * (h_img/self.backbone.img_size))
        self.h_crop = self.w_crop = self.h_stride = self.w_stride = self.backbone.img_size
        h_grids = max(h_img - self.h_crop + self.h_stride - 1, 0) // self.h_stride + 1
        w_grids = max(w_img - self.w_crop + self.w_stride - 1, 0) // self.w_stride + 1
        preds1 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds2 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds3 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds4 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        # preds5 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        # preds6 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        # preds7 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        # preds8 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        count_mat = img.new_zeros((batch_size, num_classes, h_encode, w_encode))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * self.h_stride
                x1 = w_idx * self.w_stride
                y2 = min(y1 + self.h_crop, h_img)
                x2 = min(x1 + self.w_crop, w_img)
                y1 = max(y2 - self.h_crop, 0)
                x1 = max(x2 - self.w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                # 对于每个网格位置，从输入图像中裁剪一个块（crop_img），根据指定的裁剪参数。
                crop_seg_logit = self.extract_feat(crop_img)
                # 调用self.extract_feat(crop_img)函数来提取裁剪图像的特征。
                # self.extract_feat()包含了backbone，也就是my_vit_local，会返回4个特征图(p6, p12, p18, p24),shape=(1,256,32,32)
                preds1[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[0]
                preds2[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[1]
                preds3[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[2]
                preds4[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[3]
                # preds5[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[4]
                # preds6[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[5]
                # preds7[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[6]
                # preds8[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[7]
                count_mat[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] += 1 # 更新count_mat张量，以计算每个空间位置被裁剪图像块覆盖的次数。
        assert (count_mat == 0).sum() == 0 # 处理完所有网格位置后，检查count_mat张量，确保所有位置至少被覆盖一次。

        x = (preds1, preds2, preds3, preds4) #shape=(1,256,64,64) 4个
        losses = dict()
        loss_decode, local_features = self._decode_head_forward_train(x, img_metas,gt_semantic_seg)
        # 接着计算模型的主要损失，通过调用_decode_head_forward_train函数，并将返回的损失更新到losses字典中。 local_features的shape=(1,128,1024,1024)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, img_metas, gt_semantic_seg)
            # 如果模型具有辅助头（用于辅助预测的附加分支），则计算辅助损失，并将其添加到losses字典中。
            losses.update(loss_aux)

        fuse_features = torch.cat((F.interpolate(global_features,size=(1024,1024),mode='bilinear'),local_features),dim=1)
        fuse_loss = self._fuse_head_forward_train(fuse_features, gt_semantic_seg)
        # 使用_fuse_head_forward_train计算局部特征和全局特征之间的融合损失，并将其添加到losses字典中。
        losses.update(fuse_loss)
        return losses

    def inference_global_local_feature(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        batch_size, _, h_img, w_img = img.size()
        global_features = self.global_model.slide_inference_global_features(img, img_meta, rescale)
        # 调用 self.global_model.slide_inference_global_features(img, img_meta, rescale) 函数，使用滑动窗口进行全局特征推理。
        # img_meta 是包含图像信息的元数据列表，rescale 是图像的缩放比例。
        h_stride, w_stride = self.test_cfg.stride # 通过解码头模块获取滑动窗口的步长和裁剪尺寸
        h_crop, w_crop = self.test_cfg.crop_size
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1 # 计算网格的数量，用于在图像上滑动窗口。
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        h_encode = int(h_img / 8)
        w_encode = int(w_img / 8)
        # 初始化一些张量 (preds1、preds2、preds3、preds4 和 count_mat) 用于存储中间结果和计数。
        preds1 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds2 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds3 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds4 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        count_mat = img.new_zeros((batch_size, 1, h_encode, w_encode))
        # 通过两个嵌套循环，在每个网格位置上滑动窗口进行推理。
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2] # 根据当前网格位置计算裁剪图像的范围，并将其存储为 crop_img。
                crop_seg_logit = self.encode_decode(crop_img, img_meta) # 调用 self.encode_decode(crop_img, img_meta) 函数，对裁剪图像进行编码和解码操作，得到 crop_seg_logit，包含裁剪图像的预测结果。
                preds1[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[0]
                preds2[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[1]
                preds3[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[2]
                preds4[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[3]
                count_mat[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] += 1 # 更新 count_mat 张量，记录每个位置被裁剪图像覆盖的次数。
        assert (count_mat == 0).sum() == 0
        x = (preds1, preds2, preds3, preds4)
        # 构建一个元组 x，包含 preds1、preds2、preds3、preds4 张量，用于调用 _decode_head_forward_test 函数。
        local_outs, local_features = self._decode_head_forward_test(x,img_meta)
        # 调用 _decode_head_forward_test(x, img_meta) 函数，进行解码头的测试推理，并获取局部输出和局部特征。
        fuse_features = torch.cat((F.interpolate(global_features, size=(1024, 1024), mode='bilinear'), local_features),dim=1)
        fuse_logits = self._fuse_head_forward_test(fuse_features)
        # 调用 _fuse_head_forward_test(local_features, global_features) 函数，将局部特征和全局特征融合，并得到融合后的预测结果 fuse_logits。

        return fuse_logits # 返回融合后的预测结果 fuse_logits。这个结果将包含对整个图像的语义分割预测。

    def inference_global_local_feature_with_fuse_feature(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        batch_size, _, h_img, w_img = img.size()
        global_features = self.global_model.slide_inference_global_features(img, img_meta, rescale)
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        h_encode = int(h_img / 8)
        w_encode = int(w_img / 8)
        preds1 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds2 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds3 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds4 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        count_mat = img.new_zeros((batch_size, 1, h_encode, w_encode))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds1[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[0]
                preds2[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[1]
                preds3[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[2]
                preds4[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[3]
                count_mat[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] += 1
        assert (count_mat == 0).sum() == 0
        x = (preds1, preds2, preds3, preds4)
        local_outs, local_features = self._decode_head_forward_test(x,img_meta)

        fused_features = torch.cat((F.interpolate(global_features, size=(1024, 1024), mode='bilinear'), local_features),dim=1)
        fuse_features = self._fuse_features_forward_train(fused_features)
        return fuse_features

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        batch_size, _, h_img, w_img = img.size()
        img_crop = img[:, :, 0:h_img - 1, 0:w_img - 1]   # for BSDS500
        #img_crop = img[:, :, 0:h_img - 1, :]   # for NYUD
        batch_size, _, h_crop_img, w_crop_img = img_crop.size()

        global_features_crop = self.global_model.slide_inference_global_features(img_crop, img_meta, rescale).detach()

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        num_classes = self.num_classes
        h_grids = max(h_crop_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_crop_img - w_crop + w_stride - 1, 0) // w_stride + 1

        h_encode = int(h_crop_img / 8)
        w_encode = int(w_crop_img / 8)
        preds1 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds2 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds3 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds4 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        # preds5 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        # preds6 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        # preds7 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        # preds8 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        count_mat = img.new_zeros((batch_size, 1, h_encode, w_encode))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_crop_img)
                x2 = min(x1 + w_crop, w_crop_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds1[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[0].data
                preds2[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[1].data
                preds3[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[2].data
                preds4[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[3].data
                # preds5[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[4].data
                # preds6[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[5].data
                # preds7[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[6].data
                # preds8[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[7].data
                count_mat[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(count_mat.cpu().detach().numpy()).to(device=img.device)
        x_crop = (preds1, preds2, preds3, preds4)
        local_outs_crop, local_features_crop = self._decode_head_forward_test(x_crop,img_meta)
        fuse_features_crops = torch.cat((F.interpolate(global_features_crop, size=(1024, 1024), mode='bilinear'), local_features_crop),dim=1)
        fuse_outs_crop = self._fuse_head_forward_test(fuse_features_crops)

        fuse_outs = torch.zeros((batch_size,num_classes, h_img, w_img))
        fuse_outs[:, :, 0:h_img - 1, 0:w_img - 1] = fuse_outs_crop    # for BSDS500
        #fuse_outs[:, :, 0:h_img - 1, :] = fuse_outs_crop    # for NYUD
        '''
        if rescale:
            fuse_outs = resize(
                fuse_outs,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        '''
        return fuse_outs

    def slide_inference2(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        batch_size, _, h_img, w_img = img.size()

        global_features = self.global_model.slide_inference_global_features(img,img_meta, rescale)

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        h_encode = int(h_img / 8)
        w_encode = int(w_img / 8)
        preds1 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds2 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds3 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds4 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        # preds5 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        # preds6 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        # preds7 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        # preds8 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        count_mat = img.new_zeros((batch_size, 1, h_encode, w_encode))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds1[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[0].data
                preds2[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[1].data
                preds3[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[2].data
                preds4[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[3].data
                # preds5[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[4].data
                # preds6[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[5].data
                # preds7[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[6].data
                # preds8[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[7].data
                count_mat[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(count_mat.cpu().detach().numpy()).to(device=img.device)
        x = (preds1, preds2, preds3, preds4)
        local_outs, local_features = self._decode_head_forward_test(x,img_meta)
        fuse_features = torch.cat((F.interpolate(global_features, size=(1024, 1024), mode='bilinear'), local_features),
                                  dim=1)
        fuse_outs = self._fuse_head_forward_test(fuse_features)
        return fuse_outs

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            seg_logit = resize(
                seg_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)

        '''
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))
        '''
        return seg_logit

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_pred = self.inference(img, img_meta, rescale)
        #seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        #seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        assert rescale
        batch_size, _, h_img, w_img = imgs[0].size()
        seg_logit = torch.zeros([batch_size, 1, h_img, w_img]).cuda()
        img0_crop = imgs[0][:, :, 0:h_img - 1, 0:w_img - 1]
        img0_crop_seg_logit = self.slide_inference2(img0_crop, img_metas[0], rescale)
        seg_logit[:, :, 0:h_img - 1, 0:w_img - 1] = img0_crop_seg_logit
        for i in range(1, len(imgs)):
            #img_cur = imgs[i]
            cur_seg_logit = self.slide_aug_test(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit

        seg_logit /= len(imgs)

        seg_pred = seg_logit.cpu().numpy()
        # unravel batch dim
        #seg_pred = list(seg_pred)
        return seg_pred

    def slide_aug_test(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        batch_size, _, h_img, w_img = img.size()
        if h_img < w_img:
            h_crop = 320
            w_crop =480
            h_stride = 300
            w_stride = 400
        if h_img > w_img:
            h_crop = 480
            w_crop =320
            h_stride = 400
            w_stride = 300
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.slide_inference2(crop_img, img_meta, None)
                preds += F.pad(crop_seg_logit,
                              (int(x1), int(preds.shape[3] - x2), int(y1),
                               int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(count_mat.cpu().detach().numpy()).to(device=img.device)

        preds = preds / count_mat
        # '''
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        # '''
        return preds


if __name__ == '__main__':
    model = EncoderDecoder_LOCAL8x8()
    dummy_input = torch.rand(1, 3, 1024, 1024)
    output = model(dummy_input)
    for out in output:
        print(out.size())