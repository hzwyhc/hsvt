# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F

from siamban.core.config import cfg
from siamban.models.backbone import get_backbone
from siamban.models.head import get_ban_head
from siamban.models.neck import get_neck


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()
        self.bbox = bbox = None

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build ban head
        if cfg.BAN.BAN:
            self.head = get_ban_head(cfg.BAN.TYPE,
                                     **cfg.BAN.KWARGS)

    def template(self, z, bbox):
        self.bbox = bbox
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf
        self.head.init(zf, bbox)

    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.head.track(xf)
        return {
                'cls': cls,
                'loc': loc
               }

    def forward(self, data, epoch):
        """ only used in training
        """
        z_hsi = data['template'].cuda()
        x_hsi = data['search'].cuda()
        z_rgb = data['template_rgb'].cuda()
        x_rgb = data['search_image_rgb'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        target_box = data['target_box'].cuda()

        # z_rgb = self.generate_prompt(z_rgb, z_hsi)
        # x_rgb = self.generate_prompt(x_rgb, x_hsi)
        z_rgb = self.backbone(z_rgb)  # 32 256 15 15
        x_rgb = self.backbone(x_rgb)  # 32 256 31 31
        zf_rgb = self.neck(z_rgb)  # 32 256 7 7
        xf_rgb = self.neck(x_rgb)  # 32 256 7 7
        #
        #
        # z_hsi = self.backbone_hsi(z_hsi)  # 32 256 18 18
        # x_hsi = self.backbone_hsi(x_hsi)  # 32 256 34 34
        # zf_hsi = self.neck_hsi(z_hsi)  # 32 256 7 7
        # xf_hsi = self.neck_hsi(x_hsi)  # 32 256 7 7


        # zf = self.fusion(zf_rgb, zf_hsi)  # transformer fusion
        # xf = self.fusion(xf_rgb, xf_hsi)  # transformer fusion

        # cls_hsi, loc_hsi, sa = self.head_hsi(zf, xf, label_cls)  # 28 2 25 25
        cls, loc = self.head(zf_rgb, xf_rgb, label_cls)  # 28 2 25 25

        # distill_loss = self.mse(loc, loc_rgb) #+ self.mse(cls, cls_rgb)

        # weight = F.softmax(self.weight, 0)
        # cls = weight[0] * cls + weight[1] * cls_hsi


        cls = self.log_softmax(cls)

        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = select_iou_loss(loc, label_loc, label_cls)
        if self.use_sam:
            sa_loss = select_sa_loss(sa, label_cls)
            # ssa_loss = select_sa_loss(ssa, label_cls)


        outputs = {}
        if self.use_sam:
            outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                                    cfg.TRAIN.LOC_WEIGHT * loc_loss + sa_loss
            outputs['cls_loss'] = cls_loss
            outputs['loc_loss'] = loc_loss
            outputs['sa_loss'] = sa_loss
        else:
            outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                                    cfg.TRAIN.LOC_WEIGHT * loc_loss# + 0.5 * sa_loss
            outputs['cls_loss'] = cls_loss
            outputs['loc_loss'] = loc_loss
            # outputs['sa_loss'] = sa_loss
        return outputs
