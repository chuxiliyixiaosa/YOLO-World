# Copyright (c) Lin Song. All rights reserved.
import math
from typing import List, Optional, Tuple, Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import copy
import mmcv
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from mmengine.dist import get_dist_info
from mmengine.structures import InstanceData
from mmdet.structures import SampleList
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig, InstanceList)
from mmdet.models.utils import multi_apply, unpack_gt_instances, filter_scores_and_topk
from mmyolo.models.dense_heads import YOLOv8HeadModule
from mmyolo.models.utils import gt_instances_preprocess
from mmyolo.registry import MODELS, TASK_UTILS
from mmyolo.models.dense_heads.yolov5_ins_head import (
    ProtoModule, YOLOv5InsHead
)

from .yolo_world_head import ContrastiveHead, BNContrastiveHead


@MODELS.register_module()
class YOLOWorldSegHeadModule(YOLOv8HeadModule):
    def __init__(self,
                 *args,
                 embed_dims: int,
                 proto_channels: int,
                 mask_channels: int,
                 freeze_bbox: bool = False,
                 freeze_all: bool = False,
                 use_bn_head: bool = False,
                 **kwargs) -> None:
        self.embed_dims = embed_dims
        self.proto_channels = proto_channels
        self.mask_channels = mask_channels
        self.freeze_bbox = freeze_bbox
        self.freeze_all = freeze_all
        self.use_bn_head = use_bn_head
        super().__init__(*args, **kwargs)

    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        super().init_weights()
        for cls_pred, cls_contrast, stride in zip(self.cls_preds,
                                                  self.cls_contrasts,
                                                  self.featmap_strides):
            cls_pred[-1].bias.data[:] = 0.0  # reset bias
            if hasattr(cls_contrast, 'bias'):
                nn.init.constant_(
                    cls_contrast.bias.data,
                    math.log(5 / self.num_classes / (640 / stride)**2))

    def _init_layers(self) -> None:
        """initialize conv layers in YOLOv8 head."""
        # Init decouple head
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.seg_preds = nn.ModuleList()
        self.cls_contrasts = nn.ModuleList()

        reg_out_channels = max(
            (16, self.in_channels[0] // 4, self.reg_max * 4))
        seg_out_channels = max(self.in_channels[0] // 4, self.mask_channels)
        cls_out_channels = max(self.in_channels[0], self.num_classes)

        bbox_norm_cfg = self.norm_cfg
        bbox_norm_cfg['requires_grad'] = not self.freeze_bbox
        if self.freeze_all:
            self.norm_cfg['requires_grad'] = False
            bbox_norm_cfg['requires_grad'] = False

        for i in range(self.num_levels):
            self.reg_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=bbox_norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=reg_out_channels,
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=bbox_norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=reg_out_channels,
                              out_channels=4 * self.reg_max,
                              kernel_size=1)))
            self.cls_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=bbox_norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=cls_out_channels,
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=bbox_norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=cls_out_channels,
                              out_channels=self.embed_dims,
                              kernel_size=1)))
            self.seg_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=seg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=seg_out_channels,
                               out_channels=seg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=seg_out_channels,
                              out_channels=self.mask_channels,
                              kernel_size=1)))

            if self.use_bn_head:
                self.cls_contrasts.append(
                    BNContrastiveHead(self.embed_dims, self.norm_cfg))
            else:
                self.cls_contrasts.append(ContrastiveHead(self.embed_dims))

        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('proj', proj, persistent=False)

        self.proto_pred = ProtoModule(in_channels=self.in_channels[0],
                                      middle_channels=self.proto_channels,
                                      mask_channels=self.mask_channels,
                                      norm_cfg=self.norm_cfg,
                                      act_cfg=self.act_cfg)
        if self.freeze_bbox or self.freeze_bbox:
            self._freeze_all()

    def _freeze_all(self):
        frozen_list = [self.cls_preds, self.reg_preds, self.cls_contrasts]
        if self.freeze_all:
            frozen_list.extend([self.proto_pred, self.seg_preds])
        for module in frozen_list:
            for m in module.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
        super().train(mode)
        if self.freeze_bbox or self.freeze_all:
            self._freeze_all()

    def forward(self, img_feats: Tuple[Tensor],
                txt_feats: Tensor) -> Tuple[List]:
        """Forward features from the upstream network."""
        assert len(img_feats) == self.num_levels
        txt_feats = [txt_feats for _ in range(self.num_levels)]
        mask_protos = self.proto_pred(img_feats[0])
        cls_logit, bbox_preds, bbox_dist_preds, coeff_preds = multi_apply(
            self.forward_single, img_feats, txt_feats, self.cls_preds,
            self.reg_preds, self.cls_contrasts, self.seg_preds)
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds, coeff_preds, mask_protos
        else:
            return cls_logit, bbox_preds, None, coeff_preds, mask_protos

    def forward_single(self, img_feat: Tensor, txt_feat: Tensor,
                       cls_pred: nn.ModuleList, reg_pred: nn.ModuleList,
                       cls_contrast: nn.ModuleList,
                       seg_pred: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat)
        cls_logit = cls_contrast(cls_embed, txt_feat)
        bbox_dist_preds = reg_pred(img_feat)
        coeff_pred = seg_pred(img_feat)
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds, coeff_pred
        else:
            return cls_logit, bbox_preds, None, coeff_pred


@MODELS.register_module()
class YOLOWorldSegHead(YOLOv5InsHead):
    def __init__(self,
                 head_module: ConfigType,
                 prior_generator: ConfigType = dict(
                     type='mmdet.MlvlPointGenerator',
                     offset=0.5,
                     strides=[8, 16, 32]),
                 bbox_coder: ConfigType = dict(type='DistancePointBBoxCoder'),
                 loss_cls: ConfigType = dict(type='mmdet.CrossEntropyLoss',
                                             use_sigmoid=True,
                                             reduction='none',
                                             loss_weight=0.5),
                 loss_bbox: ConfigType = dict(type='IoULoss',
                                              iou_mode='ciou',
                                              bbox_format='xyxy',
                                              reduction='sum',
                                              loss_weight=7.5,
                                              return_iou=False),
                 loss_dfl=dict(type='mmdet.DistributionFocalLoss',
                               reduction='mean',
                               loss_weight=1.5 / 4),
                 mask_overlap: bool = True,
                 loss_mask: ConfigType = dict(type='mmdet.CrossEntropyLoss',
                                              use_sigmoid=True,
                                              reduction='none'),
                 loss_mask_weight=0.05,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(head_module=head_module,
                         prior_generator=prior_generator,
                         bbox_coder=bbox_coder,
                         loss_cls=loss_cls,
                         loss_bbox=loss_bbox,
                         train_cfg=train_cfg,
                         test_cfg=test_cfg,
                         init_cfg=init_cfg)
        self.loss_dfl = MODELS.build(loss_dfl)
        self.loss_obj = None
        self.mask_overlap = mask_overlap
        self.loss_mask: nn.Module = MODELS.build(loss_mask)
        self.loss_mask_weight = loss_mask_weight

    def special_init(self):
        """Since YOLO series algorithms will inherit from YOLOv5Head, but
        different algorithms have special initialization process.

        The special_init function is designed to deal with this situation.
        """
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg.assigner)
            # Add common attributes to reduce calculation
            self.featmap_sizes_train = None
            self.num_level_priors = None
            self.flatten_priors_train = None
            self.stride_tensor = None

    """YOLO World head."""

    def loss(self, img_feats: Tuple[Tensor], txt_feats: Tensor,
             batch_data_samples: Union[list, dict]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network."""

        outs = self(img_feats, txt_feats)
        # Fast version
        loss_inputs = outs + (batch_data_samples['bboxes_labels'],
                              batch_data_samples['masks'],
                              batch_data_samples['img_metas'])
        losses = self.loss_by_feat(*loss_inputs)

        return losses

    def loss_and_predict(
        self,
        img_feats: Tuple[Tensor],
        txt_feats: Tensor,
        batch_data_samples: SampleList,
        proposal_cfg: Optional[ConfigDict] = None
    ) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        outs = self(img_feats, txt_feats)

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           cfg=proposal_cfg)
        return losses, predictions

    def forward(self, img_feats: Tuple[Tensor],
                txt_feats: Tensor) -> Tuple[List]:
        """Forward features from the upstream network."""
        return self.head_module(img_feats, txt_feats)

    def predict(self,
                img_feats: Tuple[Tensor],
                txt_feats: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self(img_feats, txt_feats)
        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           rescale=rescale)
        return predictions

    def aug_test(self,
                 aug_batch_feats,
                 aug_batch_img_metas,
                 rescale=False,
                 with_ori_nms=False,
                 **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError('aug_test is not implemented yet.')

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            coeff_preds: Sequence[Tensor],
            proto_preds: Tensor,
            batch_gt_instances: Sequence[InstanceData],
            batch_gt_masks: Sequence[Tensor],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            bbox_dist_preds (Sequence[Tensor]): Box distribution logits for
                each scale level with shape (bs, reg_max + 1, H*W, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        num_imgs = len(batch_img_metas)

        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores
        ]
        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(mlvl_priors_with_stride,
                                                  dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pred info
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        # (bs, n, 4 * reg_max)
        flatten_pred_dists = [
            bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_max * 4)
            for bbox_pred_org in bbox_dist_preds
        ]

        flatten_pred_coeffs = [
            coeff_pred.permute(0, 2, 3,
                               1).reshape(num_imgs, -1,
                                          self.head_module.mask_channels)
            for coeff_pred in coeff_preds
        ]

        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.stride_tensor[..., 0])
        flatten_pred_coeffs = torch.cat(flatten_pred_coeffs, dim=1)

        assigned_result = self.assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(), self.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']
        assigned_gt_idxs = assigned_result['assigned_gt_idxs']

        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores).sum()
        loss_cls /= assigned_scores_sum

        # rescale bbox
        assigned_bboxes /= self.stride_tensor
        flatten_pred_bboxes /= self.stride_tensor

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            # when num_pos > 0, assigned_scores_sum will >0, so the loss_bbox
            # will not report an error
            # iou loss
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(assigned_scores.sum(-1),
                                              fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox = self.loss_bbox(
                pred_bboxes_pos, assigned_bboxes_pos,
                weight=bbox_weight) / assigned_scores_sum

            # dfl loss
            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.head_module.reg_max - 1,
                eps=0.01)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = self.loss_dfl(pred_dist_pos.reshape(
                -1, self.head_module.reg_max),
                                     assigned_ltrb_pos.reshape(-1),
                                     weight=bbox_weight.expand(-1,
                                                               4).reshape(-1),
                                     avg_factor=assigned_scores_sum)

            _, c, mask_h, mask_w = proto_preds.shape
            if batch_gt_masks.shape[-2:] != (mask_h, mask_w):
                batch_gt_masks = F.interpolate(batch_gt_masks[None],
                                               (mask_h, mask_w),
                                               mode='nearest')[0]

            loss_mask = torch.zeros(1, device=loss_dfl.device)
            box_sum_flag = pad_bbox_flag.long().sum(dim=1).squeeze(1)

            batch_inds = torch.zeros(num_imgs,
                                     dtype=torch.int64,
                                     device=assigned_gt_idxs.device)[:, None]
            batch_inds[1:] = box_sum_flag.cumsum(dim=0)[:-1][..., None]
            _assigned_gt_idxs = assigned_gt_idxs + batch_inds

            for bs in range(num_imgs):
                # 8400
                bbox_match_inds = assigned_gt_idxs[bs]
                mask_match_inds = _assigned_gt_idxs[bs]

                bbox_match_inds = torch.masked_select(bbox_match_inds,
                                                      fg_mask_pre_prior[bs])
                mask_match_inds = torch.masked_select(mask_match_inds,
                                                      fg_mask_pre_prior[bs])

                # mask
                mask_dim = coeff_preds[0].shape[1]
                prior_mask_mask = fg_mask_pre_prior[bs].unsqueeze(-1).repeat(
                    [1, mask_dim])
                pred_coeffs_pos = torch.masked_select(flatten_pred_coeffs[bs],
                                                      prior_mask_mask).reshape(
                                                          [-1, mask_dim])

                match_boxes = gt_bboxes[bs][bbox_match_inds] / 4
                normed_boxes = gt_bboxes[bs][bbox_match_inds] / 640

                bbox_area = (normed_boxes[:, 2:] -
                             normed_boxes[:, :2]).prod(dim=1)
                if not mask_match_inds.any():
                    continue
                assert not self.mask_overlap
                mask_gti = batch_gt_masks[mask_match_inds]
                mask_preds = (
                    pred_coeffs_pos @ proto_preds[bs].view(c, -1)).view(
                        -1, mask_h, mask_w)
                loss_mask_full = self.loss_mask(mask_preds, mask_gti)
                _loss_mask = (self.crop_mask(loss_mask_full[None],
                                             match_boxes).mean(dim=(2, 3)) /
                              bbox_area)

                loss_mask += _loss_mask.mean()

        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0
            loss_mask = flatten_pred_coeffs.sum() * 0
        _, world_size = get_dist_info()

        return dict(loss_cls=loss_cls * num_imgs * world_size,
                    loss_bbox=loss_bbox * num_imgs * world_size,
                    loss_dfl=loss_dfl * num_imgs * world_size,
                    loss_mask=loss_mask * self.loss_mask_weight * world_size)

    def get_raw_predictions_and_protos(self, img_feats: Tuple[Tensor],
                                       txt_feats: Tensor,
                                       batch_data_samples: SampleList):
        """
        Perform forward pass, flatten predictions, and return them along with mask protos.
        Corresponds to the first part of YOLOv5InsHead.predict_by_feat.
        """
        # Get raw outputs from the head module
        # outs = (cls_scores_raw, bbox_preds_raw, None, coeff_preds_raw, mask_protos)
        # Note: cls_scores_raw are logits from the contrastive head.
        cls_scores_raw, bbox_preds_raw, _, coeff_preds_raw, mask_protos = self.head_module(img_feats, txt_feats)

        num_imgs = len(batch_data_samples)
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]

        featmap_sizes = [featmap.shape[2:] for featmap in cls_scores_raw]

        # Grid priors and strides, similar to YOLOv5InsHead.predict_by_feat
        # Use a different attribute name for predict-time priors to avoid conflict with self.featmap_sizes_train used in loss
        if not hasattr(self, '_predict_featmap_sizes') or featmap_sizes != self._predict_featmap_sizes:
            self._predict_mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores_raw[0].dtype,
                device=cls_scores_raw[0].device)
            self._predict_featmap_sizes = featmap_sizes
        
        # flatten_priors are grid cell coordinates, e.g. (0.5, 0.5), (1.5, 0.5) ... for each level, unscaled by stride.
        flatten_priors = torch.cat(self._predict_mlvl_priors) 

        # Strides for each prior
        mlvl_strides = []
        for featmap_size, stride_val_tuple in zip(featmap_sizes, self.prior_generator.strides):
            # In YOLOWorldSegHeadModule, predictions are not repeated for num_base_priors like in standard YOLOv5.
            # Each grid cell directly produces one set of predictions.
            num_base_priors = 1 
            # stride_val_tuple is (stride_w, stride_h), use stride_w assuming square strides or consistent scaling
            actual_stride_to_fill = stride_val_tuple[0] 
            mlvl_strides.append(
                flatten_priors.new_full((featmap_size.numel() * num_base_priors, ), actual_stride_to_fill)
            )
        flatten_stride = torch.cat(mlvl_strides) # Shape: (total_priors, )

        # Flatten predictions
        # cls_scores_raw are logits from contrastive head. Shape: (num_imgs, num_total_priors, num_classes)
        flatten_cls_scores_logits = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes)
            for cls_score in cls_scores_raw
        ]
        flatten_cls_scores_logits = torch.cat(flatten_cls_scores_logits, dim=1)

        # bbox_preds_raw. Shape: (num_imgs, num_total_priors, 4 if reg_max<=1 else 4*reg_max)
        # These are typically distances (ltrb) in "grid cell units" if reg_max > 0.
        raw_bbox_pred_channels = bbox_preds_raw[0].shape[1] # Should be 4 or 4*reg_max (e.g. 64 if reg_max=16)
                                                          # YOLOWorldSegHeadModule.forward_single reshapes this to 4 after DFL.
                                                          # So it's always 4 here after head_module call.
        flatten_bbox_preds_unitless = [
            # YOLOWorldSegHeadModule.forward_single ensures bbox_preds are [bs, -1 (4), h, w]
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) 
            for bbox_pred in bbox_preds_raw 
        ]
        flatten_bbox_preds_unitless = torch.cat(flatten_bbox_preds_unitless, dim=1) 
        
        # coeff_preds_raw. Shape: (num_imgs, num_total_priors, mask_channels)
        flatten_coeff_preds = [
            coeff_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.head_module.mask_channels)
            for coeff_pred in coeff_preds_raw
        ]
        flatten_coeff_preds = torch.cat(flatten_coeff_preds, dim=1)

        # Decode bboxes:
        # `flatten_priors` are grid cell center coords (e.g. (0.5,0.5), unscaled by stride).
        # `flatten_bbox_preds_unitless` are distances (ltrb) in grid cell units (unscaled by stride).
        # `self.bbox_coder.decode` (DistancePointBBoxCoder) expects points and pred_bboxes to be in the *same* units.
        # The output of decode will be in those units. We then scale by stride to get pixel coordinates.
        # xyxy_grid_units = (cx - l, cy - t, cx + r, cy + b) in grid units.
        # xyxy_pixels = xyxy_grid_units * stride.
        
        decoded_bboxes_grid_units_list = []
        for i in range(num_imgs):
            # self.bbox_coder is DistancePointBBoxCoder. decode(points(N,2), pred_bboxes(N,4)) -> xyxy boxes (N,4)
            decoded_bboxes_single_img_grid = self.bbox_coder.decode(flatten_priors, flatten_bbox_preds_unitless[i])
            decoded_bboxes_grid_units_list.append(decoded_bboxes_single_img_grid)
        
        flatten_decoded_bboxes_grid_units = torch.stack(decoded_bboxes_grid_units_list) # (num_imgs, total_priors, 4)
        
        # Scale decoded bboxes to image pixel coordinates
        # flatten_stride is (total_priors, ). Unsqueeze for broadcasting: (1, total_priors, 1)
        flatten_decoded_bboxes_pixels = flatten_decoded_bboxes_grid_units * flatten_stride.view(1, -1, 1)

        # Objectness for YOLOWorld is implicitly included in class scores from the contrastive head.
        # For compatibility with some NMS logic expecting separate objectness, pass None.
        flatten_objectness_sigmoid = None 

        return {
            "flatten_cls_scores_logits": flatten_cls_scores_logits, # Logits: (num_imgs, total_priors, num_classes)
            "flatten_decoded_bboxes_pixels": flatten_decoded_bboxes_pixels, # Decoded & scaled to image pixels: (num_imgs, total_priors, 4)
            "flatten_objectness_sigmoid": flatten_objectness_sigmoid, # None for YOLOWorld
            "flatten_coeff_preds": flatten_coeff_preds, # (num_imgs, total_priors, mask_channels)
            "mask_protos": mask_protos, # (num_imgs, mask_channels, proto_h, proto_w)
            "batch_img_metas": batch_img_metas, # List of dicts
            "num_imgs": num_imgs
            # Other items like priors, strides are not directly needed by the next NMS step if bboxes are already decoded.
        }

    def perform_nms_and_get_boxes_coeffs(
            self,
            flatten_cls_scores_logits: Tensor, # Logits from get_raw_predictions_and_protos
            flatten_decoded_bboxes_pixels: Tensor, # Already decoded to pixel scale
            flatten_objectness_sigmoid: Optional[Tensor], # Expected to be None for YOLOWorld
            flatten_coeff_preds: Tensor, # Corresponding coeffs for all priors
            batch_img_metas: List[dict],
            num_imgs: int,
            cfg: ConfigDict): # Test time config
        """
        Performs NMS on raw decoded predictions and returns results (bboxes, scores, labels, coeffs).
        Corresponds to the NMS part of YOLOv5InsHead.predict_by_feat.
        Output `InstanceData` will include `coeffs` aligned with NMSed boxes.
        """
        
        cfg_for_nms = self.test_cfg if cfg is None else cfg
        # Deepcopy to avoid in-place modification issues if test_cfg is shared
        cfg_for_nms = copy.deepcopy(cfg_for_nms) 
        
        results_list = []
        
        # Apply sigmoid to logits to get scores [0,1]
        flatten_cls_scores_sigmoid = flatten_cls_scores_logits.sigmoid() # (num_imgs, total_priors, num_classes)

        for i in range(num_imgs):
            # Per-image predictions
            current_bboxes_pixels = flatten_decoded_bboxes_pixels[i] # Shape: (total_priors, 4)
            current_scores_sigmoid = flatten_cls_scores_sigmoid[i]   # Shape: (total_priors, num_classes)
            current_coeffs = flatten_coeff_preds[i]                  # Shape: (total_priors, mask_channels)
            current_img_meta = batch_img_metas[i]

            if current_scores_sigmoid.shape[0] == 0: 
                empty_results = InstanceData(img_meta=current_img_meta)
                empty_results.bboxes = current_bboxes_pixels.new_zeros((0, 4))
                empty_results.scores = current_scores_sigmoid.new_zeros(0) 
                empty_results.labels = current_scores_sigmoid.new_zeros(0, dtype=torch.long)
                empty_results.coeffs = current_coeffs.new_zeros((0, current_coeffs.shape[-1]))
                results_list.append(empty_results)
                continue

            score_thr = cfg_for_nms.get('score_thr', 0.001) 
            nms_pre = cfg_for_nms.get('nms_pre', 100000) if cfg_for_nms else 100000
            
            multi_label_nms = cfg_for_nms.get('multi_label', False) if cfg_for_nms else False
            # cfg_for_nms.multi_label might be set by YOLOv5InsHead's init `multi_label &= self.num_classes > 1`
            # For YOLOWorld, this logic might also apply from YOLOWorldSegHead's inheritance.
            # Check if self.num_classes is 1, then multi_label_nms is effectively False.
            if self.num_classes == 1:
                multi_label_nms = False

            if not multi_label_nms:
                max_scores_per_prior, class_indices_per_prior = current_scores_sigmoid.max(1) 
                payload_for_filter = dict(labels=class_indices_per_prior, coeffs=current_coeffs, bboxes=current_bboxes_pixels)
                scores_after_filter, _, keep_idxs, results_payload_after_filter = \
                    filter_scores_and_topk(
                        max_scores_per_prior.unsqueeze(1), 
                        score_thr,
                        nms_pre,
                        results=payload_for_filter
                    )
                final_scores_before_nms = scores_after_filter.squeeze(-1) 
                final_labels_before_nms = results_payload_after_filter['labels'] 
                final_bboxes_before_nms = results_payload_after_filter['bboxes'] 
                final_coeffs_before_nms = results_payload_after_filter['coeffs'] 
            else:
                # This path for multi_label=True using filter_scores_and_topk returns flattened scores and labels.
                # The bboxes and coeffs are indexed by `keep_idxs` which are prior indices.
                # `labels` from filter_scores_and_topk will be the class index for each score.
                scores_cat, labels_cat, keep_idxs, _ = \
                    filter_scores_and_topk(
                        current_scores_sigmoid, # (total_priors, num_classes)
                        score_thr,
                        nms_pre,
                        results=None # Don't need to carry payload here, will index manually
                    )
                final_scores_before_nms = scores_cat 
                final_labels_before_nms = labels_cat 
                final_bboxes_before_nms = current_bboxes_pixels[keep_idxs] # Index original bboxes by prior indices
                final_coeffs_before_nms = current_coeffs[keep_idxs]       # Index original coeffs by prior indices


            results_for_nms = InstanceData(
                bboxes=final_bboxes_before_nms, 
                scores=final_scores_before_nms,  
                labels=final_labels_before_nms,  
                coeffs=final_coeffs_before_nms   
            )
            
            if cfg_for_nms and cfg_for_nms.get('yolox_style', False): 
                cfg_for_nms.max_per_img = len(results_for_nms)

            results_after_nms = self._bbox_post_process(
                results=results_for_nms,
                cfg=cfg_for_nms, 
                rescale=False, 
                with_nms=True, 
                img_meta=current_img_meta 
            )
            results_list.append(results_after_nms)
            
        return results_list

    def generate_masks_from_coeffs_and_boxes(
            self,
            batch_results_input: List[InstanceData],  # List of InstanceData (one per image)
                                                 # containing bboxes, scores, labels, and associated_coeffs
            mask_protos_batch: Tensor,         # mask_protos from get_raw_predictions_and_protos (num_imgs, mask_channels, proto_h, proto_w)
            batch_img_metas: List[dict],       # From get_raw_predictions_and_protos
            cfg: ConfigDict,
            rescale: bool = True):
        """
        Generates masks for given boxes and coefficients.
        Corresponds to the mask processing part of YOLOv5InsHead.predict_by_feat.
        batch_results_input will be updated with masks.
        """
        cfg = self.test_cfg if cfg is None else cfg
        # cfg = copy.deepcopy(cfg) # Not strictly necessary if only reading from cfg here

        final_results_list = []

        for i, results_per_image in enumerate(batch_results_input):
            # results_per_image contains: bboxes, scores, labels, coeffs
            # These bboxes are assumed to be in the original image space if coming from fusion,
            # or in padded/resized space if coming directly from NMS without prior rescaling.
            # The rescale flag here applies to masks and final bbox adjustments.

            img_meta = batch_img_metas[i]
            mask_proto_per_image = mask_protos_batch[i] # (mask_channels, proto_h, proto_w)
            
            if len(results_per_image.bboxes) == 0:
                # If no detections, create empty masks and append
                h_out, w_out = img_meta['ori_shape'][:2] if rescale else img_meta['img_shape'][:2]
                results_per_image.masks = results_per_image.bboxes.new_zeros(
                    (0, h_out, w_out), dtype=torch.bool)
                final_results_list.append(results_per_image)
                continue

            # Bboxes needed for process_mask should be in the coordinate space of the feature map
            # that mask_protos were generated on, or process_mask should handle various coord spaces.
            # YOLOv5InsHead.process_mask takes bboxes in input image shape (before rescale to ori_shape)
            # and mask_proto. It upsamples masks then crops.
            
            # The bboxes in results_per_image could be:
            # 1. Directly from perform_nms_and_get_boxes_coeffs: these are in input image space (potentially padded).
            # 2. From weighted_boxes_fusion: these should also be in input image space if inputs to WBF were.
            
            # `process_mask` expects bboxes relative to the shape that masks will be interpolated to.
            # In YOLOv5InsHead, this shape is `batch_input_shape`.
            current_input_shape = img_meta['batch_input_shape'] # Shape of the network input tensor (H, W)

            # Call self.process_mask (from YOLOv5InsHead)
            # process_mask(self, mask_proto, mask_coeff_pred, bboxes, shape, upsample=True)
            # - mask_proto: (mask_channels, H_proto, W_proto)
            # - mask_coeff_pred: (num_dets, mask_channels)
            # - bboxes: (num_dets, 4) in coordinates of `shape`
            # - shape: tuple (H_target, W_target) for upsampling masks
            # Returns: (1, num_dets, H_target, W_target) after sigmoid
            
            # Ensure bboxes are correctly scaled for process_mask if they are not already in 'current_input_shape' coords.
            # If bboxes are already in 'current_input_shape' (e.g. from NMS without rescale), direct use is fine.
            # If bboxes are in 'ori_shape' (e.g. after fusion and normalization), they need to be scaled back.
            # For now, assume bboxes in results_per_image are in 'current_input_shape' or compatible.
            # This might need adjustment based on where fusion happens and how coords are managed.
            
            # Let's assume `results_per_image.bboxes` are in the `current_input_shape` coordinate system.
            # (This is true if they come from `perform_nms_and_get_boxes_coeffs` without rescaling)

            masks_logits = self.process_mask(
                mask_proto=mask_proto_per_image,
                mask_coeff_pred=results_per_image.coeffs,
                bboxes=results_per_image.bboxes, # Assumed to be in current_input_shape coords
                shape=current_input_shape, # Target shape for mask upsampling
                upsample=True # Upsample to current_input_shape
            ) # Output: (1, num_dets, current_input_shape_h, current_input_shape_w)
            
            # masks_logits are sigmoided values.

            # Rescale and finalize masks (logic from YOLOv5InsHead.predict_by_feat)
            if rescale:
                ori_shape = img_meta['ori_shape']
                if 'pad_param' in img_meta and img_meta['pad_param'] is not None:
                    pad_param = img_meta['pad_param']
                    # Crop padding from masks
                    top_pad, bottom_pad, left_pad, right_pad = pad_param
                    input_h, input_w = current_input_shape
                    
                    masks_logits = masks_logits[:, :, 
                                                top_pad : input_h - bottom_pad, 
                                                left_pad : input_w - right_pad]
                    
                    # Adjust bboxes if they were in padded space and need to be in original image space for output
                    # This adjustment should ideally happen to the bboxes that are *returned* to the user.
                    # If results_per_image.bboxes were already rescaled to ori_shape by fusion, this is not needed here for bboxes.
                    # However, if bboxes are from NMS (in padded space), they need rescaling.
                    # For consistency, let's assume bboxes are always brought to ori_shape if rescale=True.
                    # The bboxes used for process_mask are different from the final output bboxes.
                    
                    # This part of bbox adjustment is tricky if fusion happens outside.
                    # Let's assume the final bboxes in results_per_image are already in ori_shape if rescale is true.
                    # Or, if they are from NMS, we adjust them here.
                    # For now, this method primarily focuses on mask generation. Bbox rescaling is complex with fusion.
                    # The plan is that demo/predict.py will handle bbox scaling for fused boxes.
                    # So, here we just focus on mask scaling.
                
                # Interpolate masks to original image shape
                # masks_logits is (1, num_dets, H_cropped, W_cropped)
                # Need (num_dets, H_cropped, W_cropped) for interpolate if batch_first=False for F.interpolate
                # Or (1, num_dets, H_cropped, W_cropped) is fine.
                # F.interpolate expects (N, C, H, W) or (C, H, W) if N=1
                # Here it's (batch_size=1, channels=num_dets, H, W)
                
                masks_final_scale = F.interpolate(
                    masks_logits, # (1, num_dets, H_in, W_in)
                    size=ori_shape[:2], # (H_out, W_out)
                    mode='bilinear',
                    align_corners=False
                ) # Output: (1, num_dets, H_ori, W_ori)
                
                # Threshold masks
                # mask_thr_binary is from test_cfg, e.g., 0.5
                mask_thr = cfg.get('mask_thr_binary', 0.5) if cfg else 0.5
                final_masks_bool = masks_final_scale.squeeze(0) > mask_thr # (num_dets, H_ori, W_ori)
            else:
                # No rescaling, just threshold masks at current_input_shape (after potential crop if pad_param was handled)
                # This path is less common if final output is expected in original image coordinates.
                # If pad_param was handled, masks_logits are already cropped.
                h_out, w_out = masks_logits.shape[-2:]
                mask_thr = cfg.get('mask_thr_binary', 0.5) if cfg else 0.5
                final_masks_bool = masks_logits.squeeze(0) > mask_thr # (num_dets, h_out, w_out)

            results_per_image.masks = final_masks_bool.bool() # Ensure boolean type
            
            # Bbox clamping to original image dimensions (if rescale was True and bboxes are now in ori_shape)
            # This should be done by the caller (demo/predict.py) after fusion and potential rescaling of bboxes.
            # If results_per_image.bboxes were rescaled along with masks:
            # if rescale:
            #     results_per_image.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            #     results_per_image.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            final_results_list.append(results_per_image)
            
        return final_results_list
