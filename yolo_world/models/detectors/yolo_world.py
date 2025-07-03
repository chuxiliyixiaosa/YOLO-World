# Copyright (c) Tencent Inc. All rights reserved.
from typing import List, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS
from mmengine.structures import InstanceData

@MODELS.register_module()
class YOLOWorldDetector(YOLODetector):
    """Implementation of YOLOW Series"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes
        super().__init__(*args, **kwargs)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_train_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        losses = self.bbox_head.loss(img_feats, txt_feats, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        # self.bbox_head.num_classes = self.num_test_classes
        self.bbox_head.num_classes = txt_feats[0].shape[0]
        
        # Call the new head methods
        raw_preds_and_protos = self.bbox_head.get_raw_predictions_and_protos(
            img_feats, txt_feats, batch_data_samples)

        # Pass necessary items from raw_preds_and_protos to NMS method
        # cfg needs to be passed to NMS method, it's typically self.test_cfg for the head
        # but predict() takes a rescale arg, not the full test_cfg.
        # The head's test_cfg is usually already configured.
        # Using self.bbox_head.test_cfg if available, or None if predict is called without specific cfg.
        # However, YOLOWorldSegHead.perform_nms_and_get_boxes_coeffs expects a cfg.
        # The original bbox_head.predict took 'cfg' which was self.test_cfg.
        # Let's assume self.bbox_head has access to its test_cfg.
        
        pred_instances_list = self.bbox_head.perform_nms_and_get_boxes_coeffs(
            flatten_cls_scores_logits=raw_preds_and_protos["flatten_cls_scores_logits"],
            flatten_decoded_bboxes_pixels=raw_preds_and_protos["flatten_decoded_bboxes_pixels"],
            flatten_objectness_sigmoid=raw_preds_and_protos["flatten_objectness_sigmoid"],
            flatten_coeff_preds=raw_preds_and_protos["flatten_coeff_preds"],
            batch_img_metas=raw_preds_and_protos["batch_img_metas"],
            num_imgs=raw_preds_and_protos["num_imgs"],
            cfg=self.bbox_head.test_cfg # Pass the head's test_cfg for NMS parameters
        )

        # The original predict method updated batch_data_samples.
        # Here, we need to return a structure that demo/predict.py can use.
        # This structure should contain pred_instances_list and mask_protos.
        # batch_data_samples[i].pred_instances = pred_instances_list[i] (after converting InstanceData to desired format if needed)
        # For now, let's prepare a list of dicts, one per image, or a single dict if batch size is 1.
        # The output of model.test_step(data_batch)[0] in demo/predict.py expects a SampleList like structure.
        # So, we should update batch_data_samples.
        
        for i, data_sample in enumerate(batch_data_samples):
            # pred_instances_list[i] is an InstanceData object
            # It contains 'bboxes', 'scores', 'labels', 'coeffs'
            data_sample.pred_instances = pred_instances_list[i] 
            # We also need to carry mask_protos. Store it per data_sample or return separately.
            # Storing on data_sample is convenient if it's used per image.
            # mask_protos is (num_imgs, mask_channels, proto_h, proto_w)
            data_sample.mask_protos = raw_preds_and_protos["mask_protos"][i]
            # Store other raw components if needed by generate_masks_for_fused_results
            data_sample.img_metas = raw_preds_and_protos["batch_img_metas"][i] # Already on data_sample.metainfo

        # The original YOLODetector.predict returns batch_data_samples
        return batch_data_samples

    def generate_masks_for_fused_results(self,
                                         batch_fused_instances: List[InstanceData],
                                         batch_mask_protos: Tensor,
                                         batch_img_metas: List[dict], # Extracted from original batch_data_samples
                                         # batch_data_samples: SampleList, # Original data samples
                                         rescale: bool = True) -> SampleList:
        """
        Generates masks for fused bounding boxes using stored mask protos and coefficients.
        Updates batch_data_samples with the final predictions including masks.
        """
        # We need to call bbox_head.generate_masks_from_coeffs_and_boxes
        # This method expects:
        # - batch_results_input: List[InstanceData] (fused bboxes, scores, labels, coeffs)
        # - mask_protos_batch: Tensor (num_imgs, mask_channels, proto_h, proto_w)
        # - batch_img_metas: List[dict]
        # - cfg: ConfigDict (test_cfg from head)
        # - rescale: bool

        # The batch_fused_instances already contains bboxes, scores, labels, and associated coeffs.
        # batch_mask_protos is passed directly.
        # batch_img_metas is passed directly.
        
        final_pred_instances_with_masks = self.bbox_head.generate_masks_from_coeffs_and_boxes(
            batch_results_input=batch_fused_instances,
            mask_protos_batch=batch_mask_protos,
            batch_img_metas=batch_img_metas,
            cfg=self.bbox_head.test_cfg, # Use head's test_cfg
            rescale=rescale
        )
        
        # Need to construct a SampleList or update the original batch_data_samples
        # For simplicity, let's create a new list of SampleData/InstanceData to return,
        # as modifying the original batch_data_samples might be complex if it's not passed in.
        # However, the predict method returns batch_data_samples.
        # Let's assume we need to return a SampleList like structure.
        # We can create new data_samples and populate them.

        # Create a new SampleList to return
        # This part depends on how `demo/predict.py` expects the output.
        # If it expects a modified version of the original `batch_data_samples`,
        # then this method should take `batch_data_samples` as input and modify `data_sample.pred_instances`.

        # For now, let's assume this method is called with all necessary components
        # and returns a list of InstanceData which will then be put into data_samples by the caller.
        # The `inference` function in demo/predict.py currently does:
        # `output = model.test_step(data_batch)[0]` -> this is a `DetDataSample`
        # `pred_instances = output.pred_instances`
        # So, this method should ideally return a list of DetDataSample, or allow the caller to construct it.

        # Let's make this function return the list of InstanceData with masks.
        # The caller (`demo/predict.py`) will then update its `DetDataSample`.
        return final_pred_instances_with_masks


    def reparameterize(self, texts: List[List[str]]) -> None:
        # encode text embeddings into the detector
        self.texts = texts
        self.text_feats = self.backbone.forward_text(texts)

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        txt_feats = None
        if batch_data_samples is None:
            texts = self.texts
            txt_feats = self.text_feats
        elif isinstance(batch_data_samples,
                        dict) and 'texts' in batch_data_samples:
            texts = batch_data_samples['texts']
        elif isinstance(batch_data_samples, list) and hasattr(
                batch_data_samples[0], 'texts'):
            texts = [data_sample.texts for data_sample in batch_data_samples]
        elif hasattr(self, 'text_feats'):
            texts = self.texts
            txt_feats = self.text_feats
        else:
            raise TypeError('batch_data_samples should be dict or list.')
        if txt_feats is not None:
            # forward image only
            img_feats = self.backbone.forward_image(batch_inputs)
        else:
            img_feats, txt_feats = self.backbone(batch_inputs, texts)
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats


@MODELS.register_module()
class SimpleYOLOWorldDetector(YOLODetector):
    """Implementation of YOLO World Series"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 prompt_dim=512,
                 num_prompts=80,
                 embedding_path='',
                 reparameterized=False,
                 freeze_prompt=False,
                 use_mlp_adapter=False,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_training_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.prompt_dim = prompt_dim
        self.num_prompts = num_prompts
        self.reparameterized = reparameterized
        self.freeze_prompt = freeze_prompt
        self.use_mlp_adapter = use_mlp_adapter
        super().__init__(*args, **kwargs)

        if not self.reparameterized:
            if len(embedding_path) > 0:
                import numpy as np
                self.embeddings = torch.nn.Parameter(
                    torch.from_numpy(np.load(embedding_path)).float())
            else:
                # random init
                embeddings = nn.functional.normalize(torch.randn(
                    (num_prompts, prompt_dim)),
                                                     dim=-1)
                self.embeddings = nn.Parameter(embeddings)

            if self.freeze_prompt:
                self.embeddings.requires_grad = False
            else:
                self.embeddings.requires_grad = True

            if use_mlp_adapter:
                self.adapter = nn.Sequential(
                    nn.Linear(prompt_dim, prompt_dim * 2), nn.ReLU(True),
                    nn.Linear(prompt_dim * 2, prompt_dim))
            else:
                self.adapter = None

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_training_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            losses = self.bbox_head.loss(img_feats, batch_data_samples)
        else:
            losses = self.bbox_head.loss(img_feats, txt_feats,
                                         batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        self.bbox_head.num_classes = self.num_test_classes
        if self.reparameterized:
            results_list = self.bbox_head.predict(img_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)
        else:
            results_list = self.bbox_head.predict(img_feats,
                                                  txt_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            results = self.bbox_head.forward(img_feats)
        else:
            results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        # only image features
        img_feats, _ = self.backbone(batch_inputs, None)

        if not self.reparameterized:
            # use embeddings
            txt_feats = self.embeddings[None]
            if self.adapter is not None:
                txt_feats = self.adapter(txt_feats) + txt_feats
                txt_feats = nn.functional.normalize(txt_feats, dim=-1, p=2)
            txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)
        else:
            txt_feats = None
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats
