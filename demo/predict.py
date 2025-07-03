# -*- coding: utf-8 -*-
import sys
sys.path.append('/yolo/third_party/mmdetection-3.0.0')
# Copyright (c) Tencent Inc. All rights reserved.
import os.path as osp
from tqdm import tqdm
import cv2
import torch
import numpy as np
import json
import os
from ensemble_boxes import weighted_boxes_fusion
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.structures import InstanceData
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg

def masks2segs(masks,method='largest'):
    segs = []
    try:
        for mask in masks:
            binary_img = np.where(mask, 255, 0).astype(np.uint8)
            
            # 找到轮廓
            contours, _ = cv2.findContours(
                binary_img,
                mode=cv2.RETR_EXTERNAL,     # 仅检测最外层轮廓
                method=cv2.CHAIN_APPROX_SIMPLE  # 压缩水平、垂直和对角线段
            )

            if method == 'largest':
                # 按面积从大到小排序
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                single_contour = contours[0]
            else:
                raise ValueError(f"不支持的筛选方法: {method}")
            if len(single_contour) >=6:
                segs.append(single_contour.reshape(-1).tolist())
    except:
        pass
    
    return segs

def inference(model, image, texts, test_pipeline, score_thr=0.3, max_dets=100):
    image = cv2.imread(image)
    image = image[:, :, [2, 1, 0]]
    data_info = dict(img=image, img_id=0, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])
    with torch.no_grad():
        # output is a DetDataSample (or similar structure from mmdet)
        # It now contains .pred_instances (with bboxes, scores, labels, coeffs from NMS)
        # and .mask_protos
        output_data_sample = model.test_step(data_batch)[0] 
    
    # pred_instances_before_fusion is an InstanceData object
    pred_instances_before_fusion = output_data_sample.pred_instances 
    mask_protos_per_image = output_data_sample.mask_protos # (mask_channels, H_proto, W_proto) for this single image
    img_metas_per_image = output_data_sample.metainfo # Contains ori_shape, batch_input_shape etc.

    # Extract data for fusion
    # Boxes are expected by WBF in normalized format [x_min, y_min, x_max, y_max] (0-1 range)
    # Scores are a list of floats. Labels are a list of ints.
    # The bboxes from pred_instances_before_fusion are in pixel coordinates of the input image size.
    
    initial_boxes_pixels = pred_instances_before_fusion.bboxes.cpu().numpy()
    initial_scores = pred_instances_before_fusion.scores.cpu().numpy()
    initial_labels = pred_instances_before_fusion.labels.cpu().numpy()
    initial_coeffs = pred_instances_before_fusion.coeffs.cpu().numpy() # (num_boxes, num_coeffs)

    # Normalize boxes for WBF
    # img_shape is the shape of the model input (after transforms like letterboxing)
    # ori_shape is the original image shape.
    # WBF typically works better with coordinates relative to the *original* image dimensions if aspect ratios change significantly.
    # However, the model's predictions (bboxes) are in the space of `img_meta['batch_input_shape']`.
    # For consistency, let's normalize based on `batch_input_shape`.
    # The fusion happens in this normalized space, then results are mapped back if needed.
    
    h_input, w_input = img_metas_per_image['batch_input_shape']
    
    boxes_for_wbf = []
    if initial_boxes_pixels.shape[0] > 0:
        boxes_for_wbf = [
            [
                box[0] / w_input, box[1] / h_input,
                box[2] / w_input, box[3] / h_input
            ] for box in initial_boxes_pixels
        ]

    # WBF expects lists of lists for boxes, scores, labels for each model.
    # Here we have one "model" (the output before fusion).
    wbf_boxes_list = [boxes_for_wbf]
    wbf_scores_list = [initial_scores.tolist()]
    wbf_labels_list = [initial_labels.tolist()]

    # WBF parameters (can be tuned)
    iou_thr_wbf = 0.55 # Standard NMS threshold used in many detection models
    skip_box_thr_wbf = score_thr # Skip boxes with score lower than this; use the input score_thr
    weights_wbf = [1] # Weight for our single set of predictions

    if initial_boxes_pixels.shape[0] > 0 :
        fused_boxes_norm, fused_scores, fused_labels = weighted_boxes_fusion(
            wbf_boxes_list,
            wbf_scores_list,
            wbf_labels_list,
            weights=weights_wbf,
            iou_thr=iou_thr_wbf,
            skip_box_thr=skip_box_thr_wbf
        )
        
        # Denormalize fused boxes back to pixel coordinates of batch_input_shape
        fused_boxes_pixels = np.array([
            [
                box[0] * w_input, box[1] * h_input,
                box[2] * w_input, box[3] * h_input
            ] for box in fused_boxes_norm
        ])
    else:
        fused_boxes_pixels = np.empty((0, 4))
        fused_scores = np.empty((0,))
        fused_labels = np.empty((0,), dtype=int)

    # Associate coefficients with fused boxes. This is a crucial and potentially tricky step.
    # Simplest approach: For each fused box, find the original box with highest IoU and use its coeffs.
    fused_coeffs = []
    if fused_boxes_pixels.shape[0] > 0 and initial_boxes_pixels.shape[0] > 0:
        from mmdet.structures.bbox import bbox_overlaps
        iou_matrix = bbox_overlaps(torch.from_numpy(fused_boxes_pixels), torch.from_numpy(initial_boxes_pixels)).numpy()
        # iou_matrix shape: (num_fused_boxes, num_initial_boxes)
        
        for i in range(fused_boxes_pixels.shape[0]):
            if iou_matrix.shape[1] > 0:
                best_original_idx = np.argmax(iou_matrix[i])
                fused_coeffs.append(initial_coeffs[best_original_idx])
            else: # Should not happen if initial_boxes_pixels was not empty
                fused_coeffs.append(np.zeros_like(initial_coeffs[0])) # Fallback, should be handled better
        fused_coeffs = np.array(fused_coeffs) if fused_coeffs else np.empty((0, initial_coeffs.shape[1]))
    elif initial_boxes_pixels.shape[0] == 0 and fused_boxes_pixels.shape[0] > 0 : # Fused boxes from nowhere? Should not happen.
        fused_coeffs = np.empty((fused_boxes_pixels.shape[0], initial_coeffs.shape[1] if initial_coeffs.shape[0]>0 else 32 )) # Default 32 if initial_coeffs unknown
    else: # No fused boxes or no initial boxes to get coeffs from
        fused_coeffs = np.empty((0, initial_coeffs.shape[1] if initial_coeffs.shape[0] > 0 else 32))


    # Prepare InstanceData for generating masks with fused results
    # The model.generate_masks_for_fused_results expects a list of InstanceData (one per image in batch)
    # Here, inference is per image, so we create one InstanceData.
    fused_instance_data = InstanceData(
        bboxes=torch.from_numpy(fused_boxes_pixels).float().to(model.device),
        scores=torch.from_numpy(fused_scores).float().to(model.device),
        labels=torch.from_numpy(fused_labels).long().to(model.device),
        coeffs=torch.from_numpy(fused_coeffs).float().to(model.device) if fused_coeffs.shape[0] > 0 else torch.empty(0, fused_coeffs.shape[1] if fused_coeffs.ndim > 1 else 32).float().to(model.device)
    )
    
    # Call the new model method to generate masks
    # This method needs:
    # - batch_fused_instances: List[InstanceData] -> [fused_instance_data]
    # - batch_mask_protos: Tensor (num_imgs, C, H, W) -> mask_protos_per_image.unsqueeze(0)
    # - batch_img_metas: List[dict] -> [img_metas_per_image]
    # - rescale: bool
    with torch.no_grad():
        # The detector's generate_masks_for_fused_results returns a list of InstanceData.
        # Since we process one image at a time in this inference function, we take the first element.
        final_pred_instances_list = model.generate_masks_for_fused_results(
            batch_fused_instances=[fused_instance_data],
            batch_mask_protos=mask_protos_per_image.unsqueeze(0).to(model.device), # Add batch dim
            batch_img_metas=[img_metas_per_image],
            rescale=True # Rescale masks to original image shape
        )
    final_pred_instances = final_pred_instances_list[0] # Get the InstanceData for the single image

    # Apply score thresholding and max detections to the *final* fused & masked predictions
    final_pred_instances = final_pred_instances[final_pred_instances.scores.float() > score_thr]
    if len(final_pred_instances.scores) > max_dets:
        indices = final_pred_instances.scores.float().topk(max_dets)[1]
        final_pred_instances = final_pred_instances[indices]

    # Extract final results
    final_pred_instances_np = final_pred_instances.cpu().numpy() # Convert all fields at once if possible
                                                                # Or access fields and then .cpu().numpy()
    
    # Check if 'masks' field exists and is not empty
    if 'masks' in final_pred_instances and final_pred_instances.masks.shape[0] > 0:
        masks = final_pred_instances.masks.cpu().numpy()
        segs = masks2segs(masks)
    else:
        segs = [[] for _ in range(len(final_pred_instances))] # Empty segs if no masks

    boxes = final_pred_instances.bboxes.cpu().numpy()
    labels = final_pred_instances.labels.cpu().numpy()
    scores = final_pred_instances.scores.cpu().numpy()
    
    label_texts = [texts[x][0] for x in labels] # texts is the global list of class names
    return segs, boxes, labels, label_texts, scores


if __name__ == "__main__":

    config_file = "configs/segmentation/yolo_world_seg_l_dual_vlpan_2e-4_80e_8gpus_allmodules_finetune_fusai.py"
    checkpoint = "weights/example.pth"

    cfg = Config.fromfile(config_file)
    cfg.work_dir = osp.join('./work_dirs')
    # init model
    cfg.load_from = checkpoint
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = init_detector(cfg, checkpoint=checkpoint, device=device)
    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(test_pipeline_cfg)
    # Define the texts for detection
    texts = [["storage_tank"],["vehicle"],["ship"],["aircraft"],["bridge"],["railway_station"],["submarine"],["airport"],["tank"],["train"]]

    preds = []

    test_images_info = json.load(open("/input_path/test/test_images_info.json", "r"))["images"]
    test_images_root = "/input_path/test/images"
    ann_id = 0
    for content in tqdm(test_images_info):
        image = os.path.join(test_images_root, content['file_name'])
        print(f"starting to detect: {image}")
        results = inference(model, image, texts, test_pipeline,score_thr=0.1)
        for idx, (seg,box, lbl, lbl_text, score) in enumerate(zip(*results)):
            if len(seg)==0:
                continue
            preds.append({
                "image_id": content['id'],
                "category_id": int(lbl)+1,
                "bbox": box.tolist(),
                "segmentation": [seg],
                "score": score.item(),
                "id":ann_id
            })
            ann_id += 1
        # break
    json.dump(preds,open("/output_path/preds_fusai_test.json","w"),indent=4)
    
