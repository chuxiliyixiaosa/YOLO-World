# Copyright (c) Tencent Inc. All rights reserved.
import os.path as osp
from tqdm import tqdm
import cv2
import torch
import numpy as np
import json
import os
from mmengine.config import Config
from mmengine.dataset import Compose
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
        output = model.test_step(data_batch)[0]
    pred_instances = output.pred_instances
    # score thresholding
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
    # max detections
    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]

    pred_instances = pred_instances.cpu().numpy()
    masks = pred_instances['masks']
    segs = masks2segs(masks)
    boxes = pred_instances['bboxes']
    labels = pred_instances['labels']
    scores = pred_instances['scores']
    label_texts = [texts[x][0] for x in labels]
    return segs,boxes, labels, label_texts, scores


if __name__ == "__main__":

    config_file = "configs/segmentation/yolo_world_seg_l_dual_vlpan_2e-4_80e_8gpus_allmodules_finetune_fusai.py"
    checkpoint = "weights/example.pth"

    cfg = Config.fromfile(config_file)
    cfg.work_dir = osp.join('./work_dirs')
    # init model
    cfg.load_from = checkpoint
    model = init_detector(cfg, checkpoint=checkpoint, device='cuda:0')
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
    
