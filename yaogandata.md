- DIOR-RSVG
  - ape_coco_val_internVL_extend.jsonl
- Levir-CC-dataset_lw
  - internvl/LevirCC_fintuned_captions_internvl_CN.jsonl
- NWPU-Captions
  - NWPU-Captions_train_interVL_CN_extend_filter.jsonl
- RSVQAHR
  - RSVQAHR_train_interVL_CN_Com.jsonl 对话
  - RSVQAHR_train_interVL_CN.json 单次问答
- RSVQALR
  - RSVQALR_train_interVL_CN_Com.jsonl 对话
  - RSVQALR_train_interVL_CN.jsonl 单次问答

全是internvl所需的数据格式，需修改jsonl文件中图片的地址前缀信息。若用其他模型微调，需调整json的格式。