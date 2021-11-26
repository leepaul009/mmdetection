
## 1. Prepare Datasets
annotations should be the format of COCO.
Note: Images in 'images/test' are validation set(sorry for bad naming).
```
./data/ped
├── annotations
│   └── dhd_traffic_train.json
│   └── dhd_traffic_val.json
├── images
│   └── train
│   └── test
```

## 2. Installation
### 2.1 setup environment
step env and install torch
```
# create and enter mmdetection env:
conda create -n mmdetection python=3.8
conda activate mmdetection

# install necessary packages:
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
pip install pycocotools
pip install  tqdm scipy pandas
pip install opencv-python
```
install the mmdetection
```
# Install mmcv-full:
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html

# install MMDetection:
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```


## 3 Train and inference
### 3.1 Download pre-trained model:
Download following pretrained model from https://github.com/open-mmlab/mmdetection:
```
# for hrnet cascade(box):
  cascade_rcnn_hrnetv2p_w32_20e_coco_20200208-928455a4.pth
# for gfl:
  gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco_20200630_102002-134b07df.pth
# for vfnet:
  vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-b5f6da5e.pth
```
And put the models in root folder of repository, and later config file will read those model.

### 3.2 Train:
Different models are implemented within config files.
#### 3.2.1 Train hrnet cascade(box) (best result at epoch 10):
```
python -m torch.distributed.launch --nproc_per_node=4 tools/train.py configs/_ped/cascade_hrnet_box.py --launcher pytorch --work-dir ./work_dirs/ped/hrnet_cascade_box
```
Evaluation result will be automatically writen into folder 'eval'(create by trainning code).
Extract evaluation result into a csv format output file. 
This output file could be used in multi-model-fusion step.
```
python submit.py --input work_dirs/ped/hrnet_cascade_box/eval/?.pth --output output.csv
```
Note: evaluation data loader make sure all the images in the validation set are input to model, otherwise, the result is wrong.

#### 3.2.2 Train gfl(resnet 101, best result at final epoch):
```
python -m torch.distributed.launch --nproc_per_node=4 tools/train.py configs/_ped/gfl.py --launcher pytorch --work-dir work_dirs/ped/gfl
```
Evaluation result will be automatically writen into folder 'eval'(create by trainning code).
Extract evaluation result into a csv format output file. 
This output file could be used in multi-model-fusion step.
```
python submit.py --input work_dirs/ped/gfl/eval/?.pth --output output.csv
```

#### 3.2.3 Train vfnet(resnet 101, best result at final epoch):
```
python -m torch.distributed.launch --nproc_per_node=4 tools/train.py configs/_ped/vfnet_r101.py --launcher pytorch --work-dir work_dirs/ped/vfnet_r101
```
Extract evaluation result into a csv format output file. 
```
python submit.py --input work_dirs/ped/vfnet_r101/eval/?.pth --output output.csv
```

## Acknowledgement
* [detectron2](https://github.com/facebookresearch/detectron2)

