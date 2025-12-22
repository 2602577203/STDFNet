# STDFNet
We propose a Spatio-temporal Template based Discriminative Fusion Network (STDFNet) that dynamically suppresses background noise and enhances target features by leveraging the correlations between dynamic target/background features and search region features via designed modules (DMFM, TGFM, CAFM, MSIP) for robust RGBT tracking. 
![image](images\pipeline1.png)

### Models and Results
- You can download the model from [here](https://pan.baidu.com/s/1zg_FlYCv34o2bdzCw6TsXw?pwd=gyha).
- You can download the result from [here](https://pan.baidu.com/s/1XmeyH_U8IiUapVviF6BZSg?pwd=ff9r).

### Path Setting
Run the following command to set paths:
```
cd <PATH>
python create_default_local_file.py --workspace_dir . --data_dir <PATH_of_Datasets> --save_dir ./output
```
You can also modify paths by these two files:
```
./lib/train/admin/local.py  # paths for training
./lib/test/evaluation/local.py  # paths for testing
```

### Training
Dowmload the pretrained [foundation model](https://pan.baidu.com/s/12U1cm5xIX4mInd_7VRxY2g?pwd=bzik).
```
CUDA_VISIBLE_DEVICES=0,1  NCCL_P2P_LEVEL=NVL nohup  python tracking/train.py --script drgbt --config DRGBT603 --save_dir ./output --mode multiple --nproc_per_node 1 >  train_track.log &
```

### Testing
```
python eval_lasher.py
```

# Demonstration
As shown in the figures, our method can still ensure accurate tracking when confronted with complex background interference.
<!-- 第一行：3张图横向排列 -->
<p align="center">
  <img src="https://github.com/2602577203/STDFNet/blob/master/images/1r.gif" alt="1r" width="270" hspace="15">
  <img src="https://github.com/2602577203/STDFNet/blob/master/images/2r.gif" alt="2r" width="270" hspace="15">
  <img src="https://github.com/2602577203/STDFNet/blob/master/images/3r.gif" alt="3r" width="270" hspace="15">
</p>

<!-- 第二行：3张图横向排列 -->
<p align="center">
  <img src="https://github.com/2602577203/STDFNet/blob/master/images/1t.gif" alt="1t" width="270" hspace="15">
  <img src="https://github.com/2602577203/STDFNet/blob/master/images/2t.gif" alt="2t" width="270" hspace="15">
  <img src="https://github.com/2602577203/STDFNet/blob/master/images/3t.gif" alt="3t" width="270" hspace="15">
</p>

## Acknowledgment
This repo is based on [BAT](https://github.com/SparkTempest/BAT).