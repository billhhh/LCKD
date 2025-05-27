# ShaSpec - CVPR2023

The code repository of LCKD model from MICCAI 2023 [paper](https://arxiv.org/abs/2310.01035) "Learnable Cross-modal Knowledge Distillation for Multi-modal Learning with Missing Modality"

## Installation

```commandline
pip install -r requirements.txt
```

For more requirements, please refer to requirements.txt.

## Data Preparation

BraTS2018 dataset has 285 cases for training/validation (210 gliomas with high grade and 75 gliomas with low grade) and 66 cases for online evaluation, where each case (with four modalities, namely: Flair, T1, T1CE and T2) share one segmentation GT. The ground-truth of training set is publicly available, but the annotations of validation set is hidden and online evaluation is required.

The data can be requested from [here](https://www.kaggle.com/datasets/sanglequang/brats2018).

The data path can be changed in `datalist/BraTS18/`. There are 4 files in the folder: `BraTS18_train.csv` and `BraTS18_val.csv` for hyper-params tuning; `BraTS18_train_all.csv` for fixed iteration training with all data; and `BraTS18_test.csv` for online evaluation at [here](https://ipp.cbica.upenn.edu/).

## Model Training

Followed the official BraTS2018 settings, the models are trained on training data for a certain iterations and then tested on online evaluation data. Detailed hyper-parameters settings can be found in `run.sh` and in the paper. Note that we empirically found out a lower temperature of random modality dropout can help at the initial stage of the training as the model performance is not stable and gradually increase the dropout rate. Alternatively, we can perform a warmup with all modalities training as shown in the run.sh script.

For model training, the commandline is:

```commandline
bash run.sh [GPU id]
```

For instance:

```commandline
bash run.sh 0
```

## Model Evaluation

For model evaluation, the resume path of the tested model can be specified in the `eval.sh` file. This file is for output masks used in online evaluation. But if you would like to validate the model seg performance, please split a few samples in the training set as the validation set. The evaluation can be performed with:

```commandline
bash eval.sh [GPU id]
```

For example:

```commandline
bash eval.sh 0
```

Then if you want to perform postprocessing, please run:

```commandline
python postprocess.py
```

The folder paths can be modified in `postprocess.py`.

After postprocessing, [online evaluation](https://ipp.cbica.upenn.edu/) needed to be performed. Output folder containing 66 segmentations is required to be uploaded to the site for evaluation.

## Bug Fixing

In order to fit in a single 3090 Memory, the batchsize = 1 is used. So you may encounter the bug "ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 256, 1, 1, 1])".

In the bug, you should see something like: File "/home/anaconda3/envs/shaspec/lib/python3.9/site-packages/torch/nn/functional.py", line 2077, in instance_norm
    _verify_batch_size(input.size()). This is caused by the instanceNorm function check if it is batchsize = 1. So we just need to comment this line in the functional.py file into `# _verify_batch_size(input.size())`.

![ver_batch](https://github.com/billhhh/ShaSpec/assets/7709725/60ffe668-22cc-411b-9bf9-1543c7972688)

## Acknowledgement

If you got a chance to use our code, you can cite our paper with the following information:

```
@inproceedings{wang2023learnable,
  title={Learnable cross-modal knowledge distillation for multi-modal learning with missing modality},
  author={Wang, Hu and Ma, Congbo and Zhang, Jianpeng and Zhang, Yuan and Avery, Jodie and Hull, Louise and Carneiro, Gustavo},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={216--226},
  year={2023},
  organization={Springer}
}
```

Enjoy!!
