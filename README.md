# Exploring Intrinsic Discrimination and Consistency for Weakly Supervised Object Localization (under review)

Official PyTorch implementation of ''Exploring Intrinsic Discrimination and Consistency for Weakly Supervised Object Localization'' (IDC). 

![Framework](imgs/framework.jpg) 

## To Do

- [x] Evaluation code and pretrained models for IDC.
- [ ] Training code, more pretrained models and a more detailed readme. (Coming soon after being accepted)

## License
Our code is released under the Creative Commons BY-NC-SA 3.0 (see [LICENSE](LICENSE) for more details), available only for non-commercial use.

Currently this repository is only used to facilitate reviewers to understand implementation details.



## Requirements  
  - python 3.6 
  - scipy 1.7.3
  - torch 1.11.0
  - torchvision 0.12.0
  - opencv-python 4.6.0.66
  - PyYAML 6.0
  - scikit-image 0.19.3
  - Pillow 9.2.0

## Get started

### Start

```bash  
git clone https://github.com/vignywang/IDC.git
cd IDC-master
```

### Download Datasets

* CUB ([http://www.vision.caltech.edu/visipedia/CUB-200-2011.html](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html))
* CUB-200-2011 dataset tree structure after preprocessing.*
```
datasets
└───CUB_200_2011
    └───train
    └───test
```
### Download Pretrained Models

|    model name    | GT-known Loc %| 
|:------------------:|:-----------------------:|
| [`vgg_CUB`]( https://drive.google.com/file/d/1rEi1GJ60PqrlY_xNQPc8jyAQ6dq02pc9/view?usp=sharing)   | 93.36                   
|[`resnet_CUB`]( https://drive.google.com/file/d/1TPcqkxY3PUkdYd1iENXe7232NxLBI189/view?usp=sharing)   | 95.59                   
|[`inception_CUB`]()| 93.83                   
|[`mobilenet_CUB`]()| 93.67
|[`vgg_ILS`]()| 72.45
|[`resnet_ILS`]()| 72.90
|[`inception_ILS`]()| 73.12
|[`mobilenet_ILS`]()| 73.76
### Training <a name="63"></a> 

```
python train.py --config ${congfigs/vgg_CUB_train.yaml} --gpu 0 
```

### Evaluation 

```
python evaluation.py --config ${congfigs/vgg_CUB.yaml} --gpu 0 --epoch 29
```
The evaluation results will be displayed as:
```bash


Cls-Loc acc 0.7236417930523903
Cls-Loc acc Top 5 0.8606499876562411
GT Loc acc 0.9335930243796378
MaxBoxAccV2 acc 0.8715766827518584
```

## Acknowledgements

Part of our evaluation and training code based on [PSOL (CVPR2020)](https://github.com/tzzcl/PSOL), [BAS (CVPR2022)](https://github.com/wpy1999/BAS) and [CREAM (CVPR2022)](https://github.com/Jazzcharles/CREAM).

Thanks for their works and sharing.
