# Hierarchical-Vertebral-Landmark-Localization
The implementation of RouterNet, proposed in the paper "RouterNet: Hierarchical Point Routing Network for Robust Vertebral Landmark Localization on AP X-ray Images". 

The motivation of this work is to tackle the challenging task of vertebral landmark localization in AP X-ray images using a hierarchical and robust approach. Specifically, we propose RouterNet, a novel CNN that employs a divide-and-conquer strategy by hierarchically routing from the spinal center to each vertebral corner, enabling precise and reliable landmark localization. 

For more details, please refer to our paper.

<img src="./fig/fig.png" height='350px'>

RouterNet originally initialize the root point automatically. Besides, to make the results more controllable, we develop a doctor-friendly demo which permits a manual initialization.
<table><tr style="border:none" align="center">
<td style="border:none"><img src='./fig/box2landmarks.gif' height="435px"> <p align="center">Draw box for initialization</p></td>
<td style="border:none"><img src='./fig/point2cobb.gif' height="435px"><p align="center" >Click for initialization</p></td>
</tr></table>


## Dependencies
The packages we used in this repository are listed in below.

- Python==3.7
- opencv-python==4.8.1.78
- numpy==1.21.5
- torch==1.10.1+cu111
- tensorboardX==2.2
- albumentations==1.3.1
- einops==0.6.1

## Quickly start
To quickly start the code, an example is provided for training and validation. 
You can simply run the `train.py` for training and `val.py` for validation.


## Dataset
The available public datasets could be downloaded here:
- Anterior-posterior view X-ray datasets [AASCE dataset](https://aasce19.github.io/#challenge-dataset).
- Lateral view X-ray datasets [NHANES II dataset](https://wwwn.cdc.gov/nchs/nhanes/nhanes2/default.aspx).

We invited two local experts to manually label the preprocessed 98 test images of AASCE dataset:
- The annotations are released in folder `./dataset/MICCAI_98_Annotations/`.

## Usage
All default parameter settings we used are written in `./utils.py` files.

After configuring the environment, please use this command to train the model.
```sh
python train.py -b_size 40 -aug_num 80
```

Use this command to obtain the validation results.
```sh
python val.py 
```

Use this command to obtain the testing results.
```sh
python test.py 
```

## Pre-trained model
The pre-trained model `Jul25-1445-root_8` is provided in `./model` folder. For better performance in specific task, you can use `best_corner` in vertebral landmark localization and `best_smape` in cobb_angle calculation.

## Updates compared to paper
The implementation in this code repository has the following updates compared to the original paper description:
- We utilize the mean shape of points to achieve better point initialization at different stages of the hierarchical routing process, which is detailed in `./decoder.py`.

Please note that these updates may affect the results, so be sure to consider these updates when using or evaluating the codes.



