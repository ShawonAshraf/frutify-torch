# frutify-torch

[![DOI](https://img.shields.io/badge/DOI-10.4018%2FIJSI.2019100103-green?style=flat-square)](https://www.igi-global.com/gateway/article/236206)

<img alt="Python" src="https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white"/> <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" /> <img alt="NumPy" src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" /> 


A `pytorch` rewrite of the [Fruit Image Classification](https://github.com/ShawonAshraf/Fruit-Image-Classification) project.

## A little note
Much of the original dataset was lost due to hardware failure and has made it difficult to reproduce the results in our
paper on the project. The models trained here are trained with recovered data and results may vary. So, in that sense, this 
repo is not a reproduction of the original work.

## Pre trained models in use
Model source : [Torchvision](https://pytorch.org/vision/stable/models.html)

- `inception-v3`
- `resnet101`

All the layers of the models were frozen and the results from the final layer were passed through a linear layer and 
tuned.


## Dataset Information
- 2233 images
- 8 classes / labels
- `fresh_apple`, `rotten_apple`, `fresh_orange`, `rotten_orange`, `fresh_banana`, `rotten_banana`, `fresh_mango`, `rotten_mango`
- Link to dataset: [Dataset](https://1drv.ms/u/s!AvaDN9CoqMWhidF_QP41wcJMlQWkSA?e=NKU9J1)

## Model Information
<table>
    <tr>
        <th>Pretrained model name</th>
        <th>Epochs</th>
        <th>Batch Size</th>
        <th>Split Ratio (Train:Valid)</th>
        <th>Learning Rate</th>
        <th>Optimizer</th>
    </tr>

<tr>
    <td>Inception-V3</td>
    <td>5</td>
    <td>16</td>
    <td>0.8</td>
    <td>0.001</td>
    <td>Adam</td>
</tr>

<tr>
    <td>Resnet101</td>
    <td>5</td>
    <td>16</td>
    <td>0.8</td>
    <td>0.001</td>
    <td>Adam</td>
</tr>

    
</table>


## Evaluation
<table>

<tr>
    <th>Pretrained model name</th>
    <th>Accuracy</th>
    <th>F1(macro)</th>
    <th>F1(weighted)</th>
    <th>Precision(macro)</th>
    <th>Precision(weighted)</th>
    <th>Recall(macro)</th>
    <th>Recall(weighted)</th>
</tr>

<tr>
    <td>Inception-V3</td>
    <td>0.80</td>
    <td>0.78</td>
    <td>0.80</td>
    <td>0.79</td>
    <td>0.80</td>
    <td>0.78</td>
    <td>0.80</td>
</tr>

<tr>
    <td>Resnet101</td>
    <td>0.95</td>
    <td>0.94</td>
    <td>0.95</td>
    <td>0.94</td>
    <td>0.95</td>
    <td>0.94</td>
    <td>0.95</td>
</tr>

</table>

### Detailed

```text
Inception-V3

              precision    recall  f1-score   support

  fresh_apple       0.82      0.86      0.84        49
 rotten_apple       0.69      0.55      0.61        20
 fresh_orange       0.85      0.92      0.88        24
rotten_orange       0.79      0.81      0.80        27
 fresh_banana       0.92      0.92      0.92        36
rotten_banana       0.71      0.75      0.73        20
  fresh_mango       0.72      0.85      0.78        27
 rotten_mango       0.86      0.57      0.69        21

     accuracy                           0.80       224
    macro avg       0.79      0.78      0.78       224
 weighted avg       0.80      0.80      0.80       224

```

```text
Resnet101

               precision    recall  f1-score   support

  fresh_apple       0.98      0.98      0.98        46
 rotten_apple       0.91      0.87      0.89        23
 fresh_orange       0.90      1.00      0.95        27
rotten_orange       1.00      0.84      0.91        25
 fresh_banana       1.00      0.97      0.98        31
rotten_banana       0.92      1.00      0.96        23
  fresh_mango       0.92      0.92      0.92        25
 rotten_mango       0.92      0.96      0.94        24

     accuracy                           0.95       224
    macro avg       0.94      0.94      0.94       224
 weighted avg       0.95      0.95      0.95       224

```

## ENV setup

```bash
# Using Anaconda / Miniconda
conda env create -f fruit.yml

conda activate frutify

# Clone the repository, afterwards
cd frutify-torch
# create a directory for saving models if you're training
mkdir saved_models/ 
```

## Saved models
Already trained models can be found here which you can 
use to run inference : 

<table>
<tr>
    <th>Model</th>
    <th>Link</th>
</tr>


<tr>
    <td>InceptionV3</td>
    <td>
        <a href="https://github.com/ShawonAshraf/frutify-torch/releases/download/pre1/inception-v3_5_16_0.001_1626740279.94928.ckpt">
            Link
        </a>
    </td>
</tr>

<tr>
    <td>Resnet101</td>
    <td>
        <a href="https://github.com/ShawonAshraf/frutify-torch/releases/download/pre1/resnet101_5_16_0.001_1626743414.734695.ckpt">
            Link
        </a>
    </td>
</tr>
</table>

## Inference
Download and store the saved model in `saved_models` directory. Or you can save elsewhere and pass the path to the script.
```bash
python test.py --model --split 0.8 --batch_size --saved_path
# example
python test.py --model resnet101 --split 0.8 --batch_size 16 --saved_path "saved_models/resnet101_5_16_0.001_1626743414.734695.ckpt"

# note : if you have a powerful multicore CPU, you may want to use the --num_workers option to speed up
# data loading, pass the number of cores you want to use.
python test.py --model resnet101 --num_workers 12 --split 0.8 --batch_size 16 --saved_path "saved_models/resnet101_5_16_0.001_1626743414.734695.ckpt"
```

For more command line options check `test.py`

## Training

Download the dataset from the provided link and copy the `dataset` directory from the zip archive to the project directory.

```bash
python trainer.py --model  --device  --split  --batch_size  --epochs  --lr 
# example
python trainer.py --model resnet101 --device gpu --split 0.8 --batch_size 16 --epochs 5 --lr 1e-3

# note : if you have a powerful multicore ( > 4) CPU, you may want to use the --num_workers option to speed up
# data loading, pass the number of cores you want to use.
python trainer.py --model resnet101 --device gpu --num_workers 12 --split 0.8 --batch_size 16 --epochs 5 --lr 1e-3
```
For more command line options check `trainer.py`
