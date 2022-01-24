# frutify-torch

[![DOI](https://img.shields.io/badge/DOI-10.4018%2FIJSI.2019100103-green?style=flat-square)](https://www.igi-global.com/gateway/article/236206)

<img alt="Python" src="https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white"/> <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" /> <img alt="NumPy" src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" /> 


A `pytorch` rewrite of the [Fruit Image Classification](https://github.com/ShawonAshraf/Fruit-Image-Classification) project.

## A little note
Much of the original dataset was lost due to hardware failure and has made it difficult to reproduce the results in our
paper on the project. The models trained here are trained with recovered data and results may vary. So, in that sense, this 
repo is not an actual reproduction of the original work.

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

## Training Setup
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
    <td>0.83</td>
    <td>0.83</td>
    <td>0.83</td>
    <td>0.83</td>
    <td>0.84</td>
    <td>0.83</td>
    <td>0.83</td>
</tr>

<tr>
    <td>Resnet101</td>
    <td>0.93</td>
    <td>0.93</td>
    <td>0.93</td>
    <td>0.93</td>
    <td>0.93</td>
    <td>0.93</td>
    <td>0.93</td>
</tr>

</table>

### Detailed

#### Graphs (from comet.ml)
![training](./loss,validation_loss%20VS%20step.svg)

The interactive version of this graph can be found [here](https://www.comet.ml/embedded-panel/?chartId=DDaVP3-loss&projectId=f1963281d29547269678eca5f228dd0c&viewId=TOC3133xtC83u0nddC557DSLi).
#### Metrics

```text
*************************
inception-v3


               precision    recall  f1-score   support

  fresh_apple       0.80      0.86      0.83        43
 rotten_apple       0.72      0.68      0.70        19
 fresh_orange       0.76      0.81      0.79        27
rotten_orange       0.82      0.77      0.79        30
 fresh_banana       0.94      0.91      0.92        33
rotten_banana       0.85      1.00      0.92        17
  fresh_mango       0.91      0.88      0.90        34
 rotten_mango       0.83      0.71      0.77        21

     accuracy                           0.83       224
    macro avg       0.83      0.83      0.83       224
 weighted avg       0.84      0.83      0.83       224

*************************

*************************
resnet101

               precision    recall  f1-score   support

  fresh_apple       0.95      0.98      0.97        43
 rotten_apple       0.89      0.84      0.86        19
 fresh_orange       0.87      0.96      0.91        27
rotten_orange       0.93      0.83      0.88        30
 fresh_banana       1.00      1.00      1.00        33
rotten_banana       1.00      1.00      1.00        17
  fresh_mango       0.94      0.91      0.93        34
 rotten_mango       0.86      0.90      0.88        21

     accuracy                           0.93       224
    macro avg       0.93      0.93      0.93       224
 weighted avg       0.93      0.93      0.93       224

*************************

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

## Testing
Download and store the saved model in `saved_models` directory. Or you can save elsewhere and pass the path to the script.
```bash
python test.py --inception_path --resnet_path --split 0.8 --batch_size --saved_path
# example
python test.py --inception_path "./saved_models/inception-v3_5_16_0.001_1626740279.94928.ckpt" --resnet_path "./saved_models/resnet101_5_16_0.001_1626743414.734695.ckpt" --batch_size 16 --split 0.8  

# note : if you have a powerful multicore CPU, you may want to use the --num_workers option to speed up
# data loading, pass the number of cores you want to use.
python test.py --inception_path "./saved_models/inception-v3_5_16_0.001_1626740279.94928.ckpt" --resnet_path "./saved_models/resnet101_5_16_0.001_1626743414.734695.ckpt" --num_workers 2 --batch_size 16 --split 0.8  
```

For more command line options check `test.py`

## Training

Download the dataset from the provided link and copy the `dataset` directory from the zip archive to the project directory.
 _Note: Use a GPU(>= 8GB VRAM), unless you want to wait for 20 minutes++ for each epoch to finish._

```bash
python trainer.py --model  --device  --split  --batch_size  --epochs  --lr 
# example
python trainer.py --model resnet101 --device gpu --split 0.8 --batch_size 16 --epochs 5 --lr 1e-3

# note : if you have a powerful multicore ( > 4) CPU, you may want to use the --num_workers option to speed up
# data loading, pass the number of cores you want to use.
python trainer.py --model resnet101 --device gpu --num_workers 12 --split 0.8 --batch_size 16 --epochs 5 --lr 1e-3
```
For more command line options check `trainer.py`

### Running with comet.ml logger
[comet.ml](https://www.comet.ml) logger doesn't work with multiple workers, which is a known issue. So if you want to use comet.ml for model training 
visualization, don't use the `num_workers` option. (It'll be slower but this is the only way, sadly!).

```bash
# example for using comet ml logger
python trainer.py --model inception-v3 --device gpu --split 0.8 --batch_size 16 --epochs 5 --lr 1e-3 --comet True
```

Check their [website](https://www.comet.ml) on how to get an API key and get started with pytorch and pytorch-lightning.

## Other implementations
One of the co-authors, Md Abdul Ahad Chowdhury implemented a C# based rewrite of the project, which you can find 
[here](https://github.com/maacpiash/Connery).

## Citing the original paper

``` 
@article{ashraf2019fruit,
  title={Fruit Image Classification Using Convolutional Neural Networks},
  author={Ashraf, Shawon and Kadery, Ivan and Chowdhury, Md Abdul Ahad and Mahbub, Tahsin Zahin and Rahman, Rashedur M},
  journal={International Journal of Software Innovation (IJSI)},
  volume={7},
  number={4},
  pages={51--70},
  year={2019},
  publisher={IGI Global}
}
```
