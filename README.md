# A Semantic-based Medical Image Fusion Approach
## Dependencies
Contents of requirements.txt:

```
pytorch==1.0.1
torchvision==0.2.2
scikit-image==0.14.1
```

Install dependencies by the following command(recommended to use conda):

```
conda install --yes --file requirements.txt
```

## Prerequisite
Our data was downloaded from the Harvard Medical School [website](http://www.med.harvard.edu/AANLIB/). The script for downloading data is as follows:

```
python3 download_data.py
```

Use the following code to generate a dataset partition used by pytorch.

```
python3 generate_path.py
```

All images are registered. Image examples:

![avatar](./img/ct1_015.tif)
![avatar](./img/mr2_015.tif)

## Fusion task
We saved the parameters of the FW-Net model used in the paper, and you can directly predict the input image using the following code:

```
python3 predict.py ct_img_path mr_img_path
```

If you want to train your own model based on your dataset, the training code is visualized in **train.ipynb**, or you can execute ```python3 train.py``` directly.

The comparative methods used in this article can be found in the following links:
[GF](http://xudongkang.weebly.com/index.html), [NSCT-PCDC](https://sites.google.com/site/goravdma), [NSCT-RPCNN](https://sites.google.com/site/wodrsdas/), [LP-CNN](http://www.escience.cn/people/liuyu1/Codes.html) and [NSST-PAPCNN](http://www.escience.cn/people/liuyu1/Codes.html)


## Semantic loss evaluation task
The code of semantic loss network is in **semantic\_loss\_metric.ipynb**.


## Reference
*Our code is based on the U-Net implementation [here](https://github.com/milesial/Pytorch-UNet).