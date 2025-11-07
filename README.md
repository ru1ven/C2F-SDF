<div align="center">

<h1>[ECCV2024] Coarse-to-Fine Implicit Representation Learning for 3D Hand-Object Reconstruction from a Single RGB-D Image</h1>



<h4 align="center">
  <a href="https://cv.nirc.top/2024/c2f-sdf" target='_blank'>[Project Page]</a> 
  <a href="https://link.springer.com/chapter/10.1007/978-3-031-72983-6_5" target='_blank'>[Paper Page]</a> 
  <a href="https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06748.pdf" target='_blank'>[PDF]</a> 

</h4>

</div>

<div>

## Setup with Conda
```bash
# create conda env
conda create -n dir python=3.9
# install torch
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
# install other requirements
git clone --recursive https://github.com/ru1ven/C2F-SDF.git
cd C2F-SDF
pip install -r ./requirements.txt
```

## Dataset

Download [Dexycb dataset](https://dex-ycb.github.io/) and [ObMan dataset](https://www.di.ens.fr/willow/research/obman/data/requestaccess.php).
Follow the [gSDF](https://github.com/zerchen/gSDF) repo to generate the original SDF files. Then use ```python preprocess/cocoify_dexycb.py``` or ```python preprocess/cocoify_obman.py``` script to process the SDF data.


## Training and Evaluation
1. [Here](https://drive.google.com/drive/folders/1smmPrF8GIWpf7kQYT_Eat8fk-FWXNpXb?usp=drive_link) is our pre-trained model on DexYCB and Obman. 
```bash
cd tools
bash dist_test.sh 1 1234 -e ../playground/experiments/DexYCB.yaml --gpu 0 --model_dir 'directory of the pre-trained model'
python eval.py -e ../outputs_test/C2F-SDF_dexycb/ -testset dexycb
```

2. Train the gSDF model:
```bash
bash dist_train.sh 1 1234 -e ../playground/experiments/DexYCB.yaml --gpu 0
```

## Citation
If you find this work useful, please consider citing:
```
@inproceedings{liu2025coarse,
  title={Coarse-to-Fine Implicit Representation Learning for 3D Hand-Object Reconstruction from a Single RGB-D Image},
  author={Liu, Xingyu and Ren, Pengfei and Wang, Jingyu and Qi, Qi and Sun, Haifeng and Zhuang, Zirui and Liao, Jianxin},
  booktitle={European Conference on Computer Vision},
  pages={74--92},
  year={2025},
  organization={Springer}
}
```

## Acknowledgement
Some of the codes are built upon [manopth](https://github.com/hassony2/manopth), [gSDF](https://github.com/zerchen/gSDF).
Thanks them for their great works!
