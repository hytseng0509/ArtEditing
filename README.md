<img src='imgs/teaser.png' width="800px">

# Modeling Artistic Workflows for Image Generation and Editing
[[Project Page]]()[[Paper]](https://arxiv.org/pdf/2007.07238.pdf)[[Video]](https://youtu.be/7wrImV0jidc)

Pytorch implementation for our artwork generation and editing method. The proposed design can 1) model creation workflows for different types of artwork and 2) enable both the image generation and editing at different workflow stages.

## Paper
Please cite our paper if you find the code or dataset useful for your research.

Modeling Artistic Workflows for Image Generation and Editing<br>
[Hung-Yu Tseng](https://sites.google.com/site/hytseng0509/), [Matt Fisher](https://techmatt.github.io/), [Jingwan (Cynthia) Lu](https://research.adobe.com/person/jingwan-lu/), [Yijun Li](https://yijunmaverick.github.io/), [Vladimir (Vova) Kim](http://www.vovakim.com/), [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/)<br>
European Conference on Computer Vision (ECCV), 2020<br>
```
@inproceedings{tseng2020artediting,
  author = {Tseng, Hung-Yu and Fisher, Matthew and Lu, Jingwan and Li, Yijun and Kim, Vladimir and Yang, Ming-Hsuan},
  booktitle = {European Conference on Computer Vision},
  title = {Modeling Artistic Workflows for Image Generation and Editing},
  year = {2020}
}
```

## Usage

### Installation
Clone this repo:
```
git clone https://github.com/hytseng0509/ArtEditing
cd ArtEditing
```
Install packages:
```
conda create --name artediting python=3.6
conda activate artediting
conda install pytorch torchvision -c pytorch
pip install -r requirements.txt
```

### Datasets
Download the face or anime datasets using the following script:
- Set `DATASET` to `face` or `anime`
```
cd data
python download_dataset.sh DATASET
cd ..
```

### Workflow Inference Training
We first train the workflow inference model.
```
python train_inference.py  --name face_inference --n_ep_separate 15 --n_ep_joint 15
```

### Artwork Generation Training
Then we load the trained inference model and train the artwork generation model. We need 4 GPUs for batch size of 8.
```
python train_generation.py --gpu_ids 0,1,2,3 --name face --n_ep_separate 40 --n_ep_joint 15 --load_inference face_inference/30.pth
```

### Learning-Based Regularization Training
Finally, for each workflow stage, we train the regularization for the input image reconstruction.
```
python train_regularization.py --load face/55.pth --name face_reg0 --reg_stage 0
python train_regularization.py --load face/55.pth --name face_reg1 --reg_stage 1
```

### Testing
Generate reconsturction and random editing results:
```
python test.py --name face_results --load face/55.pth --reg_load face_reg0/500.pth,face_reg1/500.pth
```
The results can be found at `results/face_results`.

## Notes
- Part of this implementation is modified from [BicycleGAN](https://github.com/junyanz/BicycleGAN/) and [MUNIT](https://github.com/NVlabs/MUNIT).
- The dataset, model, and code are for non-commercial research purposes only.
