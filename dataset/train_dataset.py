import os
from os.path import join
from PIL import Image
import random
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.transforms.functional as F


# get parameters for cropping images
def get_crop_params(img_size, output_size):
  w = img_size
  h = img_size
  th = output_size
  tw = output_size
  if w == tw and h == th:
    return 0, 0, h, w

  i = random.randint(0, h - th)
  j = random.randint(0, w - tw)
  return (i, j, th, tw)


class AlignedDataset(data.Dataset):

  def __init__(self, opts):
    self.dataroot = opts.dataroot
    self.train = True
    self.n_stage = opts.n_stage
    self.resize_size = opts.resize_size
    self.crop_size = opts.crop_size

    # image names (should correspond in various stages)
    names = os.listdir(join(self.dataroot, opts.phase + '0'))
    if self.train == False:
      names.sort()

    # images at all stages
    self.stage = [[join(self.dataroot, opts.phase + '{}'.format(i), name) for name in names] for i in range(self.n_stage)]

    # dataset related
    self.dataset_size = len(names)
    self.input_dim = opts.input_dim

    # transformation
    self.flip = False if not self.train else not opts.no_flip
    transforms = [ToTensor()]
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('  {} dataset size: {}, # stage: {}'.format(opts.phase, self.dataset_size, self.n_stage))
    return

  def __getitem__(self, index):
    # determine flipping
    flip = random.randint(0, 1) if self.flip else 0

    # determine croppoing
    crop = get_crop_params(self.resize_size, self.crop_size)

    # get image at all stages
    data = [self.load_img(self.stage[i][index], flip, crop, self.input_dim[i]) for i in range(self.n_stage)]

    return data

  def load_img(self, img_name, flip, crop, input_dim):
    # read image
    img = Image.open(img_name).convert('RGB')

    # flip
    if flip == 1:
      img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # resize
    img = F.resize(img, (self.resize_size, self.resize_size), Image.BICUBIC)

    # crop
    if self.train:
      img = F.crop(img, crop[0], crop[1], crop[2], crop[3])
    else:
      img = F.center_crop(img, (self.crop_size, self.crop_size))

    # transform
    img = self.transforms(img)

    # dimension stuff
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)

    return img

  def __len__(self):
    return self.dataset_size


class SingleStageDataset(data.Dataset):

  def __init__(self, opts, phase=None):
    self.dataroot = opts.dataroot
    phase = phase
    self.train = (phase == 'train')
    assert(self.train)
    self.resize_size = opts.resize_size
    self.crop_size = opts.crop_size

    # image names
    names = os.listdir(join(self.dataroot, phase + '0'))
    if self.train == False:
      names.sort()
    self.stage0 = [join(self.dataroot, phase + '{}'.format(opts.reg_stage), name) for name in names]
    self.stage1 = [join(self.dataroot, phase + '{}'.format(opts.reg_stage + 1), name) for name in names]

    # dataset related
    self.dataset_size = len(names)
    self.input_dim = opts.input_dim[opts.reg_stage]
    self.output_dim = opts.input_dim[opts.reg_stage + 1]

    # transformation
    self.flip = False if not self.train else not opts.no_flip
    transforms = [ToTensor()]
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('  {} dataset size {} stage {}'.format(phase, self.dataset_size, opts.reg_stage))
    return

  def __getitem__(self, index):
    # determine flipping
    flip = random.randint(0, 1) if self.flip else 0

    # determine croppoing
    crop = get_crop_params(self.resize_size, self.crop_size)

    # get data
    data0 = self.load_img(self.stage0[index], flip, crop, self.input_dim)
    data1 = self.load_img(self.stage1[index], flip, crop, self.output_dim)

    return data0, data1

  def load_img(self, img_name, flip, crop, input_dim):
    # read image
    img = Image.open(img_name).convert('RGB')

    # flip
    if flip == 1:
      img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # resize
    img = F.resize(img, (self.resize_size, self.resize_size), Image.BICUBIC)

    # crop
    if self.train:
      img = F.crop(img, crop[0], crop[1], crop[2], crop[3])
    else:
      img = F.center_crop(img, (self.crop_size, self.crop_size))

    # transform
    img = self.transforms(img)

    # dimension stuff
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)

    return img

  def __len__(self):
    return self.dataset_size
