import argparse
import torch

class TrainOptions():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    # gpu related
    self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0,1,2,3  use -1 for CPU')

    # data loader related
    self.parser.add_argument('--dataroot', type=str, default='data/face', help='path of data')
    self.parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    self.parser.add_argument('--resize_size', type=int, default=286, help='resized image size for training')
    self.parser.add_argument('--crop_size', type=int, default=256, help='cropped image size for training')
    self.parser.add_argument('--input_dim', type=str, default='1,3,3', help='number of image channel for each workflow stage')
    self.parser.add_argument('--nThreads', type=int, default=4, help='# of threads for data loader')
    self.parser.add_argument('--no_flip', action='store_true', help='specified if no flipping')

    # ouptput related
    self.parser.add_argument('--name', type=str, default='trial', help='folder name to save outputs')
    self.parser.add_argument('--output_dir', type=str, default='outputs', help='path for saving display results')

    # model related
    self.parser.add_argument('--bicycleE', type=str, default='resnet_256', help='bicycleGAN encoder [resnet_256, vgg_relu41]')
    self.parser.add_argument('--nz', type=int, default=8, help='dimension of noise z')
    self.parser.add_argument('--load', type=str, default='', help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--load_inference', type=str, default='', help='specified the dir of saved models for loading the workflow inference model')
    self.parser.add_argument('--gan_mode', type=str, default='hinge', help='gan_mode')

    # training related
    self.parser.add_argument('--lambda_ms', type=str, default='0.01,0.01', help='mode seeking loss for generation models at various stages')
    self.parser.add_argument('--no_cycle', action='store_true', help='specify to NOT use the cycle loss for training artwork generation model')
    self.parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    self.parser.add_argument('--n_ep_separate', type=int, default=40, help='number of epochs to separately train each stage')
    self.parser.add_argument('--n_ep_joint', type=int, default=20, help='number of epochs to jointly train each stage')
    self.parser.add_argument('--lr_policy', type=str, default='none', help='learning rate decay policy')
    self.parser.add_argument('--n_ep_decay', type=int, default=-1, help='epoch start decay learning rate, set -1 if no decay')

  def parse(self):
    self.opt = self.parser.parse_args()
    self.opt.phase = 'train'
    args = vars(self.opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
      print('%s: %s' % (str(name), str(value)))

    # set gpu ids
    str_ids = self.opt.gpu_ids.split(',')
    self.opt.gpu_ids = []
    for str_id in str_ids:
      id = int(str_id)
      if id >= 0:
        self.opt.gpu_ids.append(id)
    if len(self.opt.gpu_ids) > 0:
      torch.cuda.set_device(self.opt.gpu_ids[0])

    # set input_dim
    input_dim = self.opt.input_dim.split(',')
    self.opt.input_dim = [int(i) for i in input_dim]
    self.opt.n_stage = len(self.opt.input_dim)

    # set lambda_ms
    lambda_ms = self.opt.lambda_ms.split(',')
    self.opt.lambda_ms = [float(l) for l in lambda_ms]

    assert(len(self.opt.lambda_ms) == self.opt.n_stage - 1)
    return self.opt


class TestOptions():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    # data loader related
    self.parser.add_argument('--dataroot', type=str, default='data/face', help='path of data')
    self.parser.add_argument('--phase', type=str, default='test', help='phase for dataloading')
    self.parser.add_argument('--resize_size', type=int, default=286, help='resized image size for training')
    self.parser.add_argument('--crop_size', type=int, default=256, help='cropped image size for training')
    self.parser.add_argument('--nThreads', type=int, default=4, help='for data loader')
    self.parser.add_argument('--input_dim', type=str, default='1,3,3', help='')

    # ouptput related
    self.parser.add_argument('--num', type=int, default=5, help='number of random editings (re-sample latent representations) per image')
    self.parser.add_argument('--name', type=str, default='trial', help='folder name to save results')
    self.parser.add_argument('--result_dir', type=str, default='results', help='path for saving results')

    # model related
    self.parser.add_argument('--output_dir', type=str, default='outputs', help='path for loading models')
    self.parser.add_argument('--bicycleE', type=str, default='resnet_256', help='bicycleGAN encoder [resnet_256, vgg_relu41]')
    self.parser.add_argument('--nz', type=int, default=8, help='dimension of noise z')
    self.parser.add_argument('--load', type=str, required=True, help='specified the dir of saved models')

    # regularizer related
    self.parser.add_argument('--reg_load', type=str, required=True, help='specify all regularizers, separate each with camma')
    self.parser.add_argument('--n_optimize', type=int, default=150, help='specified the dir of saved models for resume the training')

  def parse(self):
    self.opt = self.parser.parse_args()
    args = vars(self.opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
      print('%s: %s' % (str(name), str(value)))

    # set gpu ids
    self.opt.gpu_ids = '0'
    str_ids = self.opt.gpu_ids.split(',')
    self.opt.gpu_ids = []
    for str_id in str_ids:
      id = int(str_id)
      if id >= 0:
        self.opt.gpu_ids.append(id)
    if len(self.opt.gpu_ids) > 0:
      torch.cuda.set_device(self.opt.gpu_ids[0])

    # set input_dim
    input_dim = self.opt.input_dim.split(',')
    self.opt.input_dim = [int(i) for i in input_dim]
    self.opt.n_stage = len(self.opt.input_dim)

    # set load_meta
    meta_load = self.opt.meta_load.split(',')
    self.opt.meta_load = [ml for ml in meta_load]

    assert(len(self.opt.meta_load) == self.opt.n_stage - 1)

    # default params
    self.opt.reg_phase = 'test'
    self.opt.batch_size = 1

    return self.opt
