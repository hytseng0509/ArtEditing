import argparse
import torch

class TrainOptions():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    # data loader related
    self.parser.add_argument('--dataroot', type=str, default='data/face', help='path of data')
    self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    self.parser.add_argument('--resize_size', type=int, default=286, help='resized image size for training')
    self.parser.add_argument('--crop_size', type=int, default=256, help='cropped image size for training')
    self.parser.add_argument('--input_dim', type=str, default='1,3,3', help='# of input channels for domain 0')
    self.parser.add_argument('--nThreads', type=int, default=1, help='# of threads for data loader')
    self.parser.add_argument('--no_flip', action='store_true', help='specified if no flipping')

    # ouptput related
    self.parser.add_argument('--name', type=str, default='trial_reg0', help='folder name to save outputs')
    self.parser.add_argument('--output_dir', type=str, default='outputs', help='path for saving display results')

    # model related
    self.parser.add_argument('--phase', type=str, default='test', help='phase for dataloading')
    self.parser.add_argument('--bicycleE', type=str, default='resnet_256', help='bicycleGAN encoder [resnet_256, vgg_relu41]')
    self.parser.add_argument('--nz', type=int, default=8, help='dimension of noise z')
    self.parser.add_argument('--load', type=str, required=True, help='specified the dir of saved models for the learner')

    # regularizer related
    self.parser.add_argument('--reg_phase', type=str, default='train', help='')
    self.parser.add_argument('--reg_stage', type=int, default=0, help='')
    self.parser.add_argument('--n_optimize', type=int, default=80, help='')
    self.parser.add_argument('--reg_load', type=str, default='', help='')
    self.parser.add_argument('--gan_mode', type=str, default='hinge', help='gan_mode')

    # training related
    self.parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    self.parser.add_argument('--weight_decay', type=float, default=1e-8, help='weight_deacy')
    self.parser.add_argument('--lambda_reconstruction', type=float, default=1, help='weight for the reconstruction loss')
    self.parser.add_argument('--n_ep', type=int, default=500, help='number of epochs')
    self.parser.add_argument('--lr_policy', type=str, default='none', help='learning rate decay policy')
    self.parser.add_argument('--n_ep_decay', type=int, default=-1, help='epoch start decay learning rate, set -1 if no decay')

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
      assert(len(self.opt.gpu_ids) == 1)
      torch.cuda.set_device(self.opt.gpu_ids[0])

    # set input_dim
    input_dim = self.opt.input_dim.split(',')
    self.opt.input_dim = [int(i) for i in input_dim]
    self.opt.n_stage = len(self.opt.input_dim)

    return self.opt

