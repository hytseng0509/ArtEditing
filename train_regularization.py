import torch
import os
from tensorboardX import SummaryWriter

from options.options_reg import TrainOptions
from dataset.train_dataset import SingleStageDataset
from models.artediting import ArtEditing as Model
from models.adainoptimize import AdaINOptimize
from models.vgg import Vgg16

def cycle(iterable):
  while True:
    for x in iterable:
      yield x

def main():

  # load option
  parser = TrainOptions()
  opts = parser.parse()

  # create output folder
  if not os.path.exists(os.path.join(opts.output_dir, 'model', opts.name)):
    os.mkdir(os.path.join(opts.output_dir, 'model', opts.name))

  # data loader
  print('\n--- load {} dataset from {} ---'.format(opts.reg_phase, opts.dataroot))
  dataset = SingleStageDataset(opts, opts.reg_phase)
  loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True,
                                       num_workers=opts.nThreads, drop_last=True)
  loader_iterator = iter(cycle(loader))
  dataset_other = SingleStageDataset(opts, opts.reg_phase)
  loader_other = torch.utils.data.DataLoader(dataset_other, batch_size=opts.batch_size, shuffle=True,
                                             num_workers=opts.nThreads, drop_last=True)
  loader_other_iterator = iter(cycle(loader_other))

  # model to be regularized
  opts.load = os.path.join(opts.output_dir, 'model', opts.load)
  print('\n--- load {}-th stage ArtEditing model from {} ---'.format(opts.reg_stage, opts.load))
  model = Model(opts)
  _, _ = model.load(opts.load)
  model = model.bicycle[opts.reg_stage]
  output_dim = opts.input_dim[opts.reg_stage + 1]
  model.cuda()
  model.eval()

  # perceptual model
  print('\n--- create perceptual model')
  vgg = Vgg16(requires_grad=False)
  vgg.cuda()
  vgg.eval()

  # regularizer
  print('\n--- create the regularizer')
  adain_optimize = AdaINOptimize(opts, model, vgg, output_dim)
  if opts.reg_load != '':
    opts.reg_load = os.path.join(opts.output_dir, 'model', opts.reg_load)
    print('  load the regularizer from {}'.format(opts.reg_load))
    ep0 = adain_optimize.load(opts.reg_load)
    ep0 += 1
  else:
    ep0 = 0
  adain_optimize.cuda()
  print('start the training at epoch {}'.format(ep0 + 1))

  # tensorboard
  tf_board = SummaryWriter(logdir=os.path.join(opts.output_dir, 'tfboard', opts.name))

  # start the training
  for it in range(ep0, opts.n_ep):
    inp, out = next(loader_iterator)

    # determine input output
    inp = inp.cuda()
    out = out.cuda()

    # refine loop for this inp/out pair
    _ = adain_optimize(inp, out, loader_other_iterator)

    # display
    adain_optimize.write_display(tf_board, it)

    if (it + 1) % (opts.n_ep // 100) == 0:
      print('Iteration {}/{}'.format(it + 1, opts.n_ep))

    # write model file
    if (it + 1) % (opts.n_ep // 10) == 0:
      adain_optimize.save(os.path.join(opts.output_dir, 'model', opts.name, '{}.pth'.format(it + 1)), it)
  tf_board.close()
  return

if __name__ == '__main__':
  main()
