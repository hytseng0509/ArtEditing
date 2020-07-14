import os
from os.path import join
import torch
from options.options import TrainOptions
from dataset.train_dataset import AlignedDataset
from models.artediting import ArtEditing as Model

def main():

  # load options
  parser = TrainOptions()
  opts = parser.parse()

  # create output folder
  if not os.path.exists(join(opts.output_dir, 'model', opts.name)):
    os.mkdir(join(opts.output_dir, 'model', opts.name))

  # data loader
  print('\n--- load {} dataset ---'.format(opts.phase))
  dataset = AlignedDataset(opts)
  loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads, drop_last=True)

  # model
  print('\n--- create  model ---')
  model = Model(opts)
  model.set_train_mode('backward')
  model.cuda()
  if opts.load != '':
    opts.load = join(opts.output_dir, 'model', opts.load)
    print('  load model file from {}'.format(opts.load))
    ep0, total_it = model.load(opts.load)
  else:
    ep0 = -1
    total_it = 0
  model.set_scheduler(opts, ep0)
  ep0 += 1

  # train
  print('\n--- train workflow inference model ---')
  print('  start the training at epoch {}'.format(ep0))
  if (ep0 + 1) < opts.n_ep_separate:
    model.set_joint(False)
    print('  separate training')
  else:
    model.set_joint(True)
    print('  joint training')
  for ep in range(ep0, opts.n_ep_separate + opts.n_ep_joint):
    # second stage?
    if (ep + 1) == opts.n_ep_separate:
      print('  start joint training')
      model.set_joint(True)

    for it, imgs in enumerate(loader):
      # on gpu
      imgs = [img.cuda() for img in imgs]

      # set input
      model.set_input(imgs)

      # forward
      model.forward()

      # update
      model.update_D()
      model.update_G()

      # display
      model.write_display(total_it)

      # for next iteration
      if (it + 1) % (len(loader) // 10) == 0:
        print('Iteration {}, EP[{}/{}]'.format(total_it + 1, ep + 1, opts.n_ep_separate + opts.n_ep_joint))
      total_it += 1

    # update learning rate
    model.update_lr()

    # write model file
    if (ep + 1) % 5 == 0:
      model.save(join(opts.output_dir, 'model', opts.name, '{}.pth'.format(ep + 1)), ep, total_it)
  return

if __name__ == '__main__':
  main()
