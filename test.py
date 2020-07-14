import os
import numpy as np
from PIL import Image
import torch

from options import TestOptions
from dataset.test_dataset import AlignedDataset
from model.artediting import ArtEditing as Model
from model.adainoptimize import AdaINOptimize
from models.vgg import Vgg16

# tensor to PIL Image
def tensor2img(img):
  img = img[0].cpu().float().numpy()
  if img.shape[0] == 1:
    img = np.tile(img, (3, 1, 1))
  img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
  return Image.fromarray(img.astype(np.uint8))

# save a set of images
def save_imgs(imgs, names, path):
  if not os.path.exists(path):
    os.mkdir(path)
  for img, name in zip(imgs, names):
    img = tensor2img(img)
    img.save(os.path.join(path, name + '.png'))

def main():

  # load option
  parser = TestOptions()
  opts = parser.parse()
  if len(opts.gpu_ids) > 1:
    raise Exception('only one GPU for testing!')

  # create result folder
  result_dir = os.path.join(opts.output_dir, 'results', opts.name)
  if not os.path.exists(result_dir):
    os.mkdir(result_dir)

  # data loader
  print('\n--- load {} dataset ---'.format(opts.phase))
  dataset = AlignedDataset(opts)
  loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.nThreads)

  # load model
  opts.load = os.path.join(opts.output_dir, 'model', opts.load)
  print('\n--- load ArtEditing model from {} ---'.format(opts.load))
  model = Model(opts)
  model.cuda()
  _, _ = model.load(opts.load)

  # perceptual model
  vgg = Vgg16(requires_grad=False)
  vgg.cuda()
  vgg.eval()

  # load regularizer
  print('\n--- AdaIN optimizer ---')
  adain_optimize = []
  for i in range(opts.n_stage - 1):
    opts.reg_load[i] = os.path.join(opts.output_dir, 'model', opts.reg_load[i])
    print('  load {}-th stage from {}'.format(opts.reg_load[i]))
    adain_optimize.append(AdaINOptimize(opts, model.bicycle[i], vgg, opts.input_dim[i + 1]))
    adain_optimize[i].cuda()
    adain_optimize[i].load(opts.reg_load[i])

  # test
  print('\n--- Testing ---')
  for idx, (imgs, imgname) in enumerate(loader):
    outs = []
    with torch.no_grad():
      imgs = [img.cuda() for img in imgs]

      # store ground-truth img at each stage
      outs += imgs
      names = ['gt_{}'.format(i) for i in range(opts.n_stage)]

      # workflow inference
      cs = model.test_forward_backward(imgs[opts.n_stage - 1])
      outs += cs
      names += ['infer_{}'.format(i) for i in range(len(cs))]

    # artwork generation + adain optimization for each stage
    zs = []
    for i in range(opts.n_stage - 1):

      # get input and reference output image
      inp = cs[i] if i == 0 else outs[-1]
      out = imgs[i + 1] if i == opts.n_stage - 2 else cs[i + 1]

      # adain optimize
      z, imgs_rec = adain_optimize[i].forward(inp, out)
      zs.append(z)

      # store reconstructed images
      outs += imgs_rec
      names += ['rec_{}_{:04d}'.format(i + 1, step*(opts.n_refine // 5)) for step in range(len(imgs_rec))]

    # get editing results
    with torch.no_grad():

      # re-sample latent representations at each stage
      for idx_stage in range(opts.n_stage - 1):
        for idx_vary in range(opts.num):
          imgs_edit = model.test_forward(cs[0], zs, vary_stage=idx_stage)
          for i, img_edit in enumerate(imgs_edit):
            outs.append(img_edit)
            names.append('edit_{}_{}'.format(idx_stage + 1 + i, idx_stage + 1, idx_vary))

      # save
      save_imgs(outs, names, os.path.join(result_dir, imgname[0]))

    print('{}/{}'.format(idx + 1, len(loader)))

  return

if __name__ == '__main__':
  main()
