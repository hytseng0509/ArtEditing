import torch
import torchvision
import torch.nn as nn
from models import networks
from models.l2regularizer import L2Regularizer

class AdaINOptimize(nn.Module):
  def __init__(self, opts, model, vgg, output_dim):
    super(AdaINOptimize, self).__init__()

    # parameter
    self.refine_iter = opts.n_optimize
    self.isTrain = (opts.reg_phase == 'train')
    self.gpu_ids = opts.gpu_ids
    self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
    self.refine_lr = 0.1
    if self.isTrain:
      self.lambda_reconstruction = opts.lambda_reconstruction

    # model and vgg model (dont require gradient calculation)
    self.num_params = model.netG.module.num_adain_params
    self.learner = model
    self.learner.eval()
    self.vgg = vgg
    self.vgg.eval()
    self.set_requires_grad(self.learner, False)
    self.set_requires_grad(self.vgg, False)

    # regularizer
    self.net = L2Regularizer(self.num_params, opts.batch_size)

    # discriminator
    if self.isTrain:
      self.netD = networks.define_D(output_dim, 64, netD='basic_256_multi', norm='instance', num_Ds=2, gpu_ids=self.gpu_ids)
      ckt = torch.load(opts.load, map_location=str(self.device))
      self.cktD = ckt['bicycle_{}'.format(opts.reg_stage)]['netD']
      self.netD.load_state_dict(self.cktD)
      self.criterionGAN = networks.GANLoss(gan_mode=opts.gan_mode).to(self.device)

    # optimizer
    if self.isTrain:
      self.opt = torch.optim.Adam(self.net.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
      self.optD = torch.optim.Adam(self.netD.parameters(), lr=2e-4, betas=(0.5, 0.999))

    return

  def resetD(self):
    self.netD.load_state_dict(self.cktD)
    self.optD = torch.optim.Adam(self.netD.parameters(), lr=2e-4, betas=(0.5, 0.999))

  def L1Loss(self, inp, tgt):
    # does not average on batch dimension as we are doing batch-wise operation
    return nn.functional.l1_loss(inp, tgt, reduction='none').sum(dim=0).mean()

  def ReconstructionLoss(self, inp, tgt, perceptual_weight):
    loss_l1 = self.L1Loss(inp, tgt)

    # preceptual loss
    loss_perceptual = 0
    inp_vggf = self.vgg(inp)
    tgt_vggf = self.vgg(tgt)
    for inp_f, tgt_f in zip(inp_vggf, tgt_vggf):
      loss_perceptual += self.L1Loss(inp_f, tgt_f)

    return loss_l1 + perceptual_weight*loss_perceptual, loss_l1, loss_perceptual

  def forward(self, imgA, imgB, loader_single_iterator=None):
    # set up input/target
    self.imgA = imgA
    self.imgB = imgB
    outBs_rec = []

    # reset meta learner and discriminator
    self.net.reset()
    if self.isTrain:
      self.resetD()

    # get encoded initial parameters
    with torch.no_grad():
      z, _ = self.learner.netE(self.imgB)
      zAdaIN = self.learner.netG.module.get_adain_params(z)
      zAdaIN_delta = torch.zeros_like(zAdaIN)
    zAdaIN_delta.requires_grad=True

    # get initial reconstruction
    outB_rec = self.learner.netG(self.imgA, zAdaIN + zAdaIN_delta)
    outBs_rec.append(outB_rec)
    self.outB_rec0 = outB_rec[:1].clone()
    loss_rec, _, _ = self.ReconstructionLoss(outB_rec, self.imgB, 10)

    # start optimization loop
    self.loss_G_refine = []
    for step in range(self.refine_iter):

      # get gradient
      if self.isTrain and step % (self.refine_iter // 5) == 0:
        self.loss_G_refine.append((loss_rec.item() / self.imgA.size(0), step + 1))
      grad = torch.autograd.grad(loss_rec, zAdaIN_delta)[0]

      # get new parameters
      zAdaIN_delta = self.net(loss_rec, grad, zAdaIN.detach(), zAdaIN_delta.detach(), self.refine_lr)
      assert(zAdaIN_delta.requires_grad)
      if not zAdaIN_delta.requires_grad:
        zAdaIN_delta.requires_grad = True

      # get new reconstruction
      outB_rec = self.learner.netG(self.imgA, zAdaIN + zAdaIN_delta)
      loss_rec, _, _ = self.ReconstructionLoss(outB_rec, self.imgB, 10)

      # record output
      if (step + 1) % (self.refine_iter // 5) == 0:
        outBs_rec.append(outB_rec)

      # --- training stage: optimize regularizer using the loss after update ---
      if self.isTrain:

        # get reconstruction loss after updates
        loss_G_rec = loss_rec

        # forward on other images
        imgA_other, imgB_other = next(loader_single_iterator)
        imgA_other = imgA_other.cuda()
        imgB_other = imgB_other.cuda()
        outB_var = self.learner.netG(imgA_other, zAdaIN + zAdaIN_delta)

        # GAN loss
        self.set_requires_grad(self.netD, False)
        pred_fake = self.netD(outB_var)
        loss_G_GAN, _ = self.criterionGAN(pred_fake, True, loss_on_D=False)

        # backward G
        loss_G = loss_G_GAN + self.lambda_reconstruction*loss_G_rec
        self.opt.zero_grad()
        loss_G.backward(retain_graph=True)
        self.opt.step()

        # update_D
        self.update_D(imgB_other, outB_var.detach())

    # for display
    if self.isTrain:
      self.loss_G_rec = loss_G_rec.item() / self.imgA.size(0)
      self.loss_G_GAN = loss_G_GAN.item() / self.imgA.size(0)
      self.loss_G = loss_G.item() / self.imgA.size(0)
      self.imgA_other = imgA_other
      self.outB_rec = outB_rec[:1]
      self.outB_var = outB_var[:1]

    return zAdaIN + zAdaIN_delta, outBs_rec

  def update_D(self, real, fake):
    self.set_requires_grad(self.netD, True)

    pred_real = self.netD(real)
    pred_fake = self.netD(fake)
    loss_D_real, _ = self.criterionGAN(pred_real, True)
    loss_D_fake, _ = self.criterionGAN(pred_fake, False)

    self.optD.zero_grad()
    loss_D = loss_D_real + loss_D_fake
    loss_D.backward()
    self.optD.step()
    self.loss_D = loss_D.item()
    return

  def set_requires_grad(self, nets, requires_grad):
    if not isinstance(nets, list):
      nets = [nets]
    for net in nets:
      if net is not None:
        for param in net.parameters():
          param.requires_grad = requires_grad

  def train(self):
    self.isTrain = True

  def eval(self):
    self.isTrain = False

  def write_display(self, tf_board, total_it):
    # write loss
    tf_board.add_scalars('loss_G_inner', {'loss_rec_{}'.format(loss_G_refine[1]): loss_G_refine[0] for loss_G_refine in self.loss_G_refine}, total_it)
    tf_board.add_scalar('loss_G_GAN', self.loss_G_GAN, total_it)
    tf_board.add_scalar('loss_G_rec', self.loss_G_rec, total_it)
    tf_board.add_scalar('loss_G', self.loss_G, total_it)
    if self.loss_D is not None:
      tf_board.add_scalar('loss_D', self.loss_D, total_it)

    # write images
    img_dis = [self.imgA[:1], self.imgB[:1], self.outB_rec0[:1], self.outB_rec[:1], self.imgA_other[:1], self.outB_var[:1]]
    img_dis = torch.cat([torch.cat([img, img, img], dim=1) if img.size(1) == 1 else img for img in img_dis], dim=0)
    img_dis = torchvision.utils.make_grid(img_dis, nrow=img_dis.size(0)//3) / 2 + 0.5
    tf_board.add_image('Image', img_dis, total_it)
    return

  def save(self, filename, it):
    print('--- save the model @ it {} ---'.format(it))
    state = {
        'it': it,
        'net': self.net.state_dict(),
        'opt': self.opt.state_dict(),
        }
    torch.save(state, filename)
    return

  def load(self, filename):
    ckt = torch.load(filename)
    self.net.load_state_dict(ckt['net'])
    if self.isTrain:
      self.opt.load_state_dict(ckt['opt'])
    return ckt['it']
