import torch
from models import networks


# We use pix2pix models to build the workflow inference model
class Pix2Pix(torch.nn.Module):
  def __init__(self, opts, input_dim, output_dim):
    super(Pix2Pix, self).__init__()
    self.isTrain = (opts.phase == 'train')
    self.gpu_ids = opts.gpu_ids
    self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

    # generator
    self.netG = networks.define_G(input_dim, output_dim, 0, 64, netG='unet_256', norm='instance', upsample='bilinear', padding_type='reflect', gpu_ids=self.gpu_ids)

    # discriminator
    if self.isTrain:
      self.netD = networks.define_D(input_dim + output_dim, 64, netD='basic_256_multi', norm='instance', num_Ds=2, gpu_ids=self.gpu_ids)

    if self.isTrain:
      # define loss functions, optimizers
      self.criterionGAN = networks.GANLoss(gan_mode=opts.gan_mode).to(self.device)
      self.criterionL1 = torch.nn.L1Loss()
      self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opts.lr, betas=(0.5, 0.999))
      self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opts.lr, betas=(0.5, 0.999))
      self.optimizers = [self.optimizer_G, self.optimizer_D]

  def set_scheduler(self, opts, ep):
    assert(self.isTrain)
    self.schedulers = [networks.get_scheduler(optimizer, opts, ep) for optimizer in self.optimizers]

  def set_input(self, imgA, imgB):
    self.real_A = imgA
    self.real_B = imgB

  def forward(self):
    self.fake_B = self.netG(self.real_A)

  def backward_D(self):
    """Calculate GAN loss for the discriminator"""
    # Fake; stop backprop to the generator by detaching fake_B
    fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
    pred_fake = self.netD(fake_AB.detach())
    loss_D_fake, _ = self.criterionGAN(pred_fake, False)
    # Real
    real_AB = torch.cat((self.real_A, self.real_B), 1)
    pred_real = self.netD(real_AB.detach())
    loss_D_real, _ = self.criterionGAN(pred_real, True)
    # combine loss and calculate gradients
    self.loss_D = (loss_D_fake + loss_D_real) * 0.5
    self.loss_D.backward()

  def backward_G(self, retain_graph=False):
    """Calculate GAN and L1 loss for the generator"""
    # First, G(A) should fake the discriminator
    fake_AB = torch.cat((self.real_A, self.fake_B), 1)
    pred_fake = self.netD(fake_AB)
    self.loss_G_GAN, _ = self.criterionGAN(pred_fake, True, loss_on_D=False)
    # Second, G(A) = B
    self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
    # combine loss and calculate gradients
    self.loss_G = self.loss_G_GAN + self.loss_G_L1*10
    self.loss_G.backward(retain_graph=retain_graph)

  def update_D(self):
    self.set_requires_grad(self.netD, True)
    self.optimizer_D.zero_grad()
    self.backward_D()
    self.optimizer_D.step()

  def zero_grad_G(self):
    self.optimizer_G.zero_grad()

  def update_G(self, retain_graph=False):
    self.set_requires_grad(self.netD, False)
    self.backward_G(retain_graph)
    self.optimizer_G.step()

  def reset_D(self):
    assert(self.isTrain == True)
    networks.init_weights(self.netD, init_type='xavier', init_gain=0.02)

  def update_lr(self):
    for sch in self.schedulers:
      sch.step()
    print('  learning rate = %.7f' % self.optimizers[0].param_groups[0]['lr'])

  def set_requires_grad(self, nets, requires_grad=False):
    if not isinstance(nets, list):
      nets = [nets]
    for net in nets:
      if net is not None:
        for param in net.parameters():
          param.requires_grad = requires_grad

  def save(self):
    state = {
        'netG': self.netG.state_dict(),
        'netD': self.netD.state_dict(),
        'optG': self.optimizer_G.state_dict(),
        'optD': self.optimizer_D.state_dict()
        }
    return state

  def load(self, ck):
    self.netG.load_state_dict(ck['netG'])

    if self.isTrain:
      self.netD.load_state_dict(ck['netD'])
      self.optimizer_G.load_state_dict(ck['optG'])
      self.optimizer_D.load_state_dict(ck['optD'])

    return
