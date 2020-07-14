import torch
from models import networks
from models.networks_munit import Gen

class BicycleGANAdaIN(torch.nn.Module):
  def __init__(self, opts, input_dim, output_dim, lambda_ms=None):
    super(BicycleGANAdaIN, self).__init__()
    self.isTrain = (opts.phase == 'train')
    self.gpu_ids = opts.gpu_ids
    self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
    self.nz = opts.nz

    # generator
    self.netG = networks.init_net(Gen(input_dim, output_dim, style_dim=opts.nz),
                                  init_type='xavier', init_gain=0.02, gpu_ids=self.gpu_ids)

    # discriminator
    if self.isTrain:
      self.netD = networks.define_D(output_dim, 64, netD='basic_256_multi', norm='instance', num_Ds=2, gpu_ids=self.gpu_ids)
      self.netD2 = networks.define_D(output_dim, 64, netD='basic_256_multi', norm='instance', num_Ds=2, gpu_ids=self.gpu_ids)

    # encoder
    self.netE = networks.define_E(output_dim, opts.nz, 64, netE=opts.bicycleE, norm='instance', vaeLike=True, gpu_ids=self.gpu_ids)

    # loss and optimizer and scheduler
    if self.isTrain:
      self.criterionGAN = networks.GANLoss(gan_mode=opts.gan_mode).to(self.device)
      self.criterionL1 = torch.nn.L1Loss()
      self.criterionZ = torch.nn.L1Loss()
      self.lambda_ms = 0. if lambda_ms is None else lambda_ms
      self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opts.lr, betas=(0.5, 0.999))
      self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opts.lr, betas=(0.5, 0.999))
      self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opts.lr, betas=(0.5, 0.999))
      self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=opts.lr, betas=(0.5, 0.999))
      self.optimizers = [self.optimizer_G, self.optimizer_E, self.optimizer_D, self.optimizer_D2]

  def set_scheduler(self, opts, ep):
    assert(self.isTrain)
    self.schedulers = [networks.get_scheduler(optimizer, opts, ep) for optimizer in self.optimizers]

  def set_input(self, imgA, imgB, imgB_real):
    if self.isTrain:
      assert(imgB_real.size(0) % 2 == 0)
    self.input_A = imgA
    self.input_B = imgB
    self.real_B = imgB_real

  def get_z_random(self, batch_size, nz, device, random_type='gauss'):
    if random_type == 'uni':
      z = torch.rand(batch_size, nz, device=device) * 2.0 - 1.0
    elif random_type == 'gauss':
      z = torch.randn(batch_size, nz, device=device)
    return z

  def encode(self, input_image):
    mu, logvar = self.netE.forward(input_image)
    std = logvar.mul(0.5).exp_()
    eps = self.get_z_random(std.size(0), std.size(1), std.device)
    z = eps.mul(std).add_(mu)
    return z, mu, logvar

  def test(self, z0=None, encode=False):
    with torch.no_grad():
      if encode:  # use encoded z
        z0, _ = self.netE(self.input_B)
      if z0 is None:
        z0 = self.get_z_random(self.input_A.size(0), self.nz, self.input_A.device)
      self.fake_B = self.netG(self.input_A, z0)
    return self.input_A, self.fake_B, self.input_B

  def forward(self):
    # get real images
    half_size = self.real_B.size(0) // 2
    # A1, B1 for encoded; A2, B2 for random
    self.input_A_encoded = self.input_A[0:half_size]
    self.input_B_encoded = self.input_B[0:half_size]
    self.real_B_encoded = self.real_B[0:half_size]
    self.real_B_random = self.real_B[half_size:]

    # get encoded z
    self.z_encoded, self.mu, self.logvar = self.encode(self.input_B_encoded)
    # get random z
    self.z_random = self.get_z_random(self.input_A_encoded.size(0), self.nz, self.input_A_encoded.device)
    self.z_random2 = self.get_z_random(self.input_A_encoded.size(0), self.nz, self.input_A_encoded.device)

    # generate fake_B_encoded
    self.fake_B_encoded = self.netG(self.input_A_encoded, self.z_encoded)
    # generate fake_B_random
    self.fake_B_random = self.netG(self.input_A_encoded.detach(), self.z_random)
    if self.lambda_ms > 1e-5:
      self.fake_B_random2 = self.netG(self.input_A_encoded.detach(), self.z_random2)

    # for discriminator
    self.fake_data_encoded = self.fake_B_encoded
    self.fake_data_random = self.fake_B_random
    self.real_data_encoded = self.real_B_encoded
    self.real_data_random = self.real_B_random

    # compute z_predict
    self.mu2, logvar2 = self.netE(self.fake_B_random)  # mu2 is a point estimate

  def backward_D(self, netD, real, fake):
    # real
    pred_real = netD(real)
    loss_D_real, _ = self.criterionGAN(pred_real, True)

    # Fake, stop backprop to the generator by detaching fake_B
    pred_fake = netD(fake.detach())
    loss_D_fake, _ = self.criterionGAN(pred_fake, False)

    # Combined loss
    loss_D = loss_D_fake + loss_D_real
    loss_D.backward()
    return loss_D, [loss_D_fake, loss_D_real]

  def backward_G_GAN(self, fake, netD=None, ll=0.0):
    if ll > 0.0:
      pred_fake = netD(fake)
      loss_G_GAN, _ = self.criterionGAN(pred_fake, True, loss_on_D=False)
    else:
      loss_G_GAN = 0
    return loss_G_GAN * ll

  def backward_EG(self):
    self.set_requires_grad(self.netD, False)

    # 1, G(A) should fool D
    self.loss_G_GAN = self.backward_G_GAN(self.fake_data_encoded, self.netD, 1.)
    self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD2, 1.)

    # 2. KL loss
    self.loss_kl = torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp()) * (-0.5)

    # 3, reconstruction |fake_B-real_B|
    self.loss_G_L1 = self.criterionL1(self.fake_B_encoded, self.real_B_encoded)

    self.loss_G = self.loss_G_GAN + self.loss_G_GAN2 + self.loss_G_L1*10 + self.loss_kl*0.01
    self.loss_G.backward(retain_graph=True)

  def update_D(self):
    self.set_requires_grad(self.netD, True)
    # update D1
    self.optimizer_D.zero_grad()
    self.loss_D, _ = self.backward_D(self.netD, self.real_data_encoded, self.fake_data_encoded)
    self.optimizer_D.step()

    # update D
    self.optimizer_D2.zero_grad()
    self.loss_D2, _ = self.backward_D(self.netD2, self.real_data_random, self.fake_data_random)
    self.optimizer_D2.step()

  def backward_G_alone(self, retain_graph=False):
    self.set_requires_grad(self.netD, False)
    self.set_requires_grad([self.netE], False)

    # 3, reconstruction |(E(G(A, z_random)))-z_random|
    self.loss_z_L1 = torch.mean(torch.abs(self.mu2 - self.z_random))

    # 4, mode seeking loss
    if self.lambda_ms > 1e-5:
      loss_ms_denominator = torch.mean(torch.abs(self.fake_B_random2 - self.fake_B_random)) / torch.mean(torch.abs(self.z_random2 - self.z_random))
      self.loss_ms = 1 / (loss_ms_denominator + 1e-5)
    else:
      self.loss_ms = 0

    # backward
    loss_G_alone = 0.5*self.loss_z_L1 + self.lambda_ms*self.loss_ms
    loss_G_alone.backward(retain_graph=retain_graph)
    self.set_requires_grad([self.netE], True)

  def zero_grad_GE(self):
    self.optimizer_E.zero_grad()
    self.optimizer_G.zero_grad()

  def update_GE(self):
    self.backward_EG()
    self.optimizer_G.step()
    self.optimizer_E.step()
    return

  def update_G(self, retain_graph=False):
    self.backward_G_alone(retain_graph)
    self.optimizer_G.step()
    return

  '''def optimize_parameters(self):
    self.forward()

    # update G and E
    #self.update_G_and_E()
    self.zero_grad_GE()
    self.update_GE()
    self.zero_grad_GE()
    self.update_G()

    # update D
    self.update_D()'''

  def reset_D(self):
    assert(self.isTrain == True)
    networks.init_weights(self.netD, init_type='xavier', init_gain=0.02)
    networks.init_weights(self.netD2, init_type='xavier', init_gain=0.02)

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
        'netE': self.netE.state_dict(),
        'netG': self.netG.state_dict(),
        'netD': self.netD.state_dict(),
        'netD2': self.netD2.state_dict(),
        'optE': self.optimizer_E.state_dict(),
        'optG': self.optimizer_G.state_dict(),
        'optD': self.optimizer_D.state_dict(),
        'optD2': self.optimizer_D2.state_dict()}
    return state

  def load(self, ck):
    self.netE.load_state_dict(ck['netE'])
    self.netG.load_state_dict(ck['netG'])
    if self.isTrain:
      self.netD.load_state_dict(ck['netD'])
      self.netD2.load_state_dict(ck['netD2'])
      self.optimizer_E.load_state_dict(ck['optE'])
      self.optimizer_G.load_state_dict(ck['optG'])
      self.optimizer_D.load_state_dict(ck['optD'])
      self.optimizer_D2.load_state_dict(ck['optD2'])
    return

