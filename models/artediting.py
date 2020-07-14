from os.path import join
import torch
import torch.nn as nn
import torchvision
from tensorboardX import SummaryWriter
from models.pix2pix import Pix2Pix
from models.bicycleganadain import BicycleGANAdaIN as BicycleGAN

class ArtEditing(nn.Module):
  def __init__(self, opts):
    super(ArtEditing, self).__init__()

    # stage
    self.n_stage = opts.n_stage
    self.isTrain = (opts.phase == 'train')
    self.joint = not (opts.phase == 'train')
    self.train_mode = 'all' if self.isTrain else None

    # Artwork generation model
    if self.isTrain:
      self.bicycle = [BicycleGAN(opts, opts.input_dim[i], opts.input_dim[i + 1], opts.lambda_ms[i]) for i in range(self.n_stage - 1)]
    else:
      self.bicycle = [BicycleGAN(opts, opts.input_dim[i], opts.input_dim[i + 1]) for i in range(self.n_stage - 1)]

    # Workflow inference model
    self.pix2pix = [Pix2Pix(opts, opts.input_dim[i + 1], opts.input_dim[i]) for i in range(self.n_stage - 1)]

    # tensorboard
    if self.isTrain:
      self.tf_board = SummaryWriter(logdir=join(opts.output_dir, 'tfboard', opts.name))

    # cycle-consistency related
    if self.isTrain:
      self.cycle = not opts.no_cycle
      self.criterionL1 = nn.L1Loss()
    return

  def set_train_mode(self, mode):
    assert(mode in ['all', 'backward', 'forward'])
    self.train_mode = mode
    if mode == 'forward':
      [pix2pix.set_requires_grad(pix2pix.netG, False) for pix2pix in self.pix2pix]
      [pix2pix.set_requires_grad(pix2pix.netD, False) for pix2pix in self.pix2pix]
    return

  def set_scheduler(self, opts, eplast):
    if self.train_mode != 'forward':
      [pix2pix.set_scheduler(opts, eplast) for pix2pix in self.pix2pix]
    if self.train_mode != 'backward':
      [bicycle.set_scheduler(opts, eplast) for bicycle in self.bicycle]

  def set_joint(self, isJoint):
    self.joint = isJoint

  def set_input(self, imgs):
    if self.isTrain and (self.train_mode != 'backward'):
      assert(imgs[0].size(0) % 2 == 0)
    self.imgs = imgs
    return

  def test_forward_backward(self, img):
    outs = []
    with torch.no_grad():
      for i in range(self.n_stage - 2, -1, -1):
        self.pix2pix[i].set_input(img, None)
        self.pix2pix[i].forward()
        img = self.pix2pix[i].fake_B
        outs.append(img)
    outs.reverse()
    return outs

  def test_forward(self, img0, zs=None, vary_stage=-1):
    outs = []
    with torch.no_grad():
      inp = img0
      for i in range(self.n_stage - 1):
        if i == vary_stage or zs is None:
          z = self.bicycle[i].get_z_random(img0.size(0), self.bicycle[i].nz, img0.device)
        else:
          z = zs[i]
        out = self.bicycle[i].netG(inp, z)
        if i >= vary_stage:
          outs.append(out)
        inp = out
    return outs

  # workflow inference
  def forward_backward(self):
    assert(self.isTrain == True)
    # last stage
    self.pix2pix[self.n_stage - 2].set_input(self.imgs[self.n_stage - 1], self.imgs[self.n_stage - 2])
    self.pix2pix[self.n_stage - 2].forward()

    # remaining stages
    for i in range(self.n_stage - 3, -1, -1):
      if self.joint:
        self.pix2pix[i].set_input(self.pix2pix[i + 1].fake_B.detach(), self.imgs[i])
      else:
        self.pix2pix[i].set_input(self.imgs[i + 1], self.imgs[i])
      self.pix2pix[i].forward()

    # image for display
    self.img_dis_pix2pix = [self.imgs[self.n_stage - 1][:1].detach().cpu(), torch.zeros(self.imgs[self.n_stage - 1][:1].size())]
    for i in range(self.n_stage - 2, -1, -1):
      self.img_dis_pix2pix += [self.imgs[i][:1].detach().cpu(), self.pix2pix[i].fake_B[:1].detach().cpu()]
    self.img_dis_bicycle = None
    return

  def forward(self):
    assert(self.isTrain == True)
    half_size = self.imgs[0].size(0) // 2
    if self.train_mode == 'backward':
      self.forward_backward()
      return

    # workflow inference
    if self.train_mode == 'all':
      self.pix2pix[self.n_stage - 2].set_input(self.imgs[self.n_stage - 1][:half_size], self.imgs[self.n_stage - 2][:half_size])
      self.pix2pix[self.n_stage - 2].forward()
      for i in range(self.n_stage - 3, -1, -1):
        if self.joint:
          self.pix2pix[i].set_input(self.pix2pix[i + 1].fake_B.detach(), self.imgs[i][:half_size])
        else:
          self.pix2pix[i].set_input(self.imgs[i + 1][:half_size], self.imgs[i][:half_size])
        self.pix2pix[i].forward()
    elif self.joint:
      assert(self.train_mode == 'forward')
      with torch.no_grad():
        self.pix2pix[self.n_stage - 2].set_input(self.imgs[self.n_stage - 1][:half_size], self.imgs[self.n_stage - 2][:half_size])
        self.pix2pix[self.n_stage - 2].forward()
        for i in range(self.n_stage - 3, -1, -1):
          self.pix2pix[i].set_input(self.pix2pix[i + 1].fake_B.detach(), self.imgs[i][:half_size])
          self.pix2pix[i].forward()

    # artwork generation
    for i in range(self.n_stage - 1):
      if not self.joint:
        content = self.imgs[i]
        style = self.imgs[i + 1]
      elif i == 0:
        content = self.pix2pix[0].fake_B.detach()
        style = self.pix2pix[1].fake_B.detach()
      elif i == self.n_stage - 2:
        content = self.bicycle[self.n_stage - 3].fake_B_encoded.detach()
        style = self.imgs[self.n_stage - 1]
      else:
        content = self.bicycle[i - 1].fake_B_encoded.detach()
        style = self.pix2pix[i + 1].fake_B.detach()
      self.bicycle[i].set_input(content, style, self.imgs[i + 1])
      self.bicycle[i].forward()

    # cycle consistency
    if self.cycle:
      self.bicycle2pix2pix = [self.pix2pix[i].netG(self.bicycle[i].fake_B_random) for i in range(self.n_stage - 1)]
    else:
      with torch.no_grad():
        self.bicycle2pix2pix = [self.pix2pix[i].netG(self.bicycle[i].fake_B_random) for i in range(self.n_stage - 1)]

    # image for display
    self.img_dis_bicycle = []
    for i in range(self.n_stage - 2, -1, -1):
      self.img_dis_bicycle += [self.imgs[i + 1][:1].detach().cpu(), self.bicycle[i].fake_B_encoded[:1].detach().cpu(), self.bicycle[i].fake_B_random[:1].detach().cpu()]
    self.img_dis_bicycle += [self.imgs[0][:1].detach().cpu(), torch.zeros(self.imgs[0][:1].size()), torch.zeros(self.imgs[0][:1].size())]
    self.img_dis_pix2pix = [pix2pix.fake_B[:1].detach().cpu() for pix2pix in self.pix2pix] if (self.train_mode == 'all' or self.joint) else None

    return

  def update_G(self):
    # zero gradient
    [pix2pix.zero_grad_G() for pix2pix in self.pix2pix]
    [bicycle.zero_grad_GE() for bicycle in self.bicycle]

    # update artwork generation model
    if (self.train_mode != 'backward'):
      for i in range(self.n_stage - 2, -1, -1):
        '''self.bicycle[i].update_GE()
        self.bicycle[i].zero_grad_GE()
        self.bicycle[i].update_G(retain_graph=(self.joint or self.cycle))'''
        self.bicycle[i].backward_EG()
        self.bicycle[i].backward_G_alone(retain_graph=(self.joint or self.cycle))

    # update workflow inference model
    if (self.train_mode != 'forward'):
      for i in range(self.n_stage - 1):
        self.pix2pix[i].update_G(retain_graph=(self.joint or self.cycle))

    # update artwork generation model with cycle consistency
    if (self.train_mode != 'backward'):
      self.loss_B2P = [self.criterionL1(self.bicycle2pix2pix[i], self.bicycle[i].input_A_encoded.detach()) for i in range(self.n_stage - 1)]
      '''if self.cycle:
        for i in range(self.n_stage - 2, -1, -1):
          self.bicycle[i].optimizer_G.zero_grad()
          self.loss_B2P[i].backward()
          self.bicycle[i].optimizer_G.step()'''
      for i in range(self.n_stage - 2, -1, -1):
        if self.cycle:
          self.loss_B2P[i].backward()
        self.bicycle[i].optimizer_E.step()
        self.bicycle[i].optimizer_G.step()

    return

  def update_D(self):
    if (self.train_mode != 'forward'):
      [pix2pix.update_D() for pix2pix in self.pix2pix]
    if (self.train_mode != 'backward'):
      [bicycle.update_D() for bicycle in self.bicycle]
    return

  def write_display(self, total_it):
    # write losses
    if (total_it + 1) % 10 == 0:
      if (self.train_mode != 'forward'):
        [self.write_loss(total_it, self.pix2pix[i], 'workflow_inference_{}'.format(i)) for i in range(self.n_stage - 1)]
      if (self.train_mode != 'backward'):
        [self.write_loss(total_it, self.bicycle[i], 'artwork_generation_{}'.format(i)) for i in range(self.n_stage - 1)]
        [self.tf_board.add_scalar('loss_B2P_{}'.format(i), self.loss_B2P[i], total_it) for i in range(self.n_stage - 1)]

    # write images
    if (total_it + 1) % 50 == 0:
      if self.img_dis_pix2pix is not None:
        img_dis = torch.cat([torch.cat([img, img, img], dim=1) if img.size(1) == 1 else img for img in self.img_dis_pix2pix], dim=0)
        img_dis = torchvision.utils.make_grid(img_dis, nrow=max(1, img_dis.size(0)//self.n_stage)) / 2 + 0.5
        self.tf_board.add_image('Image (workflow inference)', img_dis, total_it)
      if self.img_dis_bicycle is not None:
        img_dis = torch.cat([torch.cat([img, img, img], dim=1) if img.size(1) == 1 else img for img in self.img_dis_bicycle], dim=0)
        img_dis = torchvision.utils.make_grid(img_dis, nrow=img_dis.size(0)//self.n_stage) / 2 + 0.5
        self.tf_board.add_image('Image (artwork generation)', img_dis, total_it)
    return

  def write_loss(self, total_it, model, groupname):
    members = [attr for attr in dir(model) if not callable(getattr(model, attr)) and 'loss' in attr]
    for m in members:
      self.tf_board.add_scalar(groupname + '/' + m, getattr(model, m), total_it)
    return

  def update_lr(self):
    if (self.train_mode != 'forward'):
      [pix2pix.update_lr() for pix2pix in self.pix2pix]
    if (self.train_mode != 'backward'):
      [bicycle.update_lr() for bicycle in self.bicycle]
    return

  def save(self, filename, ep, total_it):
    print('--- save the model @ ep {} ---'.format(ep))
    state = {'ep': ep, 'total_it': total_it}
    for i in range(self.n_stage - 1):
      state['pix2pix_{}'.format(i)] = self.pix2pix[i].save()
      state['bicycle_{}'.format(i)] = self.bicycle[i].save()
    torch.save(state, filename)
    return

  def load(self, filename):
    checkpoint = torch.load(filename, map_location=self.pix2pix[0].device)
    for i in range(self.n_stage - 1):
      self.pix2pix[i].load(checkpoint['pix2pix_{}'.format(i)])
      self.bicycle[i].load(checkpoint['bicycle_{}'.format(i)])
    return checkpoint['ep'], checkpoint['total_it']

  def load_inference(self, filename):
    assert(self.isTrain)
    if filename == '':
      raise Exception('No load_inference specified for loading workflow inference model!')
    checkpoint = torch.load(filename, map_location=self.pix2pix[0].device)
    for i in range(self.n_stage - 1):
      self.pix2pix[i].load(checkpoint['pix2pix_{}'.format(i)])
    return
