import torch
from torchvision import models

class Vgg16(torch.nn.Module):
  def __init__(self, requires_grad=False):
    super(Vgg16, self).__init__()
    vgg_pretrained_features = models.vgg16(pretrained=True).features
    self.slice1 = torch.nn.Sequential()
    self.slice2 = torch.nn.Sequential()
    self.slice3 = torch.nn.Sequential()
    self.slice4 = torch.nn.Sequential()
    for x in range(2):
      self.slice1.add_module(str(x), vgg_pretrained_features[x])
    for x in range(2, 4):
      self.slice2.add_module(str(x), vgg_pretrained_features[x])
    for x in range(4, 14):
      self.slice3.add_module(str(x), vgg_pretrained_features[x])
    for x in range(14, 21):
      self.slice4.add_module(str(x), vgg_pretrained_features[x])
    if not requires_grad:
      for param in self.parameters():
        param.requires_grad = False

    self.normalize_mean = [0.485, 0.456, 0.406]
    self.normalize_std = [0.229, 0.224, 0.225]

  def preprocess(self, img):
    # resize
    if img.size(1) == 1:
      img = torch.cat([img, img, img], dim=1)

    # [-1, 1] -> [0, 1]
    img = (img + 1) / 2.

    # normalize
    img = img.clone()
    mean = torch.as_tensor(self.normalize_mean, dtype=img.dtype, device=img.device)
    std = torch.as_tensor(self.normalize_std, dtype=img.dtype, device=img.device)
    img.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return img

  def forward(self, X):
    X = self.preprocess(X)
    h = self.slice1(X)
    h_relu1_1 = h
    h = self.slice2(h)
    h_relu1_2 = h
    h = self.slice3(h)
    h_relu3_2 = h
    h = self.slice4(h)
    h_relu4_2 = h
    out = (h_relu1_1, h_relu1_2, h_relu3_2, h_relu4_2)
    return out
