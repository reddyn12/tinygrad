from tinygrad import Tensor
from tinygrad.nn import Conv2d, BatchNorm2d
from tinygrad.nn.state import torch_load, load_state_dict
from tinygrad.helpers import fetch
import torch
torch.nn.AdaptiveAvgPool2d()
torch.nn.Conv2d()
Conv2d()
# NVIDA config: tfinception

# Add padding to Tensor.avg_pool2d - check Tensor.conv2d logic pad2d._pool
# Add count_include_pad=False to Tensor.avg_pool2d - may need a custom func with .where.sum
# Add tuple padding to Conv2d - Already implemented, but types not updated
class FID_Inception_V3:
  def __init__(self):
    self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
    self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
    self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
    self.maxpool1 = lambda x: Tensor.max_pool2d(x, kernel_size=(3,3), stride=2)
    
    self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
    self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
    self.maxpool2 = lambda x: Tensor.max_pool2d(x, kernel_size=(3,3), stride=2)
    
    self.Mixed_5b = FID_Inception_A(192, 32)
    self.Mixed_5c = FID_Inception_A(256, 64)
    self.Mixed_5d = FID_Inception_A(288, 64)
    
    self.Mixed_6a = FID_Inception_B(288)
    
    self.Mixed_6b = FID_Inception_C(768, 128)
    self.Mixed_6c = FID_Inception_C(768, 160)
    self.Mixed_6d = FID_Inception_C(768, 160)
    self.Mixed_6e = FID_Inception_C(768, 192)
    
    self.Mixed_7a = FID_Inception_D(768)
    
    self.Mixed_7b = FID_Inception_E(1280)
    self.Mixed_7c = FID_Inception_E(2048)
    
    # AdaptiveAvgPool2d
    self.avgpool = lambda x: Tensor.avg_pool2d(x, kernel_size=(8,8), stride=8)
    
    self.fc = None
  def __call__(self, x:Tensor):
    x = self.Conv2d_1a_3x3(x)
    x = self.Conv2d_2a_3x3(x)
    x = self.Conv2d_2b_3x3(x)
    x = self.maxpool1(x)
    x = self.Conv2d_3b_1x1(x)
    x = self.Conv2d_4a_3x3(x)
    x = self.maxpool2(x)
    x = self.Mixed_5b(x)
    x = self.Mixed_5c(x)
    x = self.Mixed_5d(x)
    x = self.Mixed_6a(x)
    x = self.Mixed_6b(x)
    x = self.Mixed_6c(x)
    x = self.Mixed_6d(x)
    x = self.Mixed_6e(x)
    x = self.Mixed_7a(x)
    x = self.Mixed_7b.forward_1(x)
    x = self.Mixed_7c.forward_2(x)
    
    x = self.avgpool(x)
    x = x.flatten(1)
      
    return x
  
  def load_pretrained(self):
    w = torch_load(fetch('https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth', 'fid.pth'))
    load_state_dict(self, w, strict=True)
      
class BasicConv2d:
  def __init__(self, in_channels:int, out_channels:int,  **kwargs):
    self.conv = Conv2d(in_channels, out_channels, bias=False, **kwargs)
    self.bn = BatchNorm2d(out_channels, eps=0.001)
        
class FID_Inception_A:
  def __init__(self, in_channels, pool_features):
    self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
    self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
    self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)
    self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
    self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
    self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
    self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

  def __call__(self, x:Tensor):
    x1 = self.branch1x1(x)
    
    x2 = self.branch5x5_1(x)
    x2 = self.branch5x5_2(x2)
    
    x3 = self.branch3x3dbl_1(x)
    x3 = self.branch3x3dbl_2(x3)
    x3 = self.branch3x3dbl_3(x3)
    
    # add padding also
    # Do not use the padded zeros in calculation: count_include_pad=False
    x4 = Tensor.avg_pool2d(x, kernel_size=(3,3), stride=1, padding=1)
    x4 = self.branch5x5_2(x4)
    
    return Tensor.cat(x1,x2,x3,x4,dim=1)

      

class FID_Inception_B:
  def __init__(self, in_channels):
    self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)
    self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
    self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
    self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

  def __call__(self, x:Tensor):
    x1 = self.branch3x3(x)
    
    x2 = self.branch3x3dbl_1(x)
    x2 = self.branch3x3dbl_2(x2)
    x2 = self.branch3x3dbl_3(x2)
    
    x3 = Tensor.max_pool2d(x, kernel_size=(3,3), stride=2)
    return Tensor.cat(x1,x2,x3,dim=1)

class FID_Inception_C:
  def __init__(self, in_channels, c7):
    self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)
    
    self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
    self.branch7x7_2 = BasicConv2d(c7,c7, kernel_size=(1,7), padding=(0,3))
    self.branch7x7_3 = BasicConv2d(c7,192, kernel_size=(7,1), padding=(3,0))
    
    self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
    self.branch7x7dbl_2 = BasicConv2d(c7,c7, kernel_size=(7,1), padding=(3,0))
    self.branch7x7dbl_3 = BasicConv2d(c7,c7, kernel_size=(1,7), padding=(0,3))
    self.branch7x7dbl_4 = BasicConv2d(c7,c7, kernel_size=(7,1), padding=(3,0))
    self.branch7x7dbl_5 = BasicConv2d(c7,192, kernel_size=(1,7), padding=(0,3))
    
    self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)
  
  def __call__(self, x:Tensor):
    x1 = self.branch1x1(x)
    
    x2 = self.branch7x7_1(x)
    x2 = self.branch7x7_2(x2)
    x2 = self.branch7x7_3(x2)
    
    x3 = self.branch7x7dbl_1(x)
    x3 = self.branch7x7dbl_2(x3)
    x3 = self.branch7x7dbl_3(x3)
    x3 = self.branch7x7dbl_4(x3)
    x3 = self.branch7x7dbl_5(x3)
    
    # add padding
    # Do not use the padded zeros in calculation: count_include_pad=False
    x4 = Tensor.avg_pool2d(x, kernel_size=(3,3), stride=1, padding=1)
    x4 = self.branch_pool(x4)
    
    return Tensor.cat(x1,x2,x3,x4, dem=1)

class FID_Inception_D:
  def __init__(self, in_channels):
    self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
    self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)
    
    self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
    self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1,7), padding=(0,3))
    self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7,1), padding=(3,0))
    self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)
  
  def __call__(self, x:Tensor):
    x1 = self.branch3x3_1(x)
    x1 = self.branch3x3_2(x1)
    
    x2 = self.branch7x7x3_1(x)
    x2 = self.branch7x7x3_2(x2)
    x2 = self.branch7x7x3_3(x2)
    x2 = self.branch7x7x3_4(x2)
    
    x3 = Tensor.max_pool2d(x, kernel_size=(3,3), stride=2)
    
    return Tensor.cat(x1, x2, x3, dim = 1)
    
 
class FID_Inception_E:
  def __init__(self, in_channels):
    self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)
    self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
    self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1,3), padding=(0,1))
    self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3,1), padding=(1,0))
    
    self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
    self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
    self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1,3), padding=(0,1))
    self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3,1), padding=(1,0))
    
    self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)
    
  def forward_1(self, x:Tensor):
    x1 = self.branch1x1(x)
    
    x2 = self.branch3x3_1(x)
    x2 = Tensor.cat(self.branch3x3_2a(x2), self.branch3x3_2b(x2), dim=1)
    
    x3 = self.branch3x3dbl_1(x)
    x3 = self.branch3x3dbl_2(x3)
    x3 = Tensor.cat(self.branch3x3dbl_3a(x3), self.branch3x3dbl_3b(x3), dim=1)
    
    # add padding
    # Do not use the padded zeros in calculation: count_include_pad=False
    x4 = Tensor.avg_pool2d(x, kernel_size=(3,3), stride=1, padding=1)
    x4 = self.branch_pool(x4)
    
    return Tensor.cat(x1,x2,x3,x4, dim=1)

  
  def forward_2(self, x:Tensor):
    x1 = self.branch1x1(x)
    
    x2 = self.branch3x3_1(x)
    x2 = Tensor.cat(self.branch3x3_2a(x2), self.branch3x3_2b(x2), dim=1)
    
    x3 = self.branch3x3dbl_1(x)
    x3 = self.branch3x3dbl_2(x3)
    x3 = Tensor.cat(self.branch3x3dbl_3a(x3), self.branch3x3dbl_3b(x3), dim=1)
    
    x4 = Tensor.max_pool2d(x, kernel_size=(3,3), stride=1, padding=1)
    x4 = self.branch_pool(x4)
    
    return Tensor.cat(x1,x2,x3,x4, dim=1)
    


