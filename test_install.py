import torch
from torch.backends import cudnn

if __name__ == '__main__':
    print('cuda :', torch.cuda.is_available())
    x = torch.Tensor([10.0])
    x = x.cuda()
    print(x)
    print('cudnn :', cudnn.is_acceptable(x))

