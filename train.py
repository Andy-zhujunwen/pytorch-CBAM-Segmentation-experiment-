import os
import numpy as np
import glob
from utils.metrics import Evaluator
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
#from networks import PAN, ResNet50
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from data import make_data_loader
import argparse
from utils.metrics import Evaluator
from utils.saver import Saver
from utils.loss import SegmentationLosses
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
import model.resnet_cbam as resnet_cbam

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a FPN Semantic Segmentation network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='Cityscapes', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
					    help='starting epoch',
					    default=1, type=int)
    parser.add_argument('--epochs', dest='epochs',
					    help='number of iterations to train',
					    default=50, type=int)
    parser.add_argument('--save_dir', dest='save_dir',
					    help='directory to save models',
					    default=None,
					    nargs=argparse.REMAINDER)
    parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',default=True, type=bool)
    parser.add_argument('--batch_size', dest='batch_size',
					    help='batch_size',
					    default=4, type=int)
    # config optimization
    parser.add_argument('--o', dest='optimizer',
					    help='training optimizer',
					    default='sgd', type=str)
    parser.add_argument('--lr', dest='lr',
					    help='starting learning rate',
					    default=0.01, type=float)
    parser.add_argument('--weight_decay', dest='weight_decay',
                        help='weight_decay',
                        default=1e-5, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
					    help='step to do learning rate decay, uint is epoch',
					    default=50, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
					    help='learning rate decay ratio',
					    default=0.1, type=float)
    parser.add_argument('--base-size', type=int, default=1024,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')
    args = parser.parse_args()
    return args

class PolyLR(_LRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma every epoch. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, max_iter, power, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - self.last_epoch/self.max_iter) ** self.power
                for base_lr in self.base_lrs]


NUM_CLASS=19
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:{}'.format(device))
args = parse_args()
kwargs = {'num_workers': 0, 'pin_memory': True}
train_loader, val_loader, test_loader, num_class = make_data_loader(args, **kwargs)

my_model = resnet_cbam.resnet50_cbam(pretrained=False)
my_model = my_model.to(device)

weight = None
criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode='ce')
optimizer = torch.optim.SGD(my_model.parameters(), lr=args.lr, momentum=0, weight_decay=args.weight_decay)

optimizer_lr_scheduler = PolyLR(optimizer, max_iter=args.epochs, power=0.9)

evaluator = Evaluator(NUM_CLASS)

def train(epoch, optimizer, train_loader):
    my_model.train()
    for iteration, batch in enumerate(train_loader):
        image, target = batch['image'], batch['label']
        inputs = image.to(device)
        labels = target.to(device)

        # grad = 0
        my_model.zero_grad()

        inputs = Variable(inputs)
        labels = Variable(labels)
        out = my_model(inputs)
        #out = out[0]
        print('out:{}'.format(out.shape))
        #out_ss = F.interpolate(out, scale_factor=4, mode='nearest')

        loss_ss = criterion(out, labels.long())
        print('loss={}'.format(loss_ss))
        loss_ss.backward(torch.ones_like(loss_ss))
        optimizer.step()

        if iteration % 10 == 0:
            print("Epoch[{}]({}/{}):Loss:{:.4f}".format(epoch, iteration, len(train_loader),
                                                        loss_ss.data))

def validation(epoch,best_pred):
    my_model.eval()
    evaluator.reset()
    test_loss = 0.0
    for iteration,batch in enumerate(val_loader):
        image,target = batch['image'],batch['label']
        image = image.to(device)
        target = target.to(device)
        with torch.no_grad():
            out = my_model(image)
        #out_ss = F.interpolate(out_ss,scale_factor=4,mode='nearest')
        
        print('out_ss :{}'.format(out.shape))
        loss_ss = criterion(out,target.long())
        loss = loss_ss.item()
        test_loss += loss
        print('epoch:{},test loss:{}'.format(epoch,test_loss/(iteration+1)))

        pred = out.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        evaluator.add_batch(target, pred)
    # Fast test during the training
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    print('Validation:')
    print('[Epoch: %d, numImages: %5d]' % (epoch, iteration * args.batch_size + image.shape[0]))
    print("Acc:{:.5f}, Acc_class:{:.5f}, mIoU:{:.5f}, fwIoU:{:.5f}".format(Acc, Acc_class, mIoU, FWIoU))
    print('Loss: %.3f' % test_loss)

    new_pred = mIoU
    if new_pred > best_pred:
        print('(mIoU)new pred ={},old best pred={}'.format(new_pred,best_pred))
        best_pred = new_pred
        model_name = r'./save_models/'+r'CBAM_'+str(epoch)+r'_.pth'
        torch.save(my_model,model_name)
    return best_pred

if __name__ == '__main__':
    best_pred = 0.0
	if not os.path.exists('./save_models'):
		os.makedirs(r'./save_models')
    for epoch in range(args.epochs):
        optimizer_lr_scheduler.step(epoch)
        print('Epoch:{}'.format(epoch))
        train(epoch, optimizer, train_loader)
        if epoch %(2-1)==0:
            best_pred = validation(epoch,best_pred)








