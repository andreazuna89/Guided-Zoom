## SCRIPT FOR EVIDENCE POOL GENERATION. THE EVIDENCE ARE EXTRACTED USING GRADCAM SALIENCY
## COMMAND TO USE:
#CUDA_VISIBLE_DEVICES=0 python gradcam_evidence_pool_generation.py --model PATH_TO_CONV_MODEL --arch resnet101 --n-classes DATA_CLASSES --data PATH_TO_DATA

import argparse
import os
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch GUIDED ZOOM patches extraction using GradCAM saliency')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--n-classes', default=200, type=int,
                    help='Number of classes.')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to model (default: none)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
                   

image_dim=448
black_patch_dimension=85
patch_size = 150


class Flatten(nn.Module):
    """One layer module that flattens its input."""
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)


def main():
    global args
    args = parser.parse_args()

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()
    print("=> creating new fc layer")
    model.fc = nn.Linear(2048 * 64, args.n_classes)
    # Split model in two parts
    if args.arch.startswith('resnet'):
        features_fn = nn.Sequential(*list(model.children())[:-2])
        classifier_fn = nn.Sequential(*(list(model.children())[-2:-1] + [Flatten()] + list(model.children())[-1:]))
    elif args.arch.startswith('vgg'):
        features_fn = model.features
        classifier_fn = nn.Sequential(*([Flatten()] + list(model.classifier.children())))
    else:
        print('Model architecture is not supported')
        return        


    if args.gpu is not None:
        model = model.cuda(args.gpu)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    # Load the model weights
    if args.model:
        if os.path.isfile(args.model):
            print("=> loading checkpoint '{}'".format(args.model))
            checkpoint = torch.load(args.model)
            args.start_epoch = checkpoint['epoch']
            acc = checkpoint['acc1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}), acc: {:.2f}"
                  .format(args.model, checkpoint['epoch'], acc))
        else:
            print("=> no checkpoint found at '{}'".format(args.model))
    else:
        print('Provide model file')
        return
        

    model.eval()

    cudnn.benchmark = True

    # Data loading code
    dir = os.path.join(args.data, 'train')
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    data = datasets.ImageFolder(dir, transforms.Compose([
            transforms.Resize((448, 448)),
            #transforms.CenterCrop(448),
            transforms.ToTensor(),
            normalize,
        ]))
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)
 

        
    for folder in os.listdir(dir):
      if not os.path.isdir(os.path.join('GCAM_saliency',args.data.split('/')[-2], folder)):
            os.makedirs(os.path.join('GCAM_saliency',args.data.split('/')[-2], folder))
      if not os.path.isdir(os.path.join('GCAM_patches',args.data.split('/')[-2], folder)):
            os.makedirs(os.path.join('GCAM_patches',args.data.split('/')[-2], folder))
    for i, (img, label) in enumerate(tqdm(loader, total=len(loader))):
        folder, file = data.imgs[i][0].split('/')[-2:]
        out=model(img.cuda())
        counter=0
        #we take always the first evidence, then we take patches until the model predicts correctly, using at maximum 2 adversarial erasing
        while((torch.argmax(out).cpu()==label or counter==0) and counter<3):
          counter+=1
          sal = GradCAM(img, label, features_fn, classifier_fn)
          sal = Image.fromarray(sal)
          sal = sal.resize((448, 448), resample=Image.LINEAR)
          sal= np.array(sal)
          
          img = denormalize_batch(img)[0]
          if counter==1:
            img_orig = img.copy()
        
          [x,y] = np.unravel_index(sal.argmax(), sal.shape)
          begin_x = max(0, x - patch_size / 2)
          end_x = min(image_dim - 1, x + patch_size / 2)
          begin_y = max(0, y - patch_size / 2)
          end_y = min(image_dim - 1, y + patch_size / 2)
          plt.imsave(os.path.join('GCAM_patches', args.data.split('/')[-2],folder, str(counter)+'_'+file), img_orig[begin_x:end_x,begin_y:end_y])

        #---------------------------------------------------------------- SALIENCY
        
          attMap = sal
          attMap -= attMap.min()
          if attMap.max() > 0:
                attMap /= attMap.max()
        
          cmap = plt.get_cmap('jet')
          attMapV = cmap(attMap)
          attMapV = np.delete(attMapV, 3, 2)
          attMap = 1 * (1 - attMap ** 0.8).reshape(attMap.shape + (1,)) * img + (attMap ** 0.8).reshape(attMap.shape + (1,)) * attMapV
          plt.imsave(os.path.join('GCAM_saliency',args.data.split('/')[-2],folder,str(counter)+'_'+file), attMap)
        #----------------------------------------------------------------
          img[max(0, x - black_patch_dimension / 2):min(image_dim - 1, x + black_patch_dimension / 2),max(0, y - black_patch_dimension / 2):min(image_dim - 1, y + black_patch_dimension / 2)]=0
        
          img=normalize_batch(np.expand_dims(img,axis=0))
          img= torch.from_numpy(img).float()
          out=model(img.cuda())
          
def normalize_batch(inp, title=None, **kwargs):
    """Imshow for Tensor."""
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225]) 
    inp = (inp - MEAN)/STD
    inp = inp.transpose((0, 3, 1, 2))	
    return inp	


def denormalize_batch(inp, title=None, **kwargs):
    """Imshow for Tensor."""
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])
    inp = inp.numpy().transpose((0, 2, 3, 1))
    inp = STD * inp + MEAN
    inp = np.clip(inp, 0, 1)
    return inp


def GradCAM(img, c, features_fn, classifier_fn):
    feats = features_fn(img.cuda())
    _, N, H, W = feats.size()
    
    out = classifier_fn(feats)

    c_score = out[0, c]
    grads = torch.autograd.grad(c_score, feats)
    w = grads[0][0].mean(-1).mean(-1)
    sal = torch.matmul(w, feats.view(N, H*W))
    sal = sal.view(H, W).cpu().detach().numpy()
    sal = np.maximum(sal, 0)
    return sal


# Given label number returns class name
def get_class_name(c):
    labels = np.loadtxt(os.path.join(args.data, 'catName.txt'), str, delimiter='\n')
    return labels[c]


if __name__ == '__main__':
    main()
