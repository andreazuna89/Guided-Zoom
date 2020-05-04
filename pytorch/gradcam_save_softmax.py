## SCRIPT TO COMPUTE AND SAVE THE SOFTMAX PREDICTED BY THE CONVENTIONAL AND THE EVIDENCE CNNs. THE EVIDENCE ARE EXTRACTED USING GRADCAM SALIENCY
## COMMAND TO USE:
#CUDA_VISIBLE_DEVICES=0 python gradcam_save_softmax.py --CONV PATH_TO_CONV_MODEL --EVID PATH_TO_EVID_MODEL --arch resnet101 --n-classes DATA_CLASSES --data PATH_TO_DATA

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
import pickle
import timeit
import utils
import operator
from skimage import transform
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch GUIDED ZOOM inference using GradCAM saliency')
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

parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--CONV', default=None, type=str,
                    help='Path to CONVENTIONAL CNN.')                  
parser.add_argument('--EVID', default=None, type=str,
                    help='Path to EVIDENCE CNN.')  


image_dim_conventional=448
image_dim_evidence=224
black_patch_dimension=85
patch_size = 150
start = timeit.default_timer()
top_classes = 5
batch_size_CONV=5



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
    ##########CONVENTIONAL CNN
    # create model
    print("=> creating model '{}'".format(args.arch))
    net_conventional = models.__dict__[args.arch]()
    print("=> creating new fc layer")
    net_conventional.fc = nn.Linear(2048 * 64, args.n_classes)
    # Split model in two parts
    if args.arch.startswith('resnet'):
        features_fn = nn.Sequential(*list(net_conventional.children())[:-2])
        classifier_fn = nn.Sequential(*(list(net_conventional.children())[-2:-1] + [Flatten()] + list(net_conventional.children())[-1:]))
    elif args.arch.startswith('vgg'):
        features_fn =net_conventional.features
        classifier_fn = nn.Sequential(*([Flatten()] + list(net_conventional.classifier.children())))
    else:
        print('Model architecture is not supported')
        return        


    if args.gpu is not None:
        net_conventional = net_conventional.cuda(args.gpu)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            net_conventional.features = torch.nn.DataParallel(net_conventional.features)
            net_conventional.cuda()
        else:
            net_conventional = torch.nn.DataParallel(net_conventional).cuda()

    # Load the model weights
    if args.CONV:
        if os.path.isfile(args.CONV):
            print("=> loading checkpoint '{}'".format(args.CONV))
            checkpoint = torch.load(args.CONV)
            args.start_epoch = checkpoint['epoch']
            acc = checkpoint['acc1']
            net_conventional.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}), acc: {:.2f}"
                  .format(args.CONV, checkpoint['epoch'], acc))
        else:
            print("=> no checkpoint found at '{}'".format(args.CONV))
    else:
        print('Provide model file')
        return
        
    
    net_conventional.eval()

    cudnn.benchmark = True

    # Data loading code
    dir = os.path.join(args.data, 'test')
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    data = datasets.ImageFolder(dir, transforms.Compose([
            transforms.Resize((image_dim_conventional, image_dim_conventional)),
            #transforms.CenterCrop(image_dim_conventional),
            transforms.ToTensor(),
            normalize,
        ]))
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size_CONV, shuffle=False,
        num_workers=args.workers, pin_memory=True)
        
    ##########EVIDENCE CNN
    # create model
    print("=> creating model '{}'".format(args.arch))
    net_evidence = models.__dict__[args.arch]()
    print("=> creating new fc layer")
    net_evidence.fc = nn.Linear(2048, args.n_classes)
    tr_evid=transforms.Compose([
            transforms.Resize((image_dim_evidence, image_dim_evidence)),
            transforms.ToTensor(),
            normalize,
        ])
     
    

    if args.gpu is not None:
        net_evidence = net_evidence.cuda(args.gpu)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            net_evidence.features = torch.nn.DataParallel(net_evidence.features)
            net_evidence.cuda()
        else:
            net_evidence = torch.nn.DataParallel(net_evidence).cuda()
    
    # Load the model weights
    if args.EVID:
        if os.path.isfile(args.EVID):
            print("=> loading checkpoint '{}'".format(args.EVID))
            checkpoint = torch.load(args.EVID)
            args.start_epoch = checkpoint['epoch']
            acc = checkpoint['acc1']
            net_evidence.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}), acc: {:.2f}"
                  .format(args.EVID, checkpoint['epoch'], acc))
        else:
            print("=> no checkpoint found at '{}'".format(args.EVID))
    else:
        print('Provide model file')
        return

    net_evidence.eval()
    
    ##===============================================================
    ##VARIABLES USED TO SAVE SOFTMAX OF CONVENTIONAL AND EVIDENCE CNN 
    conv=np.zeros((len(data),top_classes),dtype=float)
    p1=np.zeros((len(data),top_classes*3),dtype=float)
    p2=np.zeros((len(data),top_classes*3),dtype=float)
    p3=np.zeros((len(data),top_classes*3),dtype=float)
    p4=np.zeros((len(data),top_classes*3),dtype=float)
    p5=np.zeros((len(data),top_classes*3),dtype=float)
    top=np.zeros((len(data),top_classes),dtype=int)
    gt=np.zeros((len(data),1),dtype=int)
    ##===============================================================
    if not os.path.isdir('softmax_saved'):
            os.makedirs('softmax_saved')
    output = open( 'softmax_saved/'+args.data.split('/')[-2] +'_softmax_GRAD.pkl', 'wb')
    cats_file = args.data+'catName.txt'
    #----CREATE CATEGORY FOLDERS
    tags,tag2ID = utils.loadTags(cats_file)

    for i, (batch, label) in enumerate(tqdm(loader, total=len(loader))):
        gt[i*batch.shape[0]:i*batch.shape[0]+batch.shape[0],0]=label.numpy()
        out=net_conventional(batch)
        attMaps = []
        tagID_top=np.zeros((top_classes,out.shape[0]),dtype=int)
        batch_orig = denormalize_batch(batch).copy()
        for cntr in range(top_classes):
          attMaps_single_batch=np.zeros((out.shape[0],14,14),dtype=float)
          for j in range(out.shape[0]):
            
            tagScore = utils.getTagScore(out[j].detach().cpu().numpy(), tags, tag2ID)
            tagScore.sort(key = operator.itemgetter(1), reverse = True)
            tagID = tag2ID[tagScore[cntr][0]]
            tagID_top[cntr,j]=tagID
            sal = GradCAM(batch[j].unsqueeze(0), tagID, features_fn, classifier_fn)
            attMaps_single_batch[j,:]=sal  
          attMaps.append(attMaps_single_batch)   
        maxx_class=np.zeros((out.shape[0],top_classes),dtype=int)
        for j in range(out.shape[0]): 
          tagScore = utils.getTagScore(out[j].detach().cpu().numpy(), tags, tag2ID)
          tagScore.sort(key = operator.itemgetter(1), reverse = True)
          patch_data = torch.zeros(top_classes,3,image_dim_evidence,image_dim_evidence)

          for jj in range(top_classes):
            heatMap = Image.fromarray(attMaps[jj][j,:])
            heatMap = heatMap.resize((image_dim_conventional, image_dim_conventional), resample=Image.LINEAR)
            heatMap= np.array(heatMap)
            [x,y] = np.unravel_index(heatMap.argmax(), heatMap.shape)
            begin_x = max(0, x - patch_size / 2)
            end_x = min(image_dim_conventional - 1, x + patch_size / 2)
            begin_y = max(0, y - patch_size / 2)
            end_y = min(image_dim_conventional - 1, y + patch_size / 2)
            imgMap = batch_orig[j][begin_x:end_x,begin_y:end_y].copy()
            
            imgMap = transform.resize(imgMap, (image_dim_evidence, image_dim_evidence), order = 3, mode = 'edge')
            patch_in=torch.from_numpy(normalize_batch_patches(np.expand_dims(imgMap,axis=0))).float()
            patch_data[jj]=patch_in
            
          out_evid=net_evidence(patch_data.cuda())
          patch_tagScores = [] # prob vectors for every all top_classes
          
          #===SAVE CONVENTIONAL AND EVIDENCE CNN SOFTMAX
          for jj in range(top_classes):
            patch_tagScore = utils.getTagScore(out_evid[jj].detach().cpu().numpy(), tags, tag2ID)
            patch_tagScores.append(patch_tagScore)
            maxx_class[j,jj]=tag2ID[tagScore[jj][0]]
            conv[i*batch.shape[0]+j,jj]=tagScore[jj][1]
            top[i*batch.shape[0]+j,jj]=tag2ID[tagScore[jj][0]]
          
    	  for ss in range(top_classes):
            if ss==0:
              for sss in range(top_classes):
                p1[i*batch.shape[0]+j,sss]=patch_tagScores[ss][maxx_class[j,sss]][1]
            elif ss==1:
              for sss in range(top_classes):
                p2[i*batch.shape[0]+j,sss]=patch_tagScores[ss][maxx_class[j,sss]][1]
            elif ss==2:
              for sss in range(top_classes):
                p3[i*batch.shape[0]+j,sss]=patch_tagScores[ss][maxx_class[j,sss]][1]
            elif ss==3:
              for sss in range(top_classes):
                p4[i*batch.shape[0]+j,sss]=patch_tagScores[ss][maxx_class[j,sss]][1]
            elif ss==4:
              for sss in range(top_classes):
                p5[i*batch.shape[0]+j,sss]=patch_tagScores[ss][maxx_class[j,sss]][1]
          
        ##===========FIRST ADVERSARIAL ERASING ON IMAGES  
        attMaps2 = []
        for cntr in range(top_classes):
          attMaps_single_batch=np.zeros((out.shape[0],14,14),dtype=float)
          #data=torch.zeros(top_classes,3,image_dim_conventional,image_dim_conventional)
          for j in range(out.shape[0]): 
            original_data_evidence=batch_orig[j].copy()
            heatMap = Image.fromarray(attMaps[cntr][j])
            heatMap = heatMap.resize((image_dim_conventional, image_dim_conventional), resample=Image.LINEAR)
            heatMap= np.array(heatMap)
            [x,y] = np.unravel_index(heatMap.argmax(), heatMap.shape)
            original_data_evidence[max(0, x - black_patch_dimension / 2):min(image_dim_conventional - 1, x + black_patch_dimension / 2),max(0, y - black_patch_dimension / 2):min(image_dim_conventional - 1, y + black_patch_dimension / 2)]=0
            img=normalize_batch(np.expand_dims(original_data_evidence,axis=0))
            img= torch.from_numpy(img).float()
            sal = GradCAM(img, tagID_top[cntr,j], features_fn, classifier_fn)
            attMaps_single_batch[j,:]=sal
            
          attMaps2.append(attMaps_single_batch)   
        
        for j in range(out.shape[0]): 
          patch_data = torch.zeros(top_classes,3,image_dim_evidence,image_dim_evidence)
          for jj in range(top_classes):
            heatMap = Image.fromarray(attMaps2[jj][j,:])
            heatMap = heatMap.resize((image_dim_conventional, image_dim_conventional), resample=Image.LINEAR)
            heatMap= np.array(heatMap)
            [x,y] = np.unravel_index(heatMap.argmax(), heatMap.shape)
            begin_x = max(0, x - patch_size / 2)
            end_x = min(image_dim_conventional - 1, x + patch_size / 2)
            begin_y = max(0, y - patch_size / 2)
            end_y = min(image_dim_conventional - 1, y + patch_size / 2)
            imgMap = batch_orig[j][begin_x:end_x,begin_y:end_y].copy()
            imgMap = transform.resize(imgMap, (image_dim_evidence, image_dim_evidence), order = 3, mode = 'edge')
            patch_in=torch.from_numpy(normalize_batch_patches(np.expand_dims(imgMap,axis=0))).float()
            patch_data[jj]=patch_in
          out_evid=net_evidence(patch_data.cuda())
          patch_tagScores = [] # prob vectors for every all top_classes
          #===SAVE ALL EVIDENCE CNN SOFTMAX, AFTER FIRST ADVERSARIAL ERASING
          for jj in range(top_classes):
            patch_tagScore = utils.getTagScore(out_evid[jj].detach().cpu().numpy(), tags, tag2ID)
            patch_tagScores.append(patch_tagScore)      
          for ss in range(top_classes):
            if ss==0:
              for sss in range(top_classes):
                p1[i*batch.shape[0]+j,sss+top_classes]=patch_tagScores[ss][maxx_class[j,sss]][1]
            elif ss==1:
              for sss in range(top_classes):
                p2[i*batch.shape[0]+j,sss+top_classes]=patch_tagScores[ss][maxx_class[j,sss]][1]
            elif ss==2:
              for sss in range(top_classes):
                p3[i*batch.shape[0]+j,sss+top_classes]=patch_tagScores[ss][maxx_class[j,sss]][1]
            elif ss==3:
              for sss in range(top_classes):
                p4[i*batch.shape[0]+j,sss+top_classes]=patch_tagScores[ss][maxx_class[j,sss]][1]
            elif ss==4:
              for sss in range(top_classes):
                p5[i*batch.shape[0]+j,sss+top_classes]=patch_tagScores[ss][maxx_class[j,sss]][1]
          
        ##===========SECOND ADVERSARIAL ERASING ON IMAGE 
        
        attMaps3 = []
        for cntr in range(top_classes):
          attMaps_single_batch=np.zeros((out.shape[0],14,14),dtype=float)
          #data=torch.zeros(top_classes,3,image_dim_conventional,image_dim_conventional)
          for j in range(out.shape[0]): 
            original_data_evidence=batch_orig[j].copy()
            heatMap = Image.fromarray(attMaps[cntr][j])
            heatMap = heatMap.resize((image_dim_conventional, image_dim_conventional), resample=Image.LINEAR)
            heatMap= np.array(heatMap)
            [x1,y1] = np.unravel_index(heatMap.argmax(), heatMap.shape)
            heatMap = Image.fromarray(attMaps2[cntr][j])
            heatMap = heatMap.resize((image_dim_conventional, image_dim_conventional), resample=Image.LINEAR)
            heatMap= np.array(heatMap)
            [x2,y2] = np.unravel_index(heatMap.argmax(), heatMap.shape)
            original_data_evidence[max(0, x1 - black_patch_dimension / 2):min(image_dim_conventional - 1, x1 + black_patch_dimension / 2),max(0, y1 - black_patch_dimension / 2):min(image_dim_conventional - 1, y1 + black_patch_dimension / 2)]=0
            original_data_evidence[max(0, x2 - black_patch_dimension / 2):min(image_dim_conventional - 1, x2 + black_patch_dimension / 2),max(0, y2 - black_patch_dimension / 2):min(image_dim_conventional - 1, y2 + black_patch_dimension / 2)]=0
            img=normalize_batch(np.expand_dims(original_data_evidence,axis=0))
            img= torch.from_numpy(img).float()
            sal = GradCAM(img, tagID_top[cntr,j], features_fn, classifier_fn)
            attMaps_single_batch[j,:]=sal
          attMaps3.append(attMaps_single_batch)   
        for j in range(out.shape[0]): 
          patch_data = torch.zeros(top_classes,3,image_dim_evidence,image_dim_evidence)
          for jj in range(top_classes):
            heatMap = Image.fromarray(attMaps3[jj][j,:])
            heatMap = heatMap.resize((image_dim_conventional, image_dim_conventional), resample=Image.LINEAR)
            heatMap= np.array(heatMap)
            [x,y] = np.unravel_index(heatMap.argmax(), heatMap.shape)
            begin_x = max(0, x - patch_size / 2)
            end_x = min(image_dim_conventional - 1, x + patch_size / 2)
            begin_y = max(0, y - patch_size / 2)
            end_y = min(image_dim_conventional - 1, y + patch_size / 2)
            imgMap = batch_orig[j][begin_x:end_x,begin_y:end_y].copy()
            imgMap = transform.resize(imgMap, (image_dim_evidence, image_dim_evidence), order = 3, mode = 'edge')
            patch_in=torch.from_numpy(normalize_batch_patches(np.expand_dims(imgMap,axis=0))).float()
            patch_data[jj]=patch_in
          out_evid=net_evidence(patch_data.cuda())
          patch_tagScores = [] # prob vectors for every all top_classes
          #===SAVE ALL EVIDENCE CNN SOFTMAX, AFTER SECOND ADVERSARIAL ERASING
          for jj in range(top_classes):
            patch_tagScore = utils.getTagScore(out_evid[jj].detach().cpu().numpy(), tags, tag2ID)
            patch_tagScores.append(patch_tagScore)
	  
          for ss in range(top_classes):
            if ss==0:
              for sss in range(top_classes):
                p1[i*batch.shape[0]+j,sss+2*top_classes]=patch_tagScores[ss][maxx_class[j,sss]][1]
            elif ss==1:
              for sss in range(top_classes):
                p2[i*batch.shape[0]+j,sss+2*top_classes]=patch_tagScores[ss][maxx_class[j,sss]][1]
            elif ss==2:
              for sss in range(top_classes):
                p3[i*batch.shape[0]+j,sss+2*top_classes]=patch_tagScores[ss][maxx_class[j,sss]][1]
            elif ss==3:
              for sss in range(top_classes):
                p4[i*batch.shape[0]+j,sss+2*top_classes]=patch_tagScores[ss][maxx_class[j,sss]][1]
            elif ss==4:
              for sss in range(top_classes):
                p5[i*batch.shape[0]+j,sss+2*top_classes]=patch_tagScores[ss][maxx_class[j,sss]][1]
    stop = timeit.default_timer()
    print('Time: ', stop - start) 
    pickle.dump({'conv': conv,'top':top,'gt': gt,'p1':p1,'p2':p2,'p3':p3,'p4':p4,'p5':p5}, output)
          
def normalize_batch(inp, title=None, **kwargs):
    """Imshow for Tensor."""
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225]) 
    inp = (inp - MEAN)/STD
    inp = inp.transpose((0, 3, 1, 2))	
    return inp	

def normalize_batch_patches(inp, title=None, **kwargs):
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
