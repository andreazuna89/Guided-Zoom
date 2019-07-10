import os
import sys
import numpy as np
import util
from skimage import transform
import matplotlib
import argparse
import copy

CAFFE_EB_path='./'
sys.path.append(CAFFE_EB_path + '/python/')
import caffe
caffe.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
args = parser.parse_args()

#------------------------------------------------------------------- ARGUMENTS


if args.dataset == 'dogs':
  path_conventional = 'DOGS_train_CNN/resnet-101/snapshot/'
  model_file_conventional = path_conventional + 'dogs_conventional.caffemodel'
  deploy_file_conventional = 'DOGS_train_CNN/resnet-101/deploy.prototxt'
  train_file = '/data/Stanford_Dogs_Dataset/train_list.txt'
  train_img_path = '/data/Stanford_Dogs_Dataset/Images/'
  cats_file = '/data/Stanford_Dogs_Dataset/catName.txt'
  image_dim = 448
  patch_size = 150
  arch = 'resnet101'
  topLayerName = 'fc120'
  topBlobName = 'fc120'
  secondTopLayerName = 'pool5'
  secondTopBlobName = 'pool5'
  outputLayerName = 'res4a'
  outputBlobName = 'res4a'
  image_mean = [99.71090862410335, 115.21564018884108, 121.41523744391372]

elif args.dataset == 'birds':
  path_conventional = 'CUB_train_CNN/resnet-101/snapshot/'
  model_file_conventional = path_conventional + 'birds_conventional.caffemodel' 
  deploy_file_conventional = 'CUB_train_CNN/resnet-101/deploy.prototxt'
  train_file = '/data/CUB_200_2011/train_list.txt'
  train_img_path = '/data/CUB_200_2011/images/'
  cats_file = '/data/CUB_200_2011/catName.txt'
  image_dim = 448
  patch_size = 150
  arch = 'resnet101'
  topLayerName = 'fc200'
  topBlobName = 'fc200'
  secondTopLayerName = 'pool5'
  secondTopBlobName = 'pool5'
  outputLayerName = 'res4a'
  outputBlobName = 'res4a'
  image_mean = [110.07684660581486 , 127.38819773805534, 123.89104025414235 ]

elif args.dataset == 'cars':
  path_conventional = 'CARS_train_CNN/resnet-101/snapshot/'
  model_file_conventional = path_conventional + 'cars_conventional.caffemodel' 
  deploy_file_conventional = 'CARS_train_CNN/resnet-101/deploy.prototxt'
  train_file = '/data/Stanford_Cars_Dataset_New/train_list.txt'
  train_img_path = '/data/Stanford_Cars_Dataset_New/'
  cats_file = '/data/Stanford_Cars_Dataset_New/CARS_catName.txt'
  image_dim = 448
  patch_size = 150
  arch = 'resnet101'
  topLayerName = 'fc196'
  topBlobName = 'fc196'
  secondTopLayerName = 'pool5'
  secondTopBlobName = 'pool5'
  outputLayerName = 'res4a'
  outputBlobName = 'res4a'
  image_mean = [115.86383927342226, 117.13959748065012, 119.80923421042739]

elif args.dataset == 'air':
  path_conventional = 'AIRCRAFT_train_CNN_andrea/resnet-101/snapshot/'
  model_file_conventional = path_conventional + 'air_conventional.caffemodel'
  deploy_file_conventional = 'AIRCRAFT_train_CNN_andrea/resnet-101/deploy.prototxt'
  train_file =  '/data/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/train_list.txt'
  train_img_path = '/data/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images/'
  cats_file = '/data/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/catName.txt'
  image_dim = 448
  patch_size = 150
  arch = 'resnet101'
  topLayerName = 'fc100'
  topBlobName = 'fc100'
  secondTopLayerName = 'pool5'
  secondTopBlobName = 'pool5'
  outputLayerName = 'res4a'
  outputBlobName = 'res4a' 
  image_mean = [ 136.110, 130.027, 122.008] 
  
#--------------------------------------------------------------------------------

net = caffe.Net(deploy_file_conventional, model_file_conventional, caffe.TEST)

#----CREATE CATEGORY FOLDERS
cat_lines = open(cats_file, 'r')
cats = cat_lines.readlines()
cat_lines.close()
black_patch_dimension=85

# create patch folders
if not os.path.exists(args.dataset + '_multiple_patches_' + str(patch_size) + '_' + arch + '_res_' + str(image_dim) + '/'):
  os.mkdir(args.dataset + '_multiple_patches_' + str(patch_size) + '_' + arch + '_res_' + str(image_dim) + '/')
for cat in cats:  
    if not os.path.exists(args.dataset + '_multiple_patches_' + str(patch_size) + '_' + arch + '_res_' + str(image_dim) + '/' + cat[:-1]):
      os.mkdir(args.dataset + '_multiple_patches_' + str(patch_size) + '_' + arch + '_res_' + str(image_dim) + '/' + cat[:-1])

imgs_file = open(train_file,'r')
imgs_path = train_img_path
imgs_lines = imgs_file.readlines()
imgs_file.close()

resize = (475, 475) #(image_dim, image_dim)
shift_x = (resize[0] - image_dim)/2
shift_y = (resize[1] - image_dim)/2
flow= False
batch_size=5

#---- CREATE TRANSFORMER
shape = (batch_size,3,image_dim,image_dim)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_raw_scale('data', 255)

if flow:
  image_mean = [128, 128, 128]
channel_mean = np.zeros((3,image_dim,image_dim))
for channel_index, mean_val in enumerate(image_mean):
  channel_mean[channel_index, ...] = mean_val
transformer.set_mean('data', channel_mean)
transformer.set_channel_swap('data', (2, 1, 0))
transformer.set_transpose('data', (2, 0, 1))

 
for r in net.params.keys():
    if r[:5] == 'scale':
        if np.any(net.params[r][0].data<0):
            net.params['res' + r[5:]][0].data[net.params[r][0].data<0,...] *= -1
            net.params[r][0].data[...] = np.abs(net.params[r][0].data)


for v in range(0,len(imgs_lines),batch_size):
  
  caffe.set_mode_gpu()

  gt_labels = []
  original_data = []
  
  end_point = min(v+batch_size, len(imgs_lines))
  data = []
  for vv in range(v,end_point):
    frame_path = imgs_path + '/' + imgs_lines[vv].split(' ')[0]
    gt_labels.append(int(imgs_lines[vv].split(' ')[1][:-1]))
    #------ PREPROCESS IMAGE
    data_in = caffe.io.load_image(frame_path)
    if not ((data_in.shape[0] == resize[0]) & (data_in.shape[1] == resize[1])):
      data_in = caffe.io.resize_image(data_in, resize)
    shift_data_in = data_in[shift_x:shift_x+image_dim,shift_y:shift_y+image_dim,:] 
    processed_image = transformer.preprocess('data',shift_data_in)
    #------
    original_data.append(shift_data_in)
    
    data.append(processed_image)

  #----FWD
  net.blobs['data'].reshape(end_point-v,3,image_dim,image_dim)
  net.blobs['data'].data[...] = data[0:end_point-v]
  net.forward()
  #----



  #----BWD using EB
  caffe.set_mode_eb_gpu()
  attMaps = []

 
  net.blobs[topBlobName].diff[...] = 0
  for j in range(net.blobs['probs'].data.shape[0]):
    tagID = gt_labels[j]
    net.blobs[topBlobName].diff[j,tagID] = 1

  # invert the top layer weights
  net.params[topLayerName][0].data[...] *= -1
  out = net.backward(start = topLayerName, end = secondTopLayerName)
  buff = net.blobs[secondTopBlobName].diff.copy()

  # invert back
  net.params[topLayerName][0].data[...] *= -1 
  out = net.backward(start = topLayerName, end = secondTopLayerName)

  # compute the contrastive signal
  net.blobs[secondTopBlobName].diff[...] -= buff

  out = net.backward(start = secondTopLayerName, end = outputLayerName)
  attMap = np.maximum(net.blobs[outputBlobName].diff.sum(1), 0)
  #----
  
  #----SAVE PATCHES OF THE MAIN EVIDENCE OF IMGS IN THIS BATCH
  data=[]  
  original_data_2=[]
  for j in range(net.blobs['probs'].data.shape[0]): 
    temp=original_data[j].copy()
    original_data_2.append(temp)
    predict_label=np.argmax(net.blobs['probs'].data[j,:])
    heatMap = transform.resize(attMap[j,:], (image_dim,image_dim), order = 1, mode = 'edge')
    [x,y] = np.unravel_index(heatMap.argmax(), heatMap.shape)
    begin_x = max(0, x - patch_size / 2)
    end_x = min(image_dim - 1, x + patch_size / 2)
    begin_y = max(0, y - patch_size / 2)
    end_y = min(image_dim - 1, y + patch_size / 2)
    imgMap = original_data[j][begin_x:end_x,begin_y:end_y]
    imgMap = transform.resize(imgMap, (patch_size, patch_size), order = 3, mode = 'edge')
    matplotlib.image.imsave(args.dataset + '_multiple_patches_' + str(patch_size) + '_' + arch + '_res_' + str(image_dim) + '/' + cats[gt_labels[j]][:-1] + '/1_' + imgs_lines[v+j].split(' ')[0].split('/')[1], imgMap)
    original_data_2[j][max(0, x - black_patch_dimension / 2):min(image_dim - 1, x + black_patch_dimension / 2),max(0, y - black_patch_dimension / 2):min(image_dim - 1, y + black_patch_dimension / 2)]=0
    processed_image = transformer.preprocess('data',original_data_2[j])
    data.append(processed_image)

  ##===========SECOND FORWARD WITH MASK OVER THE IMAGE
  caffe.set_mode_gpu()
  net.blobs['data'].reshape(end_point-v,3,image_dim,image_dim)
  net.blobs['data'].data[...] = data[0:end_point-v]
  net.forward()
  attMaps = []
  caffe.set_mode_eb_gpu()
  net.blobs[topBlobName].diff[...] = 0
  for j in range(net.blobs['probs'].data.shape[0]):
      tagID = gt_labels[j]
      net.blobs[topBlobName].diff[j,tagID] = 1

  # invert the top layer weights
  net.params[topLayerName][0].data[...] *= -1
  out = net.backward(start = topLayerName, end = secondTopLayerName)
  buff = net.blobs[secondTopBlobName].diff.copy()

  # invert back
  net.params[topLayerName][0].data[...] *= -1 
  out = net.backward(start = topLayerName, end = secondTopLayerName)

  # compute the contrastive signal
  net.blobs[secondTopBlobName].diff[...] -= buff
  
  #----SAVE PATCHES OF THE SECOND MOST DISCRIMINATIVE EVIDENCE OF IMGS IN THIS BATCH
  out = net.backward(start = secondTopLayerName, end = outputLayerName)
  attMap = np.maximum(net.blobs[outputBlobName].diff.sum(1), 0)
  data=[]
  gt_labels_subset=[]
  for j in range(net.blobs['probs'].data.shape[0]): 
      predict_label=np.argmax(net.blobs['probs'].data[j,:])
      if predict_label==gt_labels[j]:
        gt_labels_subset.append(j)
        heatMap = transform.resize(attMap[j,:], (image_dim,image_dim), order = 1, mode = 'edge')
        [x,y] = np.unravel_index(heatMap.argmax(), heatMap.shape)
        begin_x = max(0, x - patch_size / 2)
        end_x = min(image_dim - 1, x + patch_size / 2)
        begin_y = max(0, y - patch_size / 2)
        end_y = min(image_dim - 1, y + patch_size / 2)
        imgMap = original_data[j][begin_x:end_x,begin_y:end_y]
        imgMap = transform.resize(imgMap, (patch_size, patch_size), order = 3, mode = 'edge')
        matplotlib.image.imsave(args.dataset + '_multiple_patches_' + str(patch_size) + '_' + arch + '_res_' + str(image_dim) + '/' + cats[gt_labels[j]][:-1] + '/2_' + imgs_lines[v+j].split(' ')[0].split('/')[1], imgMap)
        original_data_2[j][max(0, x - black_patch_dimension / 2):min(image_dim - 1, x + black_patch_dimension / 2),max(0, y - black_patch_dimension / 2):min(image_dim - 1, y + black_patch_dimension / 2)]=0
        processed_image = transformer.preprocess('data',original_data_2[j])
        data.append(processed_image)

  ##===========THIRD FORWARD WITH MASK OVER THE IMAGE
  if len(gt_labels_subset)!=0:
    caffe.set_mode_gpu()
    net.blobs['data'].reshape(len(gt_labels_subset),3,image_dim,image_dim)
    net.blobs['data'].data[...] = data[0:len(gt_labels_subset)]
    net.forward()
    attMaps = []
    caffe.set_mode_eb_gpu()
    net.blobs[topBlobName].diff[...] = 0
    for j in range(net.blobs['probs'].data.shape[0]):
      tagID = gt_labels[gt_labels_subset[j]]
      net.blobs[topBlobName].diff[j,tagID] = 1

  # invert the top layer weights
    net.params[topLayerName][0].data[...] *= -1
    out = net.backward(start = topLayerName, end = secondTopLayerName)
    buff = net.blobs[secondTopBlobName].diff.copy()

  # invert back
    net.params[topLayerName][0].data[...] *= -1 
    out = net.backward(start = topLayerName, end = secondTopLayerName)

  # compute the contrastive signal
    net.blobs[secondTopBlobName].diff[...] -= buff

	#----SAVE PATCHES OF THE THIRD MOST DISCRIMINATIVE EVIDENCE OF IMGS IN THIS BATCH
    out = net.backward(start = secondTopLayerName, end = outputLayerName)
    attMap = np.maximum(net.blobs[outputBlobName].diff.sum(1), 0)
 
    for j in range(net.blobs['probs'].data.shape[0]): 
      predict_label=np.argmax(net.blobs['probs'].data[j,:])
      if predict_label==gt_labels[gt_labels_subset[j]]:
        
        heatMap = transform.resize(attMap[j,:], (image_dim,image_dim), order = 1, mode = 'edge')
        [x,y] = np.unravel_index(heatMap.argmax(), heatMap.shape)
        begin_x = max(0, x - patch_size / 2)
        end_x = min(image_dim - 1, x + patch_size / 2)
        begin_y = max(0, y - patch_size / 2)
        end_y = min(image_dim - 1, y + patch_size / 2)
        imgMap = original_data[gt_labels_subset[j]][begin_x:end_x,begin_y:end_y]
        imgMap = transform.resize(imgMap, (patch_size, patch_size), order = 3, mode = 'edge')
        matplotlib.image.imsave(args.dataset + '_multiple_patches_' + str(patch_size) + '_' + arch + '_res_' + str(image_dim) + '/' + cats[gt_labels[gt_labels_subset[j]]][:-1] + '/3_' + imgs_lines[v+gt_labels_subset[j]].split(' ')[0].split('/')[1], imgMap)
        

