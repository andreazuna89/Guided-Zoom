import os
import sys
import numpy as np
import util
from skimage import transform
import matplotlib
import operator
import argparse
import timeit
CAFFE_EB_path='./'
sys.path.append(CAFFE_EB_path + '/python/')
import caffe
import pickle


parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
args = parser.parse_args()


#---- ARGUMENTS
top_classes = 5
black_patch_dimension =85
if args.dataset == 'dogs':
  
  #-- Conventional CNN
  path_conventional = 'DOGS_train_CNN/resnet-101/snapshot/'
  model_file_conventional = path_conventional + 'dogs_conventional.caffemodel'
  deploy_file_conventional = 'DOGS_train_CNN/resnet-101/deploy.prototxt'
  
  #-- Evidence CNN
  path_evidence = 'DOGS_train_CNN/resnet-101/'
  model_file_evidence = path_evidence + 'snapshot/dogs_evidence.caffemodel'
  deploy_file_evidence = path_evidence + 'deploy_patches.prototxt'

  val_file = '/data/Stanford_Dogs_Dataset/test_list.txt'
  val_img_path = '/data/Stanford_Dogs_Dataset/Images/' 
  cats_file = '/data/Stanford_Dogs_Dataset/catName.txt'
  image_dim_conventional = 448
  image_dim_evidence = 224
  patch_size = 150
  topLayerName = 'fc120'
  topBlobName = 'fc120'
  secondTopLayerName = 'pool5'
  secondTopBlobName = 'pool5'
  outputLayerName = 'res4a'
  outputBlobName = 'res4a'
  image_mean = [99.71090862410335, 115.21564018884108, 121.41523744391372] #DOGS

elif args.dataset == 'birds':
  path_conventional = 'CUB_train_CNN/resnet-101/snapshot/'
  model_file_conventional = path_conventional + 'birds_conventional.caffemodel' 
  deploy_file_conventional = 'CUB_train_CNN/resnet-101/deploy.prototxt'

  path_evidence = 'CUB_train_CNN/resnet-101/'
  model_file_evidence = path_evidence + 'snapshot/birds_evidence.caffemodel'
  deploy_file_evidence = path_evidence + 'deploy_patches.prototxt'

  val_file = '/data/CUB_200_2011/test_list.txt'
  val_img_path = '/data/CUB_200_2011/images/' 
  cats_file = '/data/CUB_200_2011/catName.txt'
  image_dim_conventional = 448
  image_dim_evidence = 224
  patch_size = 150
  topLayerName = 'fc200'
  topBlobName = 'fc200'
  secondTopLayerName = 'pool5'
  secondTopBlobName = 'pool5'
  outputLayerName = 'res4a'
  outputBlobName = 'res4a'
  image_mean = [110.07684660581486 , 127.38819773805534, 123.89104025414235 ] #BIRDS

elif args.dataset == 'cars':
  path_conventional = 'CARS_train_CNN/resnet-101/snapshot/'
  model_file_conventional = path_conventional + 'cars_conventional.caffemodel' 
  deploy_file_conventional = 'CARS_train_CNN/resnet-101/deploy.prototxt'

  path_evidence = 'CARS_train_CNN/resnet-101/'
  model_file_evidence = path_evidence + 'snapshot/cars_evidence.caffemodel'
  deploy_file_evidence = path_evidence + 'deploy_patches.prototxt'

  val_file = '/data/Stanford_Cars_Dataset/test_list.txt'
  val_img_path = '/data/Stanford_Cars_Dataset/cars_test/'
  cats_file = '/data/Stanford_Cars_Dataset/catName.txt'
  image_dim_conventional = 448
  image_dim_evidence = 224
  patch_size = 150
  topLayerName = 'fc196'
  topBlobName = 'fc196'
  secondTopLayerName = 'pool5'
  secondTopBlobName = 'pool5'
  outputLayerName = 'res4a'
  outputBlobName = 'res4a'
  image_mean = [115.86383927342226, 117.13959748065012, 119.80923421042739] #CARS

  
elif args.dataset == 'air':
  path_conventional = 'AIRCRAFT_train_CNN_andrea/resnet-101/snapshot/'
  model_file_conventional = path_conventional + 'air_conventional.caffemodel'
  deploy_file_conventional = 'AIRCRAFT_train_CNN_andrea/resnet-101/deploy.prototxt'

  path_evidence = 'AIRCRAFT_train_CNN_andrea/resnet-101/'
  model_file_evidence = path_evidence + 'snapshot/air_evidence.caffemodel'
  deploy_file_evidence = path_evidence + 'deploy_patches.prototxt'

  val_file = '/data/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/test_list.txt'
  val_img_path = '/data/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images/'
  cats_file = '/data/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/catName.txt'
  image_dim_conventional = 448
  image_dim_evidence = 224
  patch_size = 150
  topLayerName = 'fc100'
  topBlobName = 'fc100'
  secondTopLayerName = 'pool5'
  secondTopBlobName = 'pool5'
  outputLayerName = 'res4a'
  outputBlobName = 'res4a' 
  image_mean = [ 136.110, 130.027, 122.008] #AIRCRAFT

#----CREATE CATEGORY FOLDERS
tags,tag2ID = util.loadTags(cats_file)
cat_lines = open(cats_file, 'r')
cats = cat_lines.readlines()
cat_lines.close()
#----

imgs_file = open(val_file,'r')
imgs_path = val_img_path
imgs_lines = imgs_file.readlines()
imgs_file.close()

net_conventional = caffe.Net(deploy_file_conventional, model_file_conventional, caffe.TEST)

net_evidence = caffe.Net(deploy_file_evidence, model_file_evidence, caffe.TEST)

#---- CREATE TRANSFORMER CONVENTIONAL CNN
resize_conventional = (475, 475)
shift_x_conventional = (resize_conventional[0] - image_dim_conventional)/2
shift_y_conventional = (resize_conventional[1] - image_dim_conventional)/2
flow= False
batch_size=5


shape_conventional = (batch_size,3,image_dim_conventional,image_dim_conventional)
transformer_conventional = caffe.io.Transformer({'data': net_conventional.blobs['data'].data.shape})
transformer_conventional.set_raw_scale('data', 255)

if flow:
  image_mean = [128, 128, 128]
channel_mean_conventional = np.zeros((3,image_dim_conventional,image_dim_conventional))
for channel_index, mean_val in enumerate(image_mean):
  channel_mean_conventional[channel_index, ...] = mean_val
transformer_conventional.set_mean('data', channel_mean_conventional)
transformer_conventional.set_channel_swap('data', (2, 1, 0))
transformer_conventional.set_transpose('data', (2, 0, 1))

#---- CREATE TRANSFORMER EVIDENCE CNN
resize_evidence = (256, 256)
shift_x_evidence = (resize_evidence[0] - image_dim_evidence)/2
shift_y_evidence = (resize_evidence[1] - image_dim_evidence)/2
flow= False

shape_evidence = (batch_size,3,image_dim_evidence,image_dim_evidence)
transformer_evidence = caffe.io.Transformer({'data': net_evidence.blobs['data'].data.shape})
transformer_evidence.set_raw_scale('data', 255)

device_net_conventional=0
device_net_evidence=1

if flow:
  image_mean = [128, 128, 128]
channel_mean_evidence = np.zeros((3,image_dim_evidence,image_dim_evidence))
for channel_index, mean_val in enumerate(image_mean):
  channel_mean_evidence[channel_index, ...] = mean_val
transformer_evidence.set_mean('data', channel_mean_evidence)
transformer_evidence.set_channel_swap('data', (2, 1, 0))
transformer_evidence.set_transpose('data', (2, 0, 1))

tot_data=0

start = timeit.default_timer()

##===============================================================
##VARIABLES USED TO SAVE SOFTMAX OF CONVENTIONAL AND EVIDENCE CNN 
conv=np.zeros((len(imgs_lines),top_classes),dtype=float)
p1=np.zeros((len(imgs_lines),top_classes*3),dtype=float)
p2=np.zeros((len(imgs_lines),top_classes*3),dtype=float)
p3=np.zeros((len(imgs_lines),top_classes*3),dtype=float)
p4=np.zeros((len(imgs_lines),top_classes*3),dtype=float)
p5=np.zeros((len(imgs_lines),top_classes*3),dtype=float)
top=np.zeros((len(imgs_lines),top_classes),dtype=int)
gt=np.zeros((len(imgs_lines),1),dtype=int)
##===============================================================
output = open( args.dataset +'_softmax.pkl', 'wb')

for v in range(0,len(imgs_lines),batch_size):
  
  caffe.set_mode_gpu()
  gt_labels = []
  original_data = []
  predict_labels = []
  patches_predict_labels_first_evidence = []
  patches_predict_labels_second_evidence =[]
  patches_predict_labels_second_evidence_weight =[]
  patches_predict_labels_majority=[]
  end_point = min(v+batch_size, len(imgs_lines))
  data = []
  for vv in range(v,end_point):
    gt[tot_data]=int(imgs_lines[vv].split(' ')[1][:-1])
    tot_data+=1
    frame_path = imgs_path + '/' + imgs_lines[vv].split(' ')[0]
    gt_labels.append(int(imgs_lines[vv].split(' ')[1][:-1]))
    #------ PREPROCESS IMAGE
    data_in = caffe.io.load_image(frame_path)
    
    if not ((data_in.shape[0] == resize_conventional[0]) & (data_in.shape[1] == resize_conventional[1])):
      data_in = caffe.io.resize_image(data_in, resize_conventional)
    shift_data_in = data_in[shift_x_conventional:shift_x_conventional+image_dim_conventional,shift_y_conventional:shift_y_conventional+image_dim_conventional,:] 
    processed_image = transformer_conventional.preprocess('data',shift_data_in)
    
    #-------------------
    original_data.append(shift_data_in)
    data.append(processed_image)
  
  #----FWD
  caffe.set_device(device_net_conventional)
  net_conventional.blobs['data'].reshape(end_point-v,3,image_dim_conventional,image_dim_conventional)
  net_conventional.blobs['data'].data[...] = data[0:end_point-v]
  net_conventional.forward()
  #----
 
  #----BWD using EB
  caffe.set_mode_eb_gpu() 
  
  attMaps = []
  tagID_top=np.zeros((top_classes,net_conventional.blobs['probs'].data.shape[0]),dtype=int)
  for cntr in range(top_classes):
    net_conventional.blobs[topBlobName].diff[...] = 0
    for j in range(net_conventional.blobs['probs'].data.shape[0]):
      tagScore = util.getTagScore(net_conventional.blobs[topLayerName].data[j,:], tags, tag2ID)
      tagScore.sort(key = operator.itemgetter(1), reverse = True)
      if cntr==0:
        predict_labels.append(np.argmax(net_conventional.blobs['probs'].data[j,:].copy()))
      tagID = tag2ID[tagScore[cntr][0]]
      tagID_top[cntr,j]=tagID
      net_conventional.blobs[topBlobName].diff[j,tagID] = 1

    # invert the top layer weights
    net_conventional.params[topLayerName][0].data[...] *= -1
    out = net_conventional.backward(start = topLayerName, end = secondTopLayerName)
    buff = net_conventional.blobs[secondTopBlobName].diff.copy()

    # invert back
    net_conventional.params[topLayerName][0].data[...] *= -1 
    out = net_conventional.backward(start = topLayerName, end = secondTopLayerName)

    # compute the contrastive signal
    net_conventional.blobs[secondTopBlobName].diff[...] -= buff

    out = net_conventional.backward(start = secondTopLayerName, end = outputLayerName)
    attMaps.append(np.maximum(net_conventional.blobs[outputBlobName].diff.sum(1), 0))
  #----
  
  # in attMap WE HAVE top_classes attention maps for each image in the batch 
  
  maxx_class=np.zeros((net_conventional.blobs['probs'].data.shape[0],top_classes),dtype=int)
  for j in range(net_conventional.blobs['probs'].data.shape[0]): 
    # AGGREGATE
    tagScore = util.getTagScore(net_conventional.blobs[topLayerName].data[j,:].copy(), tags, tag2ID)
    tagScore.sort(key = operator.itemgetter(1), reverse = True)
    patch_data = []

    for jj in range(top_classes):
      heatMap = transform.resize(attMaps[jj][j,:], (image_dim_conventional,image_dim_conventional), order = 1, mode = 'edge')
	  
      [x,y] = np.unravel_index(heatMap.argmax(), heatMap.shape)
      begin_x = max(0, x - patch_size / 2)
      end_x = min(image_dim_conventional - 1, x + patch_size / 2)
      begin_y = max(0, y - patch_size / 2)
      end_y = min(image_dim_conventional - 1, y + patch_size / 2)
      imgMap = original_data[j][begin_x:end_x,begin_y:end_y].copy()
      imgMap = transform.resize(imgMap, (patch_size, patch_size), order = 3, mode = 'edge')
      
      patch_in = imgMap
      if not ((patch_in.shape[0] == resize_evidence[0]) & (data_in.shape[1] == resize_evidence[1])):
        patch_in = caffe.io.resize_image(patch_in, resize_evidence)
      shift_patch_in = patch_in[shift_x_evidence:shift_x_evidence+image_dim_evidence,shift_y_evidence:shift_y_evidence+image_dim_evidence,:].copy()
      processed_patch_image = transformer_evidence.preprocess('data',shift_patch_in)
      patch_data.append(processed_patch_image)
	  

    #----FWD
    caffe.set_device(device_net_evidence)
    caffe.set_mode_gpu()
    
    net_evidence.blobs['data'].reshape(top_classes,3,image_dim_evidence,image_dim_evidence)
    net_evidence.blobs['data'].data[...] = patch_data[0:top_classes]
    
    net_evidence.forward()
    #----

    patch_tagScores = [] # prob vectors for every all top_classes
    for jj in range(top_classes):
      patch_tagScore = util.getTagScore(net_evidence.blobs[topLayerName +'_new'].data[jj,:].copy(), tags, tag2ID)
      
     
      patch_tagScores.append(patch_tagScore)
      
   
    
    #===SAVE CONVENTIONAL AND EVIDENCE CNN SOFTMAX
    
    for ss in range(top_classes):
          
      maxx_class[j,ss]=tag2ID[tagScore[ss][0]]
      conv[v+j,ss]=tagScore[ss][1]
      top[v+j,ss]=tag2ID[tagScore[ss][0]]
    	  
    for ss in range(top_classes):
      
      if ss==0:
        for sss in range(top_classes):
          p1[v+j,sss]=patch_tagScores[ss][maxx_class[j,sss]][1]
      elif ss==1:
        for sss in range(top_classes):
          p2[v+j,sss]=patch_tagScores[ss][maxx_class[j,sss]][1]
      elif ss==2:
        for sss in range(top_classes):
          p3[v+j,sss]=patch_tagScores[ss][maxx_class[j,sss]][1]
      elif ss==3:
        for sss in range(top_classes):
          p4[v+j,sss]=patch_tagScores[ss][maxx_class[j,sss]][1]
      elif ss==4:
        for sss in range(top_classes):
          p5[v+j,sss]=patch_tagScores[ss][maxx_class[j,sss]][1]
    
    
    
  ##===========FIRST ADVERSARIAL ERASING ON IMAGES
  
  
  
  attMaps2 = []
  for cntr in range(top_classes):
    original_data_evidence=[]
    data=[]
    for j in range(net_conventional.blobs['probs'].data.shape[0]): 
      temp=original_data[j].copy()
      original_data_evidence.append(temp)
      heatMap = transform.resize(attMaps[cntr][j], (image_dim_conventional,image_dim_conventional), order = 1, mode = 'edge')
      [x,y] = np.unravel_index(heatMap.argmax(), heatMap.shape)
      original_data_evidence[j][max(0, x - black_patch_dimension / 2):min(image_dim_conventional - 1, x + black_patch_dimension / 2),max(0, y - black_patch_dimension / 2):min(image_dim_conventional - 1, y + black_patch_dimension / 2)]=0
      processed_image = transformer_conventional.preprocess('data',original_data_evidence[j])
      data.append(processed_image)

	##===========SECOND FORWARD WITH MASK OVER THE IMAGE
    
    caffe.set_mode_gpu() 
    caffe.set_device(device_net_conventional)
    net_conventional.blobs['data'].reshape(end_point-v,3,image_dim_conventional,image_dim_conventional)
    net_conventional.blobs['data'].data[...] = data[0:end_point-v]
    net_conventional.forward()
    
    
    caffe.set_mode_eb_gpu()
    net_conventional.blobs[topBlobName].diff[...] = 0
    for j in range(net_conventional.blobs['probs'].data.shape[0]):
      net_conventional.blobs[topBlobName].diff[j,tagID_top[cntr,j]] = 1
    
    net_conventional.params[topLayerName][0].data[...] *= -1
    
    out = net_conventional.backward(start = topLayerName, end = secondTopLayerName)
    
    buff = net_conventional.blobs[secondTopBlobName].diff.copy()
    
    # invert back
    net_conventional.params[topLayerName][0].data[...] *= -1 
    out = net_conventional.backward(start = topLayerName, end = secondTopLayerName)

    # compute the contrastive signal
    net_conventional.blobs[secondTopBlobName].diff[...] -= buff
   
    out = net_conventional.backward(start = secondTopLayerName, end = outputLayerName)
    attMaps2.append(np.maximum(net_conventional.blobs[outputBlobName].diff.sum(1), 0))
  
  # --------------------------------------------------------------------------------

  
  for j in range(net_conventional.blobs['probs'].data.shape[0]): 
    
    patch_data = []

    for jj in range(top_classes):
      heatMap = transform.resize(attMaps2[jj][j,:], (image_dim_conventional,image_dim_conventional), order = 1, mode = 'edge')
	  
      [x,y] = np.unravel_index(heatMap.argmax(), heatMap.shape)
      begin_x = max(0, x - patch_size / 2)
      end_x = min(image_dim_conventional - 1, x + patch_size / 2)
      begin_y = max(0, y - patch_size / 2)
      end_y = min(image_dim_conventional - 1, y + patch_size / 2)
      imgMap = original_data[j][begin_x:end_x,begin_y:end_y].copy()
      imgMap = transform.resize(imgMap, (patch_size, patch_size), order = 3, mode = 'edge')
      
      patch_in = imgMap
      if not ((patch_in.shape[0] == resize_evidence[0]) & (data_in.shape[1] == resize_evidence[1])):
        patch_in = caffe.io.resize_image(patch_in, resize_evidence)
      shift_patch_in = patch_in[shift_x_evidence:shift_x_evidence+image_dim_evidence,shift_y_evidence:shift_y_evidence+image_dim_evidence,:]
      processed_patch_image = transformer_evidence.preprocess('data',shift_patch_in)
      patch_data.append(processed_patch_image)
   
    
    caffe.set_mode_gpu()
    caffe.set_device(device_net_evidence)
    #----FWD
    net_evidence.blobs['data'].reshape(top_classes,3,image_dim_evidence,image_dim_evidence)
    net_evidence.blobs['data'].data[...] = patch_data[0:top_classes]
    net_evidence.forward()
    #----
    
    patch_tagScores = [] # prob vectors for every all top_classes
    for jj in range(top_classes):
      patch_tagScore = util.getTagScore(net_evidence.blobs[topLayerName +'_new'].data[jj,:], tags, tag2ID)
      
      patch_tagScores.append(patch_tagScore)
      
    #===SAVE ALL EVIDENCE CNN SOFTMAX, AFTER FIRST ADVERSARIAL ERASING
    
    for ss in range(top_classes):
      
      if ss==0:
        for sss in range(top_classes):
          p1[v+j,sss+top_classes]=patch_tagScores[ss][maxx_class[j,sss]][1]
      elif ss==1:
        for sss in range(top_classes):
          p2[v+j,sss+top_classes]=patch_tagScores[ss][maxx_class[j,sss]][1]
      elif ss==2:
        for sss in range(top_classes):
          p3[v+j,sss+top_classes]=patch_tagScores[ss][maxx_class[j,sss]][1]
      elif ss==3:
        for sss in range(top_classes):
          p4[v+j,sss+top_classes]=patch_tagScores[ss][maxx_class[j,sss]][1]
      elif ss==4:
        for sss in range(top_classes):
          p5[v+j,sss+top_classes]=patch_tagScores[ss][maxx_class[j,sss]][1]
	  
	 
  ##===========SECOND ADVERSARIAL ERASING ON IMAGE 
  
  attMaps3 = []
  for cntr in range(top_classes):
    original_data_evidence=[]
    data=[]
    for j in range(net_conventional.blobs['probs'].data.shape[0]): 
      temp=original_data[j].copy()
      original_data_evidence.append(temp)
      heatMap = transform.resize(attMaps[cntr][j], (image_dim_conventional,image_dim_conventional), order = 1, mode = 'edge')
      [x1,y1] = np.unravel_index(heatMap.argmax(), heatMap.shape)
      heatMap = transform.resize(attMaps2[cntr][j], (image_dim_conventional,image_dim_conventional), order = 1, mode = 'edge')
      [x2,y2] = np.unravel_index(heatMap.argmax(), heatMap.shape)
      
      original_data_evidence[j][max(0, x1 - black_patch_dimension / 2):min(image_dim_conventional - 1, x1 + black_patch_dimension / 2),max(0, y1 - black_patch_dimension / 2):min(image_dim_conventional - 1, y1 + black_patch_dimension / 2)]=0
      original_data_evidence[j][max(0, x2 - black_patch_dimension / 2):min(image_dim_conventional - 1, x2 + black_patch_dimension / 2),max(0, y2 - black_patch_dimension / 2):min(image_dim_conventional - 1, y2 + black_patch_dimension / 2)]=0
      processed_image = transformer_conventional.preprocess('data',original_data_evidence[j])
      data.append(processed_image)

    ##===========SECOND FORWARD WITH MASK OVER THE IMAGE
    
    caffe.set_mode_gpu() 
    caffe.set_device(device_net_conventional)
    net_conventional.blobs['data'].reshape(end_point-v,3,image_dim_conventional,image_dim_conventional)
    net_conventional.blobs['data'].data[...] = data[0:end_point-v]
    net_conventional.forward()
    
    
    caffe.set_mode_eb_gpu()
    
    net_conventional.blobs[topBlobName].diff[...] = 0
    for j in range(net_conventional.blobs['probs'].data.shape[0]):
      net_conventional.blobs[topBlobName].diff[j,tagID_top[cntr,j]] = 1
	  
    net_conventional.params[topLayerName][0].data[...] *= -1
    
    out = net_conventional.backward(start = topLayerName, end = secondTopLayerName)
    
    buff = net_conventional.blobs[secondTopBlobName].diff.copy()
    
    # invert back
    net_conventional.params[topLayerName][0].data[...] *= -1 
    out = net_conventional.backward(start = topLayerName, end = secondTopLayerName)

    # compute the contrastive signal
    net_conventional.blobs[secondTopBlobName].diff[...] -= buff
   
    out = net_conventional.backward(start = secondTopLayerName, end = outputLayerName)
    attMaps3.append(np.maximum(net_conventional.blobs[outputBlobName].diff.sum(1), 0))
  
  # --------------------------------------------------------------------------------
  
  
  for j in range(net_conventional.blobs['probs'].data.shape[0]): 
   
    
    patch_data = []

    

    for jj in range(top_classes):
      heatMap = transform.resize(attMaps3[jj][j,:], (image_dim_conventional,image_dim_conventional), order = 1, mode = 'edge')
	  
      [x,y] = np.unravel_index(heatMap.argmax(), heatMap.shape)
      begin_x = max(0, x - patch_size / 2)
      end_x = min(image_dim_conventional - 1, x + patch_size / 2)
      begin_y = max(0, y - patch_size / 2)
      end_y = min(image_dim_conventional - 1, y + patch_size / 2)
      imgMap = original_data[j][begin_x:end_x,begin_y:end_y]
      imgMap = transform.resize(imgMap, (patch_size, patch_size), order = 3, mode = 'edge')
      
      patch_in = imgMap
      if not ((patch_in.shape[0] == resize_evidence[0]) & (data_in.shape[1] == resize_evidence[1])):
        patch_in = caffe.io.resize_image(patch_in, resize_evidence)
      shift_patch_in = patch_in[shift_x_evidence:shift_x_evidence+image_dim_evidence,shift_y_evidence:shift_y_evidence+image_dim_evidence,:]
      processed_patch_image = transformer_evidence.preprocess('data',shift_patch_in)
      patch_data.append(processed_patch_image)
   
    
    caffe.set_mode_gpu()
    caffe.set_device(device_net_evidence)
    #----FWD
    net_evidence.blobs['data'].reshape(top_classes,3,image_dim_evidence,image_dim_evidence)
    net_evidence.blobs['data'].data[...] = patch_data[0:top_classes]
    net_evidence.forward()
    #----
    
    patch_tagScores = [] # prob vectors for every all top_classes
    for jj in range(top_classes):
      patch_tagScore = util.getTagScore(net_evidence.blobs[topLayerName +'_new'].data[jj,:], tags, tag2ID)
      
      patch_tagScores.append(patch_tagScore)
	  
    #===SAVE ALL EVIDENCE CNN SOFTMAX, AFTER SECOND ADVERSARIAL ERASING
    
    for ss in range(top_classes):
      
     
      if ss==0:
        for sss in range(top_classes):
          p1[v+j,sss+2*top_classes]=patch_tagScores[ss][maxx_class[j,sss]][1]
      elif ss==1:
        for sss in range(top_classes):
          p2[v+j,sss+2*top_classes]=patch_tagScores[ss][maxx_class[j,sss]][1]
      elif ss==2:
        for sss in range(top_classes):
          p3[v+j,sss+2*top_classes]=patch_tagScores[ss][maxx_class[j,sss]][1]
      elif ss==3:
        for sss in range(top_classes):
          p4[v+j,sss+2*top_classes]=patch_tagScores[ss][maxx_class[j,sss]][1]
      elif ss==4:
        for sss in range(top_classes):
          p5[v+j,sss+2*top_classes]=patch_tagScores[ss][maxx_class[j,sss]][1]
	
stop = timeit.default_timer()
print('Time: ', stop - start) 
pickle.dump({'conv': conv,'top':top,'gt': gt,'p1':p1,'p2':p2,'p3':p3,'p4':p4,'p5':p5}, output)
