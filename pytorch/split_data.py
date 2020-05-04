## SCRIPT FOR SPLIT THE DATA IN FOLDERS ACCORDINGLY TO THE DATA CLASSES

import pandas as pd
import os
from shutil import copy

dataset = ['CUB_200_2011', 'Stanford_Dogs_Dataset', 'Stanford_Cars_Dataset', 'fgvc-aircraft-2013b/fgvc-aircraft-2013b/data'][0]
data_dir = '/research/axns3/data/{}/'.format(dataset)
if dataset == 'fgvc-aircraft-2013b/fgvc-aircraft-2013b/data':
  data_out= '/research/axns3/data/Split/{}/'.format('fgvc-aircraft-2013b')
else:
  data_out= '/research/axns3/data/Split/{}/'.format(dataset)
if not os.path.isdir(data_out):
    os.mkdir(data_out)
if dataset == 'fgvc-aircraft-2013b/fgvc-aircraft-2013b/data':
    copy(data_dir + 'variants.txt', os.path.join(data_out, 'catName.txt'))
else:
    copy(data_dir + 'catName.txt', os.path.join(data_out, 'catName.txt'))

for mode in ['train', 'test']:
    if not os.path.isdir(os.path.join(data_out, mode)):
        os.mkdir(os.path.join(data_out, mode))
    df = pd.read_csv(data_dir + '{}_list.txt'.format(mode), sep=' ', header=None)
    df.columns = ['Path', 'ClassID']
    
    for i, (path, category) in df.iterrows():
        if dataset == 'Stanford_Cars_Dataset':
            if not os.path.isdir(os.path.join(data_out, mode, '%03d' % category)):
                os.mkdir(os.path.join(data_out, mode, '%03d' % category))
            copy(os.path.join(data_dir, path),
                 os.path.join(data_out, mode, '%03d' % category, path.split('/')[1]))
        elif dataset == 'fgvc-aircraft-2013b/fgvc-aircraft-2013b/data':
            if not os.path.isdir(os.path.join(data_out, mode, str(category))):
                os.mkdir(os.path.join(data_out, mode, '%03d' % category))
            copy(os.path.join(data_dir, 'images', path),
                 os.path.join(data_out, mode, '%03d' % category, path))
        elif dataset == 'Stanford_Dogs_Dataset':
            if not os.path.isdir(os.path.join(data_out, mode, path.split('/')[0])):
                os.mkdir(os.path.join(data_out, mode, path.split('/')[0]))
            copy(os.path.join(data_dir, 'Images', path),
                 os.path.join(data_out, mode, path))
        else:
            if not os.path.isdir(os.path.join(data_out, mode, path.split('/')[0])):
                os.mkdir(os.path.join(data_out, mode, path.split('/')[0]))
            copy(os.path.join(data_dir, 'images', path),
                 os.path.join(data_out, mode, path))
