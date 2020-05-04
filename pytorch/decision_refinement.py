## SCRIPT FOR THE FINAL DECISION REFINEMENT 
## COMMAND TO USE:
# python decision_refinement.py --dataset NAME_DATASET --top_classes TOP_CLASSES_CONSIDERED --EVID_type EVIDENCE_CNN_TYPE

import numpy as np
import cPickle
import argparse
import pickle


parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
parser.add_argument("--top_classes") #top classes
parser.add_argument("--EVID_type")
args = parser.parse_args()


with open('./softmax_saved/'+args.dataset + '_softmax_'+str(args.EVID_type)+'.pkl') as f:
  data_dict=cPickle.load(f)

tc = 5
top_classes = int(args.top_classes) #top_classes

conv = data_dict['conv']
top = data_dict['top']
gt = data_dict['gt']
p1 = data_dict['p1']
p2 = data_dict['p2']
p3 = data_dict['p3']
if tc == 5:
  p4 = data_dict['p4'] 
  p5 = data_dict['p5']

num = len(gt)
numf = float(len(gt))

top1 = len(np.where(gt[:,0]-top[:,0] == 0)[0])

topk = 0
for i in range(num):
  if gt[i][0] in top[i,0:top_classes]:
    topk = topk + 1

top1 = top1 / numf * 100
topk = topk / numf * 100

print('-----------------------------')
print('Dataset: ' + args.dataset)
print('Top-1 Accuracy: ' + str(top1))
print('Top-' + str(top_classes) + ' Accuracy: ' + str(topk) + '\n')

correct = 0
correctAdv = 0
correctAdv2 = 0
for i in range(num):

  if top_classes == 3:

    mean = (conv[i,0:3] + p1[i,0:3] + p2[i,0:3] + p3[i,0:3]) / 4
    meanAdv = (conv[i,0:3] + p1[i,0:3] + p2[i,0:3] + p3[i,0:3] + p1[i,tc:tc+3] + p2[i,tc:tc+3] + p3[i,tc:tc+3]) / 7 
    meanAdv2 = (conv[i,0:3] + p1[i,0:3] + p2[i,0:3] + p3[i,0:3] + p1[i,tc:tc+3] + p2[i,tc:tc+3] + p3[i,tc:tc+3] + p1[i,2*tc:2*tc+3] + p2[i,2*tc:2*tc+3] + p3[i,2*tc:2*tc+3]) / 10

  elif top_classes == 5:

    mean = (conv[i,0:5] + p1[i,0:5] + p2[i,0:5] + p3[i,0:5] + p4[i,0:5] + p5[i,0:5]) / 6
    meanAdv = (conv[i,0:5] + p1[i,0:5] + p2[i,0:5] + p3[i,0:5] + p4[i,0:5] + p5[i,0:5] + p1[i,tc:tc+5] + p2[i,tc:tc+5] + p3[i,tc:tc+5] + p4[i,tc:tc+5] + p5[i,tc:tc+5]) / 11
    meanAdv2 = (conv[i,0:5] + p1[i,0:5] + p2[i,0:5] + p3[i,0:5] + p4[i,0:5] + p5[i,0:5] + p1[i,tc:tc+5] + p2[i,tc:tc+5] + p3[i,tc:tc+5] + p4[i,tc:tc+5] + p5[i,tc:tc+5] + p1[i,2*tc:2*tc+5] + p2[i,2*tc:2*tc+5] + p3[i,2*tc:2*tc+5] + p4[i,2*tc:2*tc+5] + p5[i,2*tc:2*tc+5]) / 16

  if gt[i][0] == top[i, np.argmax(mean)]:
    correct = correct + 1
  if gt[i][0] == top[i, np.argmax(meanAdv)]:
    correctAdv = correctAdv + 1
  if gt[i][0] == top[i, np.argmax(meanAdv2)]:
    correctAdv2 = correctAdv2 + 1

correct = correct / numf * 100
correctAdv = correctAdv / numf * 100
correctAdv2 = correctAdv2 / numf * 100

print('---------Strategy 1----------')
print('Mean Pool w/out adv: ' + str(correct))
print('Mean Pool w/ adv: ' + str(correctAdv))
print('Mean Pool w/ adv2: ' + str(correctAdv2) + '\n')

correct = 0
correctAdv = 0
correctAdv2 = 0
for i in range(num):

  if top_classes == 3:

    maxx = np.max(np.array([[conv[i,0:3]], [p1[i,0:3]], [p2[i,0:3]], [p3[i,0:3]]]), axis=0)
    maxxAdv = np.max(np.array([[conv[i,0:3]], [p1[i,0:3]], [p2[i,0:3]], [p3[i,0:3]], [p1[i,tc:tc+3]], [p2[i,tc:tc+3]], [p3[i,tc:tc+3]]]), axis=0)
    maxxAdv2 = np.max(np.array([[conv[i,0:3]], [p1[i,0:3]], [p2[i,0:3]], [p3[i,0:3]], [p1[i,tc:tc+3]], [p2[i,tc:tc+3]], [p3[i,tc:tc+3]], [p1[i,2*tc:2*tc+3]], [p2[i,2*tc:2*tc+3]], [p3[i,2*tc:2*tc+3]]]), axis=0)

  elif top_classes == 5:

    maxx = np.max(np.array([[conv[i,0:tc]], [p1[i,0:tc]], [p2[i,0:tc]], [p3[i,0:tc]], [p4[i,0:tc]], [p5[i,0:tc]]]), axis=0)
    maxxAdv = np.max(np.array([[conv[i,0:tc]], [p1[i,0:tc]], [p2[i,0:tc]], [p3[i,0:tc]], [p4[i,0:tc]], [p5[i,0:tc]], [p1[i,tc:2*tc]], [p2[i,tc:2*tc]], [p3[i,tc:2*tc]], [p4[i,tc:2*tc]], [p5[i,tc:2*tc]]]), axis=0)
    maxxAdv2 = np.max(np.array([[conv[i,0:tc]], [p1[i,0:tc]], [p2[i,0:tc]], [p3[i,0:tc]], [p4[i,0:tc]], [p5[i,0:tc]], [p1[i,tc:2*tc]], [p2[i,tc:2*tc]], [p3[i,tc:2*tc]], [p4[i,tc:2*tc]], [p5[i,tc:2*tc]], [p1[i,2*tc:3*tc]], [p2[i,2*tc:3*tc]], [p3[i,2*tc:3*tc]], [p4[i,2*tc:3*tc]], [p5[i,2*tc:3*tc]]]), axis=0)

  if gt[i][0] == top[i, np.argmax(maxx)]:
    correct = correct + 1
  if gt[i][0] == top[i, np.argmax(maxxAdv)]:
    correctAdv = correctAdv + 1
  if gt[i][0] == top[i, np.argmax(maxxAdv2)]:
    correctAdv2 = correctAdv2 + 1

correct = correct / numf * 100
correctAdv = correctAdv / numf * 100
correctAdv2 = correctAdv2 / numf * 100

print('---------Strategy 2----------')
print('Argmax w/out adv: ' + str(correct))
print('Argmax w/ adv: ' + str(correctAdv))
print('Argmax w/ adv2: ' + str(correctAdv2) + '\n')



# weights[0] is the weighting parameter for conv
# weights[1] is the weighting parameter for the most discriminative evidence
# weights[2] is the weighting parameter for the next most discriminative evidence
# weights[3] is the weighting parameter for the next next most discriminative evidence
weights=[0.4,0.3,0.2,0.1]
correct = 0
correctAdv = 0
correctAdv2 = 0
for i in range(num):

          if top_classes == 3:

            tot = [weights[0]*conv[i,0] + weights[1]*p1[i,0], 
                   weights[0]*conv[i,1] + weights[1]*p2[i,1], 
                   weights[0]*conv[i,2] + weights[1]*p3[i,2]]

            totAdv = [weights[0]*conv[i,0] + weights[1]*p1[i,0] + weights[2]*p1[i,tc], 
                      weights[0]*conv[i,1] + weights[1]*p2[i,1] + weights[2]*p2[i,tc+1], 
                      weights[0]*conv[i,2] + weights[1]*p3[i,2] + weights[2]*p3[i,tc+2]]

            totAdv2 = [weights[0]*conv[i,0] + weights[1]*p1[i,0] + weights[2]*p1[i,tc] + weights[3]*p1[i,2*tc], 
                       weights[0]*conv[i,1] + weights[1]*p2[i,1] + weights[2]*p2[i,tc+1] + weights[3]*p2[i,2*tc+1], 
                       weights[0]*conv[i,2] + weights[1]*p3[i,2] + weights[2]*p3[i,tc+2] + weights[3]*p3[i,2*tc+2]]

          elif top_classes == 5:

            tot = [weights[0]*conv[i,0] + weights[1]*p1[i,0], 
                   weights[0]*conv[i,1] + weights[1]*p2[i,1], 
                   weights[0]*conv[i,2] + weights[1]*p3[i,2], 
                   weights[0]*conv[i,3] + weights[1]*p4[i,3], 
                   weights[0]*conv[i,4] + weights[1]*p5[i,4]]

            totAdv = [weights[0]*conv[i,0] + weights[1]*p1[i,0] + weights[2]*p1[i,tc], 
                      weights[0]*conv[i,1] + weights[1]*p2[i,1] + weights[2]*p2[i,tc+1], 
                      weights[0]*conv[i,2] + weights[1]*p3[i,2] + weights[2]*p3[i,tc+2], 
                      weights[0]*conv[i,3] + weights[1]*p4[i,3] + weights[2]*p4[i,tc+3], 
                      weights[0]*conv[i,4] + weights[1]*p5[i,4] + weights[2]*p5[i,tc+4]]

            totAdv2 = [weights[0]*conv[i,0] + weights[1]*p1[i,0] + weights[2]*p1[i,tc] + weights[3]*p1[i,2*tc], 
                       weights[0]*conv[i,1] + weights[1]*p2[i,1] + weights[2]*p2[i,tc+1] + weights[3]*p2[i,2*tc+1], 
                       weights[0]*conv[i,2] + weights[1]*p3[i,2] + weights[2]*p3[i,tc+2] + weights[3]*p3[i,2*tc+2], 
                       weights[0]*conv[i,3] + weights[1]*p4[i,3] + weights[2]*p4[i,tc+3] + weights[3]*p4[i,2*tc+3], 
                       weights[0]*conv[i,4] + weights[1]*p5[i,4] + weights[2]*p5[i,tc+4] + weights[3]*p5[i,2*tc+4]]

          
          if gt[i][0] == top[i, np.argmax(tot)]:
            correct = correct + 1
          if gt[i][0] == top[i, np.argmax(totAdv)]:
            correctAdv = correctAdv + 1
          if gt[i][0] == top[i, np.argmax(totAdv2)]:
            correctAdv2 = correctAdv2 + 1
            

correct = correct / numf * 100
correctAdv = correctAdv / numf * 100
correctAdv2 = correctAdv2 / numf * 100

        

print('---------Our Strategy----------')
print('Ours w/out adv weighted: ' + str(correct) + ' , Params: ' + str(weights[0]) + ', ' + str(weights[1]))
print('Ours w/ adv weighted: ' + str(correctAdv) + ' , Params: ' + str(weights[0]) + ', ' + str(weights[1]) + ', ' + str(weights[2]) )
print('Ours w/ adv2 weighted: ' + str(correctAdv2) + ' , Params: ' + str(weights[0]) + ', ' + str(weights[1]) + ', ' + str(weights[2]) + ', ' + str(weights[3])+ '\n')

