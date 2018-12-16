import numpy as np
from math import sqrt
import os
import time
from datetime import timedelta
import sys
import math
import copy
import random
import numpy as np
import os

def get_token(subject):
    token_file = open('token_'+str(dim1)+'x'+str(dim2)+'/'+str(subject),'r')
    token=np.loadtxt(token_file)
    return token

# Generate Binary Templates and Store File in Database

def generate_bit_file(feature):

  feature2=[]
  for i in range(0,len(feature)):
    cls=i
    if len(feature)!=dim1:
       cls=int(test_labels[i])
    print(cls)
    f1=feature[i]
    token=get_token(cls)
    f2=np.dot(f1,token)
    feature2.append(f2)
  feature2=np.asarray(feature2)
  print('Feature_reduced : ',feature2.shape)

  # Quantisation into binary values using threshold
  threshold=0
  binary_feature=np.zeros(feature2.shape)
  binary_feature=(feature2>threshold).astype(float)
  print('Binary_templates : ',binary_feature.shape)

  return binary_feature#,feature2

####################################################################

def subject_hamming_distn(test_vector):
    distn=[]
    for i in range(0,len(templates)):
        dist1 = np.square(templates[i]-test_vector).sum()
        distn.append(dist1)
    distn=np.asarray(distn)
    min_index=np.argmin(distn)
    return min_index

def subject_jaccard_distn(test_vector):
    distn=[]
    for i in range(0,len(templates)):
        sum1=np.minimum(templates[i],test_vector).sum()
        sum2=np.maximum(templates[i],test_vector).sum()
        J=sum1/sum2
        distn.append(1-J)
    distn=np.asarray(distn)
    min_index=np.argmin(distn)
    min_distn=np.min(distn)
    return min_index

def subject_score(test_vector):
    score=[]
    for i in range(0,len(templates)):
        dist1 = sqrt(np.square(templates[i]-test_vector).sum())
        norm1 = sqrt(np.square(templates[i]).sum())
        norm2 = sqrt(np.square(test_vector).sum())
        s=1-dist1/(norm1+norm2)
        score.append(s)
    score=np.asarray(score)
    max_score=np.max(score)
    max_index=np.argmax(score)
    return max_index, max_score

def test_accuracy(test_feature,test_labels):
   count1=0
   count2=0
   count3=0
   for i in range(len(test_feature)):
      true_cls=int(test_labels[i])
      test_bit=test_feature[i]

      # Prediction using Hamming distance
      pred_cls=subject_hamming_distn(test_bit)
      if pred_cls==true_cls:
         count1 += 1

      # Prediction using Jaccard distance
      pred_cls=subject_jaccard_distn(test_bit)
      if pred_cls==true_cls:
         count2 += 1

      # Prediction using Score
      pred_cls,score=subject_score(test_bit)
      if pred_cls==true_cls:
         count3 += 1

   # Print accuracy
   acc=100*count1/len(test_labels)
   print('Accuracy - Hamming Distance : ',acc)

   acc=100*count2/len(test_labels)
   print('Accuracy - Jaccard Distance : ',acc)

   acc=100*count3/len(test_labels)
   print('Accuracy - Score : ',acc)

####################################################################

# Create User-specific tokens
dim1=503
dim2=128


train_file='major_train_sbl5x3.txt'
test_file='major_test_sbl5x3.txt'
labels_file='major_test_labels.txt'
threshold=0

# Load feature vector of dimension 503 for 503 classes

feature=np.loadtxt(train_file)
print('Feature_extracted :', feature.shape)

print('Generating Binary Templates...') 
templates=generate_bit_file(feature)
np.savetxt('nail'+str(dim2)+'.txt',templates)

# Load feature vectors for testing

test_feature=np.loadtxt(test_file)
test_labels=np.loadtxt(labels_file)

binary_feature=generate_bit_file(test_feature)
np.savetxt('nail'+str(dim2)+'_test.txt',binary_feature)


print('Testing...')
test_accuracy(binary_feature,test_labels)

