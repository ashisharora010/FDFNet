import sys
import math
import copy
import random
import numpy as np
import os

num_classes=503
dim1=503
dim2=128

folder='token_'+str(dim1)+'x'+str(dim2)+'/'

if not os.path.exists(folder):
    os.makedirs(folder)

os.chdir(folder)

l=[]
i=0
l6=[]
same=0
while i<num_classes:
     l1=[]
     for x in range(0,dim1):
             file=open(str(i),"w")
             l=[random.randrange(-100000,100000) for _ in range(0, dim2)]
             l1.append(l)
     '''
     for j in range(0,len(l6)):
             if np.array(l6[j]).all()==np.array(l1).all():
                same=1
     if same==1:
           same=0
           continue   
     '''
     l1, r = np.linalg.qr(np.array(l1))   
     for xx in range(0,dim1):
            for yy in range(0,dim2):
                file.write(str(l1[xx][yy]))
                if yy !=dim2-1:
                    file.write(" ")
            if xx !=dim1-1:
                file.write("\n") 
     i=i+1
     print(i)
     l6.append(l1)






