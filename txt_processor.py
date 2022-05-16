# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 04:32:17 2021

@author: yuwen
"""

# =============================================================================
# with open('Harry Potter 1.txt', 'r') as rf, open ('./Harry_Potter_1.txt', 'w') as wf:
#     line = rf.readline()
#     para = ''
#     while line:
#         if line != '\n':
#             para += line[0:-1]
#             para += ' '
#         else:
#             print(para)
#             wf.write(para)
#             wf.write('\n\n')
#             para=''
#         line = rf.readline()
# =============================================================================
        

#set train and test files

import numpy as np
import random
with open ('noName-Dumbledore', 'r') as d, open ('noName-Snape', 'r') as s:
    d_line = d.readline()
    dumbledore = []
    while d_line:
        dumbledore.append(d_line)
        d_line = d.readline()
    print('len(dumbledore) = ', len(dumbledore))
    
    snape = []
    s_line = s.readline()
    while s_line:
        snape.append(s_line)
        s_line = s.readline()
    print('len(snape) = ', len(snape))
    r = 20
    d_num_list = []
    for i in range(0, r):
        num = np.random.randint(0,len(dumbledore)-1)
        while num in d_num_list:
            num = np.random.randint(0,len(dumbledore)-1)
        d_num_list.append(num)
    print(d_num_list)
    
            
    s_num_list = []
    for i in range(0, r):
        num = np.random.randint(0,len(snape)-1)
        while num in d_num_list:
            num = np.random.randint(0,len(snape)-1)
        s_num_list.append(num)
    print(s_num_list)
    d.close()
    s.close()

d_train = open('d_train', 'w')
s_train = open('s_train', 'w')
for i in range(0, len(dumbledore)):
    if i not in d_num_list:
        d_train.write(dumbledore[i])
d_train.close()

for i in range(0, len(snape)):
    if i not in s_num_list:
        s_train.write(snape[i])
s_train.close()

d_test = open('d_test', 'w')
s_test = open('s_test', 'w')
for n in d_num_list:
    d_test.write(dumbledore[n])
d_test.close()
for n in s_num_list:
    s_test.write(snape[n])
s_test.close()
    
    
    
    
    
    
    
    
