# coding: utf-8
import numpy as np
v = np.arange(12).reshape(3, 4)
#('v: ', array([[ 0,  1,  2,  3],
#       [ 4,  5,  6,  7],
#       [ 8,  9, 10, 11]]))
print("v: ",v)
print("v[:0][1]: ", v[:,0]) # 1列目 array([0, 4, 8])
print("v[:0][0]: ", v[:,1]) # ２列目 array([1, 5, 9]

print("v[:0][1]: ", v[:,0][0])# 1*1 0
print("v[:0][1]: ", v[:,0][1])# 2*1 4

print(v[0,1]) # 1*2 1
