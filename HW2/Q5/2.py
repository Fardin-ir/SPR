import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.fftpack import dct as dct1
from PIL import Image
import collections
import math

def blockshaped(arr, b0, b1):
    return  np.asarray(arr).reshape(85,75,8,8)

def dct(blocks):
    mat = blocks
    for x in range(0, mat.shape[0]):
        for y in range(0, mat.shape[1]):
            mat[x,y] = dct1(dct1(mat[x,y].T, norm='ortho').T, norm='ortho')
    return mat

def zig_zag(matrix):
    mat = matrix
    out = np.zeros((85,75,64))
    for x in range(0, mat.shape[0]):
        for y in range(0, mat.shape[1]):
            t = np.concatenate([np.diagonal(mat[x,y][::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-mat[x,y].shape[0], mat[x,y].shape[0])])
            out[x,y]=t
    return out

def sec_max(matrix):
    mat = matrix
    out = np.empty((85,75,1))
    for x in range(0, mat.shape[0]):
        for y in range(0, mat.shape[1]):
            out[x,y] = np.argsort(mat[x,y] ,kind='stable')[1]
    return out

def rgb2gray(image):
    grayValue = 0.07 * image[:,:,2] + 0.72 * image[:,:,1] + 0.21 * image[:,:,0]
    gray_img = grayValue.astype(np.uint8)
    return gray_img

def mode1(x):
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    return values[m], counts[m]

t_blocks= blockshaped(np.asarray(Image.open('leo1.png').convert('L')),8,8)
t_blocks = t_blocks.copy()
t_dct_blocks = dct(t_blocks)
t_blocks_zigzag = zig_zag(t_dct_blocks)
t_second_max = sec_max(t_blocks_zigzag)

messi = []
field = []
m_blocks=blockshaped(np.asarray(Image.open('leo1_mask.png').convert('L')),8,8)
m_blocks_zigzag = zig_zag(m_blocks)


for x in range(0, t_blocks_zigzag.shape[0]):
    for y in range(0, t_blocks_zigzag.shape[1]):
        if mode1(m_blocks_zigzag[x,y])[0]!=0:
            messi.append(int(t_second_max[x,y]))
        else:
            field.append(int(t_second_max[x,y]))



#print(messi)
plt.subplot(1,2,1)
plt.hist(messi)
plt.title('X|messi')
plt.subplot(1,2,2)
plt.hist(field)
plt.title('X|field')
plt.show()

prior_messi = len(messi)/6375
prior_field = len(field)/6375
print(prior_messi)
print(prior_field)
t2_blocks= blockshaped(np.asarray(Image.open('leo2.png').convert('L')),8,8)
t2_blocks = t2_blocks.copy()
t2_blocks_out = t2_blocks.copy()
t2_blocks_mask =t2_blocks.copy()
t2_dct_blocks = dct(t2_blocks)
t2_blocks_zigzag = zig_zag(t2_dct_blocks)

t2_second_max = sec_max(t2_blocks_zigzag)


messi_counter = collections.Counter(messi)
field_counter = collections.Counter(field)




for x in range(0, t2_second_max.shape[0]):
    for y in range(0, t2_second_max.shape[1]):
        temp = t2_second_max[x,y][0]
        if  prior_messi*messi_counter[temp] > prior_field*field_counter[temp]:
            t2_blocks_mask[x,y]=np.ones((8,8))*255
        else:
            t2_blocks_out[x,y]=np.zeros((8,8))
            t2_blocks_mask[x,y]=np.zeros((8,8))



plt.imshow(t2_blocks_out.reshape(680,600),cmap='gray')
plt.show()

plt.imshow(t2_blocks_mask.reshape(680,600),cmap='gray')
plt.show()

m_t2_blocks= blockshaped(np.asarray(Image.open('leo2_mask.png').convert('L')),8,8)
m_t2_blocks = m_t2_blocks.copy()
error = 0
for x in range(0,t2_blocks_mask.shape[0]):
    for y in range(0, t2_blocks_mask.shape[1]):
        if mode1(m_t2_blocks[x,y])[0] == mode1(t2_blocks_mask[x,y])[0]:
            pass
        else:
            error +=1
error /= (85*75)
print('error:',error)
