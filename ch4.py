#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 20:47:10 2017

@author: Quintus
"""

import numpy as np

#==============================================================================
# #============================================================================
# # The NumPy ndarray: A Multidimensional Array Object
# #============================================================================
#==============================================================================
data = np.array([[0.9526, -0.246 , -0.8856],[0.5639, 0.2379, 0.9104]])

# vetorised operations
data * 10
data + data

# all of the elements must be the same type

# return a tuple indicating the size of each dimension - (#rows, #columns)
data.shape

# return the number of rows
data.ndim

# return a dtype, an object describing the data type of the array
data.dtype


#==============================================================================
# Creating ndarrays
#==============================================================================
# np.array()
# accepts any sequence-like object (including other arrays)
# produces a new NumPy array containing the passed data
data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
arr1
data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)
arr2

np.zeros(10)
np.zeros((3, 6))
np.ones((3, 6))
np.eye(3)
np.identity(3)
np.empty((2,3,2))
np.arange(15)

# Convert input to ndarray, but do not copy if the input is already an ndarray
a = np.array([1,2,3])
b = np.asarray(a)
a[0] = 2
a
b

# ones_like takes another array and produces a ones array of the same shape and dtype
np.ones_like(a)
np.zeros_like(a)
np.empty_like(a)


#==============================================================================
# Data Types for ndarrays
#==============================================================================
arr1 = np.array([1, 2, 3], dtype=np.float64)
arr1.dtype
arr2 = np.array([1, 2, 3], dtype=np.int32)
arr2.dtype

# astype()
# convert or cast an array from one dtype to another 
arr = np.array([1, 2, 3, 4, 5])
arr.dtype
float_arr = arr.astype(np.float64)
float_arr.dtype

# cast some floating point numbers to be of integer dtype
# the decimal part will be truncated

# string representing numbers to numeric form
numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)
numeric_strings.astype(float)

# convert an array's dtype to another array's dtype
int_array = np.arange(10)
calibers = np.array([.22, .270, .357, .380, .44, .50], dtype=np.float64)
int_array.astype(calibers.dtype)

# shorthand type code strings you can also use to refer to a dtype
empty_uint32 = np.empty(8, dtype='u4')


#==============================================================================
# Operations between Arrays and Scalars
#==============================================================================
# Any arithmetic operations between equal-size arrays applies the operation elementwise
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
arr
arr * arr
arr - arr

# Arithmetic operations with scalars are as you would expect, propagating the value to each element
1 / arr
arr ** 0.5

# broadcasting: operations between differently sized arrays

#==============================================================================
# Basic Indexing and Slicing
#==============================================================================
# An important first distinction from lists is that array slices are views on the original array
arr = np.arange(10)
arr
arr_slice = arr[5:8]
arr_slice[1] = 12345
arr

# copy instead of view
arr[5:8].copy()

#==============================================================================
# Boolean Indexing
#==============================================================================
# select everything but 'Bob', you can either use != or negate the condition using -
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
data[-(names == 'Bob')]

# Selecting data from an array by boolean indexing always creates a copy of the data, 
# even if the returned array is unchanged

# The Python keywords and and or do not work with boolean arrays


#==============================================================================
# Fancy Indexing
#==============================================================================
# To select out a subset of the rows in a particular order, 
# you can simply pass a list or ndarray of integers specifying the desired order
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
arr
arr[[4, 3, 0, 6]]

arr = np.arange(32).reshape((8, 4))
arr
arr[[1, 5, 7, 2], [0, 3, 1, 2]]


arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]
arr[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])]


#==============================================================================
# Transposing Arrays and Swapping Axes
#==============================================================================
arr = np.arange(15).reshape((3, 5))
arr
arr.T

np.dot(arr.T, arr)

arr.swapaxes(1, 2)





#==============================================================================
# #============================================================================
# # Universal Functions: Fast Element-wise Array Functions
# #============================================================================
#==============================================================================
x = np.randn(8)
x
y = np.randn(8)
y
np.maximum(x, y) 

# randn
# returns the fractional and integral parts of a floating point array:
arr = np.random.randn(7) * 5
a, b = np.modf(arr)





#==============================================================================
# #============================================================================
# # Data Processing Using Arrays
# #============================================================================
#==============================================================================
# meshgrid
# takes two 1D arrays 
# produces two 2D matrices corresponding to all pairs of (x, y) in the two arrays
points = np.arange(-5, 5, 1)
xs, ys = np.meshgrid(points, points)
import matplotlib.pyplot as plt
plt.scatter(xs, ys)

#==============================================================================
# Expressing Conditional Logic as Array Operations
#==============================================================================
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
result = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)]

# np.where(cond, x, y) 
# x and y are equal size array  
result = np.where(cond, xarr, yarr)

# x and y are scalars
arr = np.random.randn(4, 4)
arr
np.where(arr > 0, 2, -2)
np.where(arr > 0, 2, arr)

# advanced usage of np.where
np.where(cond1 & cond2, 0, np.where(cond1, 1, np.where(cond2, 2, 3)))


#==============================================================================
# Mathematical and Statistical Methods
#==============================================================================
arr = np.random.randn(5, 4)

# computes the statistic over the given axis
# 0: compute according to columns
# 1: compute according to rows
arr.mean(axis = 1)
arr.sum(0)

arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr.cumsum(0)
arr.cumprod(1)


#==============================================================================
# Methods for Boolean Arrays
#==============================================================================
# Number of positive values
arr = randn(100)
(arr > 0).sum() 

# any & all
bools = np.array([False, False, True, False])
bools.any()
bools.all()


#==============================================================================
# Sorting
#==============================================================================
arr = np.random.randn(8)
arr.sort()

# modify the array in place
arr = np.random.randn(5, 3)
arr.sort(1) 

# returns a sorted copy of an array
arr1 = np.sort(arr)


#==============================================================================
# Unique and Other Set Logic
#==============================================================================
# np.unique
# returns the sorted unique values in an array
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)

# np.in1d
# tests membership of the values in one array in another
values = np.array([6, 0, 0, 3, 2, 5, 6])
np.in1d(values, [2, 3, 6])

# other array set operations
np.intersect1d(x, y)
np.union1d(x, y)
np.setdiff1d(x, y)
np.setxor1d(x, y)








#==============================================================================
# #============================================================================
# # File Input and Output with Arrays
# #============================================================================
#============================================================================== 

#==============================================================================
# Storing Arrays on Disk in Binary Format
#==============================================================================
# arrays are saved by default in an uncompressed raw binary format with file extension .npy
arr = np.arange(10)
np.save('some_array', arr)
np.load('some_array.npy')

# np.savez
np.savez('array_archive.npz', a=arr, b=arr)
arch = np.load('array_archive.npz')


#==============================================================================
# Saving and Loading Text Files
#==============================================================================
# np.loadtxt
arr = np.loadtxt('array_ex.txt', delimiter=',')

# np.savetxt
# writing an array to a delimited text file








#==============================================================================
# #============================================================================
# # Linear Algebra
# #============================================================================
#==============================================================================
x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
x.dot(y)
np.dot(x, y)

# numpy.linalg
# has a standard set of matrix decompositions and things like inverse and determinant
from numpy.linalg import inv, qr
X = np.random.randn(5, 5)
mat = X.T.dot(X)
inv(mat)
q, r = qr(mat)

# other Commonly-used numpy.linalg functions
trace
det
eig
inv
pinv
svd
solve
lstsq







#==============================================================================
# #============================================================================
# # Random Number Generation
# #============================================================================
#==============================================================================
samples = np.random.normal(size=(4, 4))
#seed
#permutation
#shuffle
#rand
#randint: Draw random integers from a given low-to-high range
#randn : Draw samples from a normal distribution with mean 0 and standard deviation 1 (MATLAB-like interface)
#binomial
#beta
#chisquare
#gamma
#uniform







#==============================================================================
# #============================================================================
# # Example: Random Walks
# #============================================================================
#==============================================================================
nsteps = 1000
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()
plt.plot(walk[:100])

# first crossing time
np.abs(walk) >= 10
(np.abs(walk) >= 10).argmax()


#==============================================================================
# Simulating Many Random Walks at Once
#==============================================================================
nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size=(nwalks, nsteps))
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1)
walks

# compute the minimum crossing time to 30 or -30.
hits30 = (np.abs(walks) >= 30).any(1)
hits30
hits30.sum()
crossing_times = (np.abs(walks[hits30]) >= 30).argmax(1)
crossing_times.mean()





