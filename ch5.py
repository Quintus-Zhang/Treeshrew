#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:51:57 2017

@author: Quintus
"""
import numpy as np
from pandas import Series, DataFrame
import pandas as pd

#==============================================================================
# #==============================================================================
# # Introduction to pandas Data Structures 
# #==============================================================================
#==============================================================================

#==============================================================================
# Series
#==============================================================================
obj = Series([4, 7, -5, 3])
# return the values in numpy.ndarray
obj.values
# return the index in RangeIndex(start=0, stop=4, step=1)
obj.index

# an index identifying each data point
obj2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
# return the index in Index(['d', 'b', 'a', 'c'], dtype='object')
obj2.index
obj2['a']
obj2['d'] = 6
obj2[['c', 'a', 'd']]

# NumPy array operations, such as filtering with a boolean array, scalar multiplication, 
# or applying math functions, will preserve the index-value link
obj2
obj2[obj2 > 0]
obj2 * 2
np.exp(obj2)

# think a Series as a fixed-length, ordered dict
'b' in obj2
'e' in obj2

# dict to Series
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = Series(sdata)

# When only passing a dict, the index in the resulting Series will have the dict’s keys 
# in sorted order.
# since no value for 'California' was found, it appears as NaN (not a number) which is 
# considered in pandas to mark missing or NA values.
states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = Series(sdata, index=states)
obj4

# detect missing data
pd.isnull(obj4)
pd.notnull(obj4)
obj4.isnull()

# A critical Series feature for many applications is that it automatically aligns 
# differently-indexed data in arithmetic operations:
obj3 + obj4    

# Both the Series object itself and its index have a name attribute
obj4.name = 'population'
obj4.index.name = 'state'

# alter series's index in place by assignment
obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']


#==============================================================================
# DataFrame
#==============================================================================
# construct a DataFrame from a dict of equal-length lists
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'], 
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = DataFrame(data)
frame

# specify a sequence of columns 
DataFrame(data, columns=['year', 'state', 'pop'])

# pass a column that isn’t contained in data, it will appear with NA values
frame2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                   index=['one', 'two', 'three', 'four', 'five'])

# retrieve a column
frame2['state']
frame2.year

# retrieve a row by position or name
frame2.ix['three']

# Columns can be modified by assignment
# a scalar
frame2['debt'] = 16.5
frame2
# a np array
frame2['debt'] = np.arange(5.)
frame2
# a Series
val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
frame2

# Assigning a column that doesn’t exist will create a new column
frame2['eastern'] = frame2.state == 'Ohio'
frame2
del frame2['eastern']
frame2

# construct a DataFrame from a nested dict of dicts
# it will interpret the outer dict keys as the columns and the inner keys as the row indices
pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = DataFrame(pop)
frame3.T

# construct a DataFrame from a nested dict of dicts
pdata = {'Ohio': frame3['Ohio'][:-1], 'Nevada': frame3['Nevada'][:2]}
DataFrame(pdata)

# index and columns have their name attributes set
frame3.index.name = 'year'
frame3.columns.name = 'state'

# returns the data contained in the DataFrame as a 2D ndarray
frame3.values

# If the DataFrame’s columns are different dtypes, the dtype of the values array will be
# chosen to accomodate all of the columns
frame2.values


#==============================================================================
# Index Objects
#==============================================================================
obj = Series(range(3), index=['a', 'b', 'c'])
index = obj.index
# Index objects are immutable and thus can’t be modified by the user
index[1] = 'd'

index = pd.Index(np.arange(3))
obj2 = Series([1.5, -2.5, 0], index=index)
obj2.index is index

# Main Index objects in pandas
# Index        : The most general Index object, representing axis labels in a NumPy array of Python objects.
# Int64Index   : Specialized Index for integer values.
# MultiIndex   : “Hierarchical” index object representing multiple levels of indexing on a single axis. 
# DatetimeIndex: Stores nanosecond timestamps (represented using NumPy’sdatetime64dtype)
# PeriodIndex  : Specialized Index for Period data (timespans).

frame3
'Ohio' in frame3.columns
2003 in frame3.index

# Each Index has a number of methods and properties for set logic and answering other common 
# questions about the data it contains.






#==============================================================================
# #==============================================================================
# # Essential Functionality
# #==============================================================================
#==============================================================================

#==============================================================================
# Reindexing
#==============================================================================
obj = Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj 
# reindex
# rearranges the data according to the new index
# introducing missing values if any index values were not already present
# alter (row)index
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj.reindex(['a', 'b', 'c', 'd', 'e'], fill_value=0)
obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3.reindex(range(6), method='ffill')
obj3.reindex(range(6), method='bfill')

# alter columns
frame = DataFrame(np.arange(9).reshape((3, 3)), 
                  index=['a', 'c', 'd'], 
                  columns=['Ohio', 'Texas', 'California'])
states = ['Texas', 'Utah', 'California']
frame.reindex(columns=states)

# alter both index and columns in one shot
# though interpolation will only apply row-wise
frame.reindex(index=['a', 'b', 'c', 'd'], 
              method='ffill', 
              columns=states)

# reindexing can be done more succinctly by label-indexing with ix
frame = frame.ix[['a', 'b', 'c', 'd'], states]

#==============================================================================
# Dropping entries from an axis
#==============================================================================
# drop
# Series
obj = Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
new_obj = obj.drop('c')
new_obj
obj.drop(['d', 'c'])

# DataFrame
# index values can be deleted from either axis
# drop rows: default, axis = 0
# drop cols: axis = 1
data = DataFrame(np.arange(16).reshape((4, 4)),
                 index=['Ohio', 'Colorado', 'Utah', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
data.drop(['Colorado', 'Ohio'])
data.drop('two', axis=1)
data.drop(['two', 'four'], axis=1)


#==============================================================================
# Indexing, selection, and filtering
#==============================================================================
# Series
obj = Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
# Slicing with labels behaves differently than normal Python slicing in that the endpoint 
# is inclusive
obj['b':'c']

# DataFrame
data = DataFrame(np.arange(16).reshape((4, 4)),
                 index=['Ohio', 'Colorado', 'Utah', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
data['two']
data[['three', 'one']]
data[:2]
# the special indexing field ix
data.ix['Colorado', ['two', 'three']]
data.ix[2]
data.ix[:'Utah', 'two']
data.ix[2, 'one':'two']


#==============================================================================
# Arithmetic and data alignment
#==============================================================================
# Series
# When adding together objects, if any index pairs are not the same, 
# the respective index in the result will be the union of the index pairs.
s1 = Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])
s1 + s2

# DataFrame
# alignment is performed on both the rows and the columns
df1 = DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'),
                index=['Ohio', 'Texas', 'Colorado'])
df2 = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
                index=['Utah', 'Ohio', 'Texas', 'Oregon'])
df1
df2
df1 + df2

# Arithmetic methods with fill values
# fill first, add after
df1 = DataFrame(np.arange(12.).reshape((3, 4)), columns=list('abcd'))
df2 = DataFrame(np.arange(20.).reshape((4, 5)), columns=list('abcde'))
df1 + df2
df1.add(df2, fill_value=0)

# sub
# div
# mul

# Operations between DataFrame and Series
# broadcasting
# numpy
arr = np.arange(12.).reshape((3, 4))
arr
arr[0]
arr - arr[0]
# dataframe
frame = DataFrame(np.arange(12.).reshape((4, 3)), 
                  columns=list('bde'), 
                  index=['Utah', 'Ohio', 'Texas', 'Oregon'])
series = frame.ix[0]
frame - series

# the objects will be reindexed to form the union
# matching on the rows
series2 = Series(range(3), index=['b', 'e', 'f'])
frame + series2
# matching on the columns
series3 = frame['d']
frame.sub(series3, axis=0)


#==============================================================================
# Function application and mapping
#==============================================================================
frame = DataFrame(np.random.randn(4, 3), 
                  columns=list('bde'), 
                  index=['Utah', 'Ohio', 'Texas', 'Oregon'])
np.abs(frame)

f = lambda x: x.max() - x.min()
# apply by columns
frame.apply(f)
# apply by rows
frame.apply(f, axis=1)

# The reason for the name applymap is that Series has a map method for applying 
# an element-wise function
format = lambda x: '%.2f' % x
frame.applymap(format)

#==============================================================================
# Sorting and ranking
#==============================================================================
# sort_index
# to sort lexicographically by row or column index
obj = Series(range(4), index=['d', 'a', 'b', 'c'])
obj.sort_index()


frame = DataFrame(np.arange(8).reshape((2, 4)), 
                  index=['three', 'one'], 
                  columns=['d', 'a', 'b', 'c'])
frame.sort_index()
frame.sort_index(axis=1)
frame.sort_index(axis=1, ascending=False)

# order
# sort a Series by its values
obj = Series([4, 7, -3, 2])
obj.order()
# Any missing values are sorted to the end of the Series by default:
obj = Series([4, np.nan, 7, np.nan, -3, 2])
obj.order()

# sort_index
# sort df by the values in one or more columns
frame = DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
frame
frame.sort_index(by='b')
# To sort by multiple columns, pass a list of names
frame.sort_index(by=['a', 'b'])

# rank


#==============================================================================
# Axis indexes with duplicate values
#==============================================================================
obj = Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
obj.index.is_unique

# Indexing a value with multiple entries returns a Series while single entries return a scalar value: 
obj['a']


#==============================================================================
# #==============================================================================
# # Summarizing and Computing Descriptive Statistics
# #==============================================================================
#==============================================================================









