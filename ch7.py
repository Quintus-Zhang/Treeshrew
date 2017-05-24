#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 17:36:31 2017

@author: Quintus
"""
import numpy as np
from pandas import Series, DataFrame
import pandas as pd

#==============================================================================
# #==============================================================================
# # Combining and Merging Data Sets 
# #==============================================================================
#==============================================================================
# Data contained in pandas objects can be combined together in a number of built-in ways
# pd.merge 
# pd.concat 
# pd.combine_first

#==============================================================================
# Database-style DataFrame Merges
#==============================================================================
# pd.merge(df1, df2, on= , left_on=, right_on=, how= , )
# on:       Column names to join on / merge on wchich overlapping column
# left_on:  use column in left as its join key
# right_on: use column in right as its join key
# how:      inner(by default), outer, left, right
# suffixes: Tuple of string values to append to column names in case of overlap
# sort:     Sort merged data lexicographically by join keys; True by default.
# left_index: Use index in left as its join key (or keys, if a MultiIndex)
# right_index: Use index in right as its join key (or keys, if a MultiIndex)

    
# many-to-one merge situation
df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
df2 = DataFrame({'key': ['a', 'b', 'd'], 'data2': range(3)})
df1
df2
# If not specified which column to join on, merge uses the overlapping column names as the keys
pd.merge(df1, df2)
pd.merge(df1, df2, on='key')
# If the column names are different in each object, you can specify them separately:
df3 = DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
df4 = DataFrame({'rkey': ['a', 'b', 'd'], 'data2': range(3)})
df3
df4
pd.merge(df3, df4, left_on='lkey', right_on='rkey')
pd.merge(df1, df2, how='outer')

# Many-to-many merges
df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'], 'data1': range(6)})
df2 = DataFrame({'key': ['a', 'b', 'a', 'b', 'd'], 'data2': range(5)})
pd.merge(df1, df2, on='key', how='left')
pd.merge(df1, df2, how='inner')
pd.merge(df1, df2, how='right')

# To merge with multiple keys, pass a list of column names:
left = DataFrame({'key1': ['foo', 'foo', 'bar'], 'key2': ['one', 'two', 'one'],'lval': [1, 2, 3]})
right = DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'], 'key2': ['one', 'one', 'one', 'two'], 'rval': [4, 5, 6, 7]})
pd.merge(left, right, on=['key1', 'key2'], how='outer')
# treatment of overlapping column names
pd.merge(left, right, on='key1')
pd.merge(left, right, on='key1', suffixes=('_left', '_right'))
  key1 key2_left  lval key2_right  rval
0  foo       one     1        one     4
1  foo       one     1        one     5
2  foo       two     2        one     4
3  foo       two     2        one     5
4  bar       one     3        one     6
5  bar       one     3        two     7

#==============================================================================
# Merging on Index
#==============================================================================
# index & column
left1 = DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'], 'value': range(6)})
right1 = DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])
pd.merge(left1, right1, left_on='key', right_index=True)

# hierarchically-indexed data
# In this case, you have to indicate multiple columns to merge on as a list 
lefth = DataFrame({'key1': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'], 
                   'key2': [2000, 2001, 2002, 2001, 2002], 
                   'data': np.arange(5.)})
righth = DataFrame(np.arange(12).reshape((6, 2)), 
                   index=[['Nevada', 'Nevada', 'Ohio', 'Ohio', 'Ohio', 'Ohio'], 
                          [2001, 2000, 2000, 2000, 2001, 2002]], 
                   columns=['event1', 'event2'])
pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True)
pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True, how='outer')

# Using the indexes of both sides of the merge is also not an issue
left2 = DataFrame([[1., 2.], [3., 4.], [5., 6.]], index=['a', 'c', 'e'], columns=['Ohio', 'Nevada'])
right2 = DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]], index=['b', 'c', 'd', 'e'], columns=['Missouri', 'Alabama'])
pd.merge(left2, right2, how='outer', left_index=True, right_index=True)

# DataFrame has a more convenient join instance for merging by index.
left2.join(right2, how='outer')
# performs a left join on the join keys
# supports joining the index of the passed DataFrame on one of the columns of the calling DataFrame
left1.join(right1, on='key')

#==============================================================================
# Concatenating Along an Axis
#==============================================================================
# pd.concat([], axis=0, )
# axis: 0(by default) or 1
# join: 'inner', 'outer'(by default), 
# join_axes: specify the axes to be used on the other axes with join_axes
# keys: you wanted to create a hierarchical index on the concatenation axis
# names: Names for created hierarchical levels ifkeysand / orlevelspassed
# ignore_index: Do not preserve indexes along concatenation axis, instead producing a new range(total_length) index

# Suppose we have three Series with no index overlap
s1 = Series([0, 1], index=['a', 'b'])
s1
s2 = Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = Series([5, 6], index=['f', 'g'])
pd.concat([s1, s2, s3])

# overlapping index exists
s4 = pd.concat([s1 * 5, s3])
s4
pd.concat([s1, s4], axis=1)
pd.concat([s1, s4], axis=1, join='inner')

# specify the axes to be used on the other axes with join_axes
pd.concat([s1, s4], axis=1, join_axes=[['a', 'c', 'b', 'e']])

# you wanted to create a hierarchical index on the concatenation axis
# axis = 0
result = pd.concat([s1, s1, s3], keys=['one', 'two', 'three'])
result
# axis = 1
pd.concat([s1, s2, s3], axis=1, keys=['one', 'two', 'three'])
# same on dataframe
df1 = DataFrame(np.arange(6).reshape(3, 2), index=['a', 'b', 'c'], columns=['one', 'two'])
df2 = DataFrame(5 + np.arange(4).reshape(2, 2), index=['a', 'c'], columns=['three', 'four'])
pd.concat([df1, df2], axis=1, keys=['level1', 'level2'])
# If you pass a dict of objects instead of a list, the dict’s keys will be used for the keys option
pd.concat({'level1': df1, 'level2': df2}, axis=1)
# names
pd.concat([df1, df2], axis=1, keys=['level1', 'level2'], names=['upper', 'lower'])

# the row index is not meaningful in the context of the analysis
df1 = DataFrame(np.random.randn(3, 4), columns=['a', 'b', 'c', 'd'])
df2 = DataFrame(np.random.randn(2, 3), columns=['b', 'd', 'a'])
pd.concat([df1, df2], ignore_index=True)


#==============================================================================
# Combining Data with Overlap
#==============================================================================
a = Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan], index=['f', 'e', 'd', 'c', 'b', 'a'])
a
b = Series(np.arange(len(a), dtype=np.float64), index=['f', 'e', 'd', 'c', 'b', 'a'])
b
b[-1] = np.nan
np.where(pd.isnull(a), b, a)

# combine_first ? 






#==============================================================================
# #==============================================================================
# # Reshaping and Pivoting
# #==============================================================================
#==============================================================================

#==============================================================================
# Reshaping with Hierarchical Indexing
#==============================================================================
data = DataFrame(np.arange(6).reshape((2, 3)), index=pd.Index(['Ohio', 'Colorado'], name='state'), columns=pd.Index(['one', 'two', 'three'], name='number'))
data

# stack
# pivots the columns into the rows
result = data.stack()

# unstack
# rearrange the data back into a DataFrame
result.unstack()
result.unstack(0)
result.unstack('state')

#==============================================================================
# Pivoting “long” to “wide” Format
#==============================================================================






#==============================================================================
# #==============================================================================
# # Data Transformation
# #==============================================================================
#==============================================================================

#==============================================================================
# Removing Duplicates
#==============================================================================
data = DataFrame({'k1': ['one'] * 3 + ['two'] * 4, 'k2': [1, 1, 2, 3, 3, 4, 4]})
data

# df.duplicated()
# returns a boolean Series indicating whether each row is a duplicate or not
# by default consider all of the columns
data.duplicated()

# df.drop_duplicates()
# returns a DataFrame where the duplicated array is True
# by default consider all of the columns
data.drop_duplicates()

# specify any subset of them to detect duplicates
# by default keep the first observed value combination
data['v1'] = range(7)
data.drop_duplicates(['k1'])

# Passing take_last=True will return the last one
data.drop_duplicates(['k1', 'k2'], take_last=True)


#==============================================================================
# Transforming Data Using a Function or Mapping
#==============================================================================
data = DataFrame({'food': ['bacon', 'pulled pork', 'bacon', 
                           'Pastrami', 'corned beef', 'Bacon', 
                           'pastrami', 'honey ham','nova lox'],
                  'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
data

meat_to_animal = { 'bacon': 'pig', 'pulled pork': 'pig', 'pastrami': 'cow', 'corned beef': 'cow', 'honey ham': 'pig', 'nova lox': 'salmon'}
meat_to_animal

# map
# accepts a function or dict-like object containing a mapping
data['animal'] = data['food'].map(str.lower).map(meat_to_animal)
# or
data['food'].map(lambda x: meat_to_animal[x.lower()])


#==============================================================================
# Replacing Values
#==============================================================================
# df.replace(a, b)
# a: the element to be replaced
# b: the element to replace that in dataframe

data = Series([1., -999., 2., -999., -1000., 3.])
data
# To replace these with NA values that pandas understands
data.replace(-999, np.nan)
# If you want to replace multiple values at once, you instead pass a list then the substitute value
data.replace([-999, -1000], np.nan)
# To use a different replacement for each value, pass a list of substitutes:
data.replace([-999, -1000], [np.nan, 0])
# The argument passed can also be a dict
data.replace({-999: np.nan, -1000: 0})


#==============================================================================
# Renaming Axis Indexes
#==============================================================================
data = DataFrame(np.arange(12).reshape((3, 4)),
                 index=['Ohio', 'Colorado', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
data
data.index.map(str.upper)
data.index = data.index.map(str.upper)

# rename
# create a transformed version of a data set without modifying the original
newData = data.rename(index=str.title, columns=str.upper)

# rename can be used in conjunction with a dict-like object providing new values
# for a subset of the axis labels
data.rename(index={'OHIO': 'INDIANA'},
            columns={'three': 'peekaboo'})

# inplace = True
_ = data.rename(index={'OHIO': 'INDIANA'}, inplace=True)

data

#==============================================================================
# Discretization and Binning
#==============================================================================
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
# pd.cut(data, bins, right=, left=, labels= )
# data
# bins: explicit bin edges / number of bins
# right: True(right included) / False(right not included)
# labels: pass your own bin names by passing a list or array to the labels option

# return:
# The object pandas returns is a special Categorical object

# Categorical object's methods:
# .labels
# .levels

cats = pd.cut(ages, bins)
cats
# a labeling for the ages data in the labels attribute
cats.labels
# it contains a levels array indicating the distinct category names
cats.levels
pd.value_counts(cats)

pd.cut(ages, [18, 26, 36, 61, 100], right=False)

group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
cats = pd.cut(ages, bins, labels=group_names)
cats.levels

data = np.random.rand(20)
pd.cut(data, 4, precision=2)

# pd.qcut(data, #bins)
# #bins: number of bins / pass your own quantiles
# Since qcut uses sample quantiles instead, by definition you will obtain roughly 
# equal-size bins
data = np.random.randn(1000)
cats = pd.qcut(data, 4) # Cut into quartiles
pd.value_counts(cats)

pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.])


#==============================================================================
# Detecting and Filtering Outliers
#==============================================================================
np.random.seed(12345)
data = DataFrame(np.random.randn(1000, 4))
data.describe()

col = data[3]
col[np.abs(col) > 3]

# To select all rows having a value exceeding 3 or -3, you can use the any method 
# on a boolean DataFrame 
data[(np.abs(data) > 3).any(1)]
data[np.abs(data) > 3] = np.sign(data) * 3


#==============================================================================
# Permutation and Random Sampling
#==============================================================================
# Calling permutation with the length of the axis you want to permute produces 
# an array of integers indicating the new ordering
df = DataFrame(np.arange(5 * 4).reshape(5, 4))
df
sampler = np.random.permutation(5)
sampler
# That array can then be used in ix-based indexing or the take function
df.take(sampler)
df.ix[sampler]

# To select a random subset without replacement, one way is to slice off the 
# first k elements of the array returned by permutation, where k is the desired subset size. 
df.take(np.random.permutation(len(df))[:3])

# To generate a sample with replacement, the fastest way is to use np.random.randint to 
# draw random integers
bag = np.array([5, 7, -1, 6, 4])
bag
sampler = np.random.randint(0, len(bag), size=10)
draws = bag.take(sampler)
draws


#==============================================================================
# Computing Indicator/Dummy Variables
#==============================================================================
# pd.get_dummies(data, prefix=, )
# prefix: 
# If a column in a DataFrame has k distinct values, you would derive a matrix or 
# DataFrame containing k columns containing all 1’s and 0’s.
df = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'], 'data1': range(6)})
df
pd.get_dummies(df['key'])

# add a prefix to the columns in the indicator DataFrame,
dummies = pd.get_dummies(df['key'], prefix='key')
df_with_dummy = df[['data1']].join(dummies)
df_with_dummy





#==============================================================================
# #==============================================================================
# # String Mainipulation
# #==============================================================================
#==============================================================================

#==============================================================================
# String Object Methods
#==============================================================================
# str.split()
val = 'a,b, guido'
val.split(',')

# str.strip()
pieces = [x.strip() for x in val.split(',')]
pieces

# These substrings could be concatenated together with a two-colon delimiter using addition
first, second, third = pieces
first + '::' + second + '::' + third
# more Pythonic way
'::'.join(pieces)

# Using Python’s in keyword is the best way to detect a substring
'guido' in val
# index raises an exception if the string isn’t found
val.index(',')
# find return -1 if the string isn’t found
val.find(':')

# count
val.count(',')

# replace
val.replace(',', '::')


#==============================================================================
# Regular expressions
#==============================================================================
import re
text = "foo bar\t baz \tqux"
# split a string with a variable number of whitespace characters
# the regular expression is first compiled, then its split method is called on 
# the passed text
re.split('\s+', text)

regex = re.compile('\s+')
regex.split(text)

# findall()
# get a list of all patterns matching the regex
regex.findall(text)

# Creating a regex object with re.compile is highly recommended if you intend 
# to apply the same expression to many strings; doing so will save CPU cycles.

# findall : returns all matches in a string
# search  : returns only the first match
# match   : only matches at the beginning of the string

text = """Dave dave@google.com Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""
pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'
# re.IGNORECASE makes the regex case-insensitive 
regex = re.compile(pattern, flags=re.IGNORECASE)

regex.findall(text)
['dave@google.com', 'steve@gmail.com', 'rob@gmail.com', 'ryan@yahoo.com']

m = regex.search(text)
m
 <_sre.SRE_Match object; span=(5, 20), match='dave@google.com'>
text[m.start():m.end()]
'dave@google.com'

regex.match(text)
None

# sub
# return a new string with occurrences of the pattern replaced by the a new string:
print(regex.sub('REDACTED', text))

# 
pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
regex = re.compile(pattern, flags=re.IGNORECASE)
m = regex.match('wesm@bright.net')
m.groups()

regex.findall(text)


#==============================================================================
# Vectorized string functions in pandas
#==============================================================================
data = {'Dave': 'dave@google.com', 'Steve': 'steve@gmail.com', 'Rob': 'rob@gmail.com', 'Wes': np.nan}
data
data = Series(data)
data
data.isnull()

data.str.contains('gmail')
data.str.findall(pattern, flags=re.IGNORECASE)
matches = data.str.match(pattern, flags=re.IGNORECASE)


#==============================================================================
# #==============================================================================
# # Example: USDA Database
# #==============================================================================
#==============================================================================







