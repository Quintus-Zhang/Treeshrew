#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 21:32:41 2017

@author: Quintus
"""
from numpy.random import randn

# Chapter 3 - IPython:AnInteractiveComputingand Development Environment

#==============================================================================
# IPython Basics
#==============================================================================
# launch IPython on the command line
$ ipython

# execute arbitrary Python statements by typing them in and pressing <return>
data = {i : randn() for i in range(7)}
data

# Many kinds of Python objects are formatted to be more readable, or pretty-printed, which is distinct from normal printing with print.
print(data)


#==============================================================================
# Tab Completion 
#==============================================================================
# While entering expressions in the shell, pressing <Tab> will search the namespace for any variables matching the characters you have typed so far
an_apple = 27
an_example = 42

an<Tab>

# The same goes for methods and attributes on any object
b = [1,2,3]
b.<Tab>

# The same goes for modules
import datetime
datetime.<Tab>

# The same goes for file path (even in a Python string)

?


#==============================================================================
# Introspection 
#==============================================================================
# Using a question mark (?) before or after a variable will display some general information about the object
b?
?b

# the same goes for functions
def add_numbers(a, b): 
    """
    Add two numbers together
    Returns
    -------
    the_sum : type of arguments
    """
    return a + b
add_numbers?

# Using ?? will also show the function’s source code if possible
add_numbers??

# ? has a final usage, which is for searching the IPython namespace in a manner similar to the standard UNIX or Windows command line. 
?


#==============================================================================
# The %run Command
#==============================================================================
%run Timer.py
? 


#==============================================================================
# Executing Code from the Clipboard
#==============================================================================
# %paste magic is only defined for the terminal version of IPython, not 
# for its graphical frontends (i.e. the notebook and qtconsole) because 
# they don't need it. 


#==============================================================================
# Keyboard Shortcuts
#==============================================================================
skip

#==============================================================================
# Exceptions and Tracebacks
#==============================================================================


#==============================================================================
# Magic Commands
#==============================================================================
# check the execution time of any Python statement using the %timeit magic function 
%timeit 
a = np.random.randn(100, 100)
%timeit np.dot(a, a)

# question mark works for magic commands
%reset?

# Magic functions can be used by default without the percent sign, as long as no variable is defined with the same name as the magic function in question. This feature is called automagic and can be enabled or disabled using %automagic.
%automagic

# explore all of the special commands available by typing %quickref or %magic
%quickref
%magic
%debug


#==============================================================================
# Qt-based Rich GUI Console
#==============================================================================
# Using <Ctrl-R> gives you the same partial incremental searching capability
? 


#==============================================================================
# Input and Output Variables
#==============================================================================












