
# coding: utf-8

# #### Assignment 5: Geographic Data Analysis
# 
# In this assignment, you will be using a database of geographic data provided for you in the PySal library to create two plots, a choropleth map and a proportional symbol map. In addition to these two plots, you will compute the value of Moran's I for this data.
# 
# #### Dataset
# 
# The dataset to be used in this assignment is a dataset containing Per Capita incomes for the United States' lower 48 states. In addition to the state-by-state data, the dataset contains shape files for each state that you can use
# to create the choropleth and proportional symbol maps.
# 
# #### Administrative Notes
# 
# This assignment will be graded by Coursera's grading system. In order for your answers to be correctly registered in the system, you must place the code for your answers in the cell indicated for each question. In addition, you should submit the assignment with the output of the code in the cell's display area. The display area should contain only your answer to the question with no extraneous information, or else the answer may not be picked up correctly. Each cell that is going to be graded has a set of comment lines at the beginning of the cell. These lines are extremely important and must not be modified or removed.

# In[18]:


# Graded Cell, PartID: CkcsR
# Part 1: Using the PySal Data, create a choropleth path of the United States
# that depicts the per capita income of each US state in 2009.
# Notes: The PySal and GeoPandas libraries both contain utility functions that
# may make this task easier.

import pysal as ps
import geopandas as gpd
import matplotlib.pyplot as plt

data = ps.open(ps.examples.get_path("usjoin.csv"))
data_2009 = data.by_col('2009')

us_shape = gpd.read_file(ps.examples.get_path("us48.shp"))
us_shape['2009'] = data_2009
us_shape.plot(column='2009')


# In[23]:


# Graded Cell, PartID: FqNRm
# Part 2: Again using the PySal Data, create a proportional symbol map showing 
# a dot at the centroid of each state that is scaled to the per capita income 
# of each US state in 2009.
# Notes: The demonstration notebook for this unit contains code that performs 
# a similar task and may be a useful reference for your assignment.

import pysal as ps
import geopandas as gpd
import matplotlib.pyplot as plt

data = ps.open(ps.examples.get_path("usjoin.csv"))
data_2009 = data.by_col('2009')

us_shape_copy = gpd.read_file(ps.examples.get_path("us48.shp"))
us_shape_copy['centroid'] = us_shape_copy.centroid
us_shape_copy['2009'] = data_2009

state_centroids = list(us_shape_copy['centroid'])
df = pd.DataFrame({     'y':[state_centroids[i].y for i in range(len(state_centroids))],     'x':[state_centroids[i].x for i in range(len(state_centroids))],     'us_data':list(us_shape_copy['2009'])})

base = us_shape_copy.plot(color='white', edgecolor='black')
df.plot(kind='scatter', x='x', y='y', s=df['us_data']*0.0001, ax=base)
plt.show()


# In[28]:


# Graded Cell, PartID: CtQYv
# Part 3: Using the same data, compute the value of Moran's I for the per 
# capita income of each US state in 2009 using Rook Continuity. Report the 
# value of I rounded to 4 decimal places (i.e. x.XXXX)
# Notes: Again, the PySal and GeoPandas libraries may contain useful utility
# functions.

import pysal as ps
import numpy as np

data = ps.open(ps.examples.get_path("usjoin.csv"))
y = np.array(data.by_col['2009'])
w = ps.weights.rook_from_shapefile(ps.examples.get_path('us48.shp'))
mi = ps.Moran(y, w, two_tailed=False)
print (round(mi.I, 4))

