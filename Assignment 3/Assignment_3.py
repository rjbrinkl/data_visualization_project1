
# coding: utf-8

# # Assignment 3: Dino Fun World Analysis
# 
# The administrators of Dino Fun World, a local amusement park, have asked you, one of their data analysts, to perform three data analysis tasks for their park. These tasks will involve understanding, analysing, and graphing attendance data for one day of the park's operations that the park has provided for you to use. They have provided the data in the form of a database, described below.
# 
# ### Provided Database
# 
# The database provided by the park administration is formatted to be readable by any SQL database library. The course staff recommends the sqlite3 library. The database contains three tables, named 'checkin', 'attractions', and 'sequences'. The information contained in each of these tables is listed below:
# 
# `checkin`:
#     - Description: check-in data for all visitors for the day in the park. The data includes two types of check-ins, inferred and actual checkins.
#     - Fields: visitorID, timestamp, attraction, duration, type
# `attraction`:
#     - The attractions in the park by their corresponding AttractionID, Name, Region, Category, and type. Regions are from the VAST Challenge map such as Coaster Alley, Tundra Land, etc. Categories include Thrill rides, Kiddie Rides, etc. Type is broken into Outdoor Coaster, Other Ride, Carussel, etc.
#     - Fields: AttractionID, Name, Region, Category, type
# `sequences`:
#     - The check-in sequences of visitors. These sequences list the position of each visitor to the park every five minutes. If the visitor has not entered the part yet, the sequence has a value of 0 for that time interval. If the visitor is in the park, the sequence lists the attraction they have most recently checked in to until they check in to a new one or leave the park.
#     - Fields: visitorID, sequence
#     
# The database is named 'dinofunworld.db' and is located in the 'readonly' directory of the Jupyter Notebook environment. It can be accessed at 'readonly/dinofunworld.db'.
#     
# 
# ### Questions to Answer
# 
# 1: The park's administrators would like you to help them understand the different paths visitors take through the park and different rides they visit. In this mission, they have selected 5 visitors at random whose checkin sequences they would like you to analyze. For now, they would like you to construct a distance matrix for these 5 visitors. The five visitors have the ids: 165316, 1835254, 296394, 404385, and 448990.
# 
# 2: The park's administrators would like to understand the attendance dynamics at each ride (note that not all attractions are rides). They would like to see the minimum (non-zero) attendance at each ride, the average attendance over the whole day, and the maximum attendance for each ride on a Parallel Coordinate Plot.
# 
# 3: In addition to a PCP, the administrators would like to see a Scatterplot Matrix depicting the min, average, and max attendance for each ride as above. 
# 
# #### Administrative Notes
# 
# This assignment will be graded by Coursera's grading system. In order for your answers to be correctly registered in the system, you must place the code for your answers in the cell indicated for each question. In addition, you should submit the assignment with the output of the code in the cell's display area. The display area should contain only your answer to the question with no extraneous information, or else the answer may not be picked up correctly. Each cell that is going to be graded has a set of comment lines at the beginning of the cell. These lines are extremely important and must not be modified or removed.

# In[13]:


# Graded Cell, PartID: IiXwN
# Create a distance matrix suitable for use in hierarchical clustering of the
# checkin sequences of the 5 specified visitors. Your distance function should
# count the number of dissimilarities in the sequences without considering any
# other factors. The distance matrix should be reported as a dictionary of
# dictionaries (eg. {1: {2:0, 3:0, 4:0}, 2: {1:0, 3:0, ...}, ...}).

import sqlite3

db_filename = 'readonly/dinofunworld.db'
conn = sqlite3.connect(db_filename)
cur = conn.cursor()

cur.execute("SELECT visitorID, sequence FROM sequences WHERE visitorID = 165316 OR visitorID = 1835254 OR visitorID = 296394 OR visitorID = 404385 OR visitorID = 448990")
result = cur.fetchall()
#print(result)

temp = []
for i in range(len(result)):
    temp.append([result[i][0], list(result[i][1].split('-'))])
    
#print(temp)

temp2 = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
for i in range(len(temp)):
    for j in range(len(temp)):
        for k in range(len(temp[0][1])):
            if temp[i][0] != temp[j][0]:
                if temp[i][1][k] != temp[j][1][k]:
                    #something += 1
                    temp2[i][j] += 1
            else:
                break
                
#print(temp2)
dict = {}
for i in range(len(temp)):
    tempdict = {}
    for j in range(len(temp)):
        if temp[i][0] != temp[j][0]:
            tempdict[temp[j][0]] = temp2[i][j]
    dict[temp[i][0]] = tempdict

print(dict)


# In[44]:


# Graded Cell, PartID: 8S2jm
# Create and display a Parallel Coordinate Plot displaying the minimum, average, 
# and maximum attendance for each ride in the park (note that not all attractions
# are rides).

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates


db_filename = 'readonly/dinofunworld.db'
conn = sqlite3.connect(db_filename)
cur = conn.cursor()

cur.execute("SELECT attractionID from attraction WHERE category LIKE '%Ride%'")
result = cur.fetchall()

temp = []
for i in range(len(result)):
    temp.append(result[i][0])
templen = len(temp)

cur.execute("SELECT sequence from sequences")
result = cur.fetchall()

templist = []
for i in range(len(result)):
    templist.append(list(result[i][0].split('-')))

minlist = [1000] * templen
avglist = [0] * templen
maxlist = [0] * templen
lst = [0] * templen
countlst = [0] * templen
for i in range(len(templist)):
    
    for j in range(192):
        if int(templist[i][j]) in temp:
            lst[temp.index(int(templist[i][j]))] += 1
            countlst[temp.index(int(templist[i][j]))] += 1
        
    for k in range(templen):
        if lst[k] > maxlist[k]:
            maxlist[k] = lst[k]
        if lst[k] < minlist[k] and lst[k] > 0:
            minlist[k] = lst[k]
        lst[k] = 0
            
for i in range(templen):
    avglist[i] = countlst[i]/576

dict = {'ride':temp, 'min': minlist, 'avg': avglist, 'max': maxlist}
df = pd.DataFrame(dict)
parallel_coordinates(df, 'ride')
plt.gca().legend_.remove()
plt.show()


# In[45]:


# Graded Cell, PartID: KHoww
# Create and display a Scatterplot Matrix displaying the minimum, average, and 
# maximum attendance for each ride in the park.
# Note: This is a different view into the same data as the previous part. While
# you work on these plots, consider the different things that each chart says
# about the data.

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


db_filename = 'readonly/dinofunworld.db'
conn = sqlite3.connect(db_filename)
cur = conn.cursor()

cur.execute("SELECT attractionID from attraction WHERE category LIKE '%Ride%'")
result = cur.fetchall()

temp = []
for i in range(len(result)):
    temp.append(result[i][0])
templen = len(temp)

cur.execute("SELECT sequence from sequences")
result = cur.fetchall()

templist = []
for i in range(len(result)):
    templist.append(list(result[i][0].split('-')))

minlist = [1000] * templen
avglist = [0] * templen
maxlist = [0] * templen
lst = [0] * templen
countlst = [0] * templen
for i in range(len(templist)):
    
    for j in range(192):
        if int(templist[i][j]) in temp:
            lst[temp.index(int(templist[i][j]))] += 1
            countlst[temp.index(int(templist[i][j]))] += 1
        
    for k in range(templen):
        if lst[k] > maxlist[k]:
            maxlist[k] = lst[k]
        if lst[k] < minlist[k] and lst[k] > 0:
            minlist[k] = lst[k]
        lst[k] = 0
            
for i in range(templen):
    avglist[i] = countlst[i]/576

dict = {'ride':temp, 'min': minlist, 'avg': avglist, 'max': maxlist}
df = pd.DataFrame(dict)
columns=['min','avg','max']
scatter_matrix(df[columns])
plt.show()

