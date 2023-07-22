
# coding: utf-8

# # Assignment 1: Dino Fun World
# 
# You, in your role as a burgeoning data explorer and visualizer, have been asked by the administrators of a small amusement park in your hometown to answer a couple questions about their park operations. In order to perform the requested analysis, they have provided you with a database containing information about one day of the park's operations.
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
# ### Questions to Answer
# 
# The administrators would like you to answer four relatively simple questions about the park activities on the day in question. These questions all deal with park operations and can be answered using the data provided.
# 
# Question 1: What is the most popular attraction to visit in the park?
# Question 2: What ride (note that not all attractions are rides) has the longest visit time?
# Question 3: Which Fast Food offering has the fewest visitors?
# Question 4: Compute the Skyline of number of visits and visit time for the park's ride and report the rides that appear in the Skyline.
# 
# #### Administrative Notes
# 
# This assignment will be graded by Coursera's grading system. In order for your answers to be correctly registered in the system, you must place the code for your answers in the cell indicated for each question. In addition, you should submit the assignment with the output of the code in the cell's display area. The display area should contain only your answer to the question with no extraneous information, or else the answer may not be picked up correctly. Each cell that is going to be graded has a set of comment lines at the beginning of the cell. These lines are extremely important and must not be modified or removed.

# In[32]:


# Graded Cell, PartID: NDnou
# Question 1: What is the most popular attraction to visit in the park?
# Notes: Your output should be the name of the attraction.

import sqlite3

db_filename = "readonly/dinofunworld.db"
conn = sqlite3.connect(db_filename)
cur = conn.cursor()

cur.execute("SELECT Name, MAX(attraction_counts) FROM (SELECT attraction, Name, COUNT(*) as attraction_counts FROM checkin INNER JOIN attraction ON checkin.attraction = attraction.AttractionID GROUP BY attraction)")
counts = cur.fetchall()
print(counts[0][0])


# In[42]:


# Graded Cell, PartID: FXGHp
# Question 2: What ride (note that not all attractions are rides) has the longest average visit time?
# Notes: Your output should be the name of the ride.

import sqlite3

db_filename = "readonly/dinofunworld.db"
conn = sqlite3.connect(db_filename)
cur = conn.cursor()

cur.execute("SELECT AVG(duration), attraction, category, Name FROM checkin INNER JOIN attraction ON checkin.attraction = attraction.AttractionID WHERE category LIKE '%Ride%' GROUP BY attraction ORDER BY AVG(duration) DESC")
result = cur.fetchall()
print(result[0][3])


# In[51]:


# Graded Cell, PartID: KALua
# Question 3: Which Fast Food offering in the park has the fewest visitors?
# Notes: Your output should be the name of the fast food offering.

import sqlite3

db_filename = "readonly/dinofunworld.db"
conn = sqlite3.connect(db_filename)
cur = conn.cursor()

cur.execute("SELECT Name, Category, COUNT(*) FROM attraction INNER JOIN checkin ON checkin.attraction = attraction.AttractionID WHERE Category LIKE '%Food%' GROUP BY Name ORDER BY COUNT(*) ASC")
result = cur.fetchall()
print(result[0][0])


# In[117]:


# Graded Cell, PartID: B0LUP
# Question 4: Compute the Skyline of number of visits and visit time for the park's ride and 
#  report the rides that appear in the Skyline. 
# Notes: Remember that in this case, higher visits is better and lower visit times are better. 
#  Your output should be formatted as an array listing the names of the rides in the Skyline.

import sqlite3

db_filename = "readonly/dinofunworld.db"
conn = sqlite3.connect(db_filename)
cur = conn.cursor()

cur.execute("SELECT Name, attraction, COUNT(*), SUM(substr(duration,1,1)*3600 + substr(duration,3,2)*60 + substr(duration,6,2)) as s FROM checkin INNER JOIN attraction ON checkin.attraction = attraction.AttractionID WHERE category LIKE '%Ride%' GROUP BY Name")
result = cur.fetchall()
#print(result)

temp = list(result)

templist = []
templist2 = []
templist3 = []
for i in range(len(temp)):
    k = 0
    m = 0
    for j in range(len(temp)):
        if (temp[i][2] > temp[j][2]):
            k += 1
        if (temp[i][3] < temp[j][3]):
            m += 1
    if k == len(temp)-1 or m == len(temp)-1:
        templist.append(temp[i])
    else:
        templist2.append(temp[i])
        
for i in range(len(templist2)):
    k = 0
    m = 0
    for j in range(len(templist2)):
        if (templist2[i][2] > templist2[j][2]):
            k += 1
        if (templist2[i][3] < templist2[j][3]):
            m += 1
    if k == len(templist2)-1 or m == len(templist2)-1:
        templist3.append(temp[i])
        
for i in range(len(templist3)):
    if templist[0][2]
#print('\n')
print(templist)

#get the id, count(*), sum(duration) then process with for loops to get the skyline
#need to return a python list of ride ids

