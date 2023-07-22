
# coding: utf-8

# # Assignment 2: Graphing Dino Fun World
# 
# Impressed by your previous work, the administrators of Dino Fun World have asked you to create some charts that they can use in their next presentation to upper management. The data used for this assignment will be the same as the data used for the previous asisgnment.
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
# The administrators would like you to create four graphs: a pie chart, a bar chart, a line chart, and a box-and-whisker plot. All of these plots can be created with the data provided.
# 
# Chart 1: A Pie Chart depicting visits to thrill ride attractions.
# Chart 2: A Bar Chart depicting total visits to food stalls.
# Chart 3: A Line Chart depicting attendance at the newest ride, Atmosfear over the course of the day.
# Chart 4: A Box-and-Whisker Plot depicting total visits to the park's Kiddie Rides.
# 
# #### Administrative Notes
# 
# This assignment will be graded by Coursera's grading system. In order for your answers to be correctly registered in the system, you must place the code for your answers in the cell indicated for each question. In addition, you should submit the assignment with the output of the code in the cell's display area. The display area should contain only your answer to the question with no extraneous information, or else the answer may not be picked up correctly. Each cell that is going to be graded has a set of comment lines at the beginning of the cell. These lines are extremely important and must not be modified or removed.

# In[9]:


# Graded Cell, PartID: gtMqY
# Make a Pie Chart of the visits to Thrill Ride attractions. For this question,
#  display the pie chart in the notebook and print the data used to create the
#  pie chart as a list of lists (ex: [['Ride 1', 10], ['Ride 2', 100], ...])

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

db_filename = 'readonly/dinofunworld.db'
conn = sqlite3.connect(db_filename)
cur = conn.cursor()

cur.execute("SELECT Name, COUNT(*) FROM attraction INNER JOIN checkin ON attraction.AttractionID = checkin.attraction WHERE Category LIKE '%Thrill%' GROUP BY Name")
result = cur.fetchall()
#print(result)

temp = []
for i in range(len(result)):
    temp.append([result[i][0], result[i][1]])
    
print(temp)

df = pd.DataFrame.from_records(result, columns=['ThrillRides', 'Visits'])
plt.pie(df['Visits'], labels=df['ThrillRides'], shadow=False)
plt.axis('equal')
plt.show()


# In[4]:


# Graded Cell, PartID: 9Ocyl
# Make a bar chart of total visits to food stalls. For this question,
#  display the bar chart in the notebook and print the data used to create the
#  bar chart as a list of lists (ex: [['Stall 1', 10], ['Stall 2', 50], ...])

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

db_filename = 'readonly/dinofunworld.db'
conn = sqlite3.connect(db_filename)
cur = conn.cursor()

cur.execute("SELECT Name, COUNT(*) FROM attraction INNER JOIN checkin ON attraction.AttractionID = checkin.attraction WHERE Category LIKE '%Food%' GROUP BY Name")
result = cur.fetchall()
#print(result)

temp = []
for i in range(len(result)):
    temp.append([result[i][0], result[i][1]])
    
print(temp)

df = pd.DataFrame.from_records(result, columns=['FoodStalls', 'Visits'])
plt.bar(range(len(df['Visits'])), df['Visits'])
plt.xticks([])
plt.show()


# In[59]:


# Graded Cell, PartID: 0zcEV
# Make a line chart of attendance at Atmosfear every five minutes. Again,
#  display the line chart in the notebook and print the data used to create the
#  chart as a list of lists (ex: [['Stall 1', 10], ['Stall 2', 50], ...])

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

db_filename = 'readonly/dinofunworld.db'
conn = sqlite3.connect(db_filename)
cur = conn.cursor()

cur.execute("SELECT visitorID, sequence FROM sequences WHERE sequence LIKE '%-8-%'")
result = cur.fetchall()

temp = [0] * 192
temp2 = []
length = len(result[0])

for i in range(len(result)):
    dashnum = 0
    for j in range(len(result[0][1])):
        if result[i][1][j] == "8":
            if result[i][1][j+1] == "-" and j != 0 and result[i][1][j-1] == "-":
                temp[dashnum] += 1
            elif j == 0 and result[i][1][j+1] == "-":
                temp[dashnum] += 1
        elif result[i][1][j] == "-":
            dashnum += 1
            if dashnum == 192:
                break
            
for i in range(len(temp)):
    temp2.append(['Bin ' + str(i), temp[i]])
print (temp2)

df = pd.DataFrame.from_records(temp2, columns=['Time', 'Visits'])
plt.plot(df['Time'], df['Visits'], color='red', marker='o')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Visits', fontsize=14)
plt.grid(True)
plt.show()


# In[8]:


# Graded Cell, PartID: zdzaT
# Make a box plot of total visits to rides in the Kiddie Rides category. For
#  this question, display the box plot in the notebook and print the number of
#  visits to each ride as a list (ex: [3, 4, 5, 6, ...])

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

db_filename = 'readonly/dinofunworld.db'
conn = sqlite3.connect(db_filename)
cur = conn.cursor()

cur.execute("SELECT AttractionID, COUNT(*) FROM attraction INNER JOIN checkin ON attraction.AttractionID = checkin.attraction WHERE Category LIKE '%Kiddie%' GROUP BY AttractionID ORDER BY AttractionID")
result = cur.fetchall()
#print(result)

temp = []
for i in range(len(result)):
    temp.append(result[i][1])
print(temp)

df = pd.DataFrame.from_records(result, columns=['KiddieRides', 'Visits'])
plt.boxplot(df['Visits'])
plt.show()

