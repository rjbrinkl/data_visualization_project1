import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


"""
Theme Colors in PowerPoint:
TAN  #e5ac77
STEEL GRAY #3f4d53

Secondary Colors:
OFF-WHITE #f3f3f3
WHITE #FFF
NAVY #0e2a47ff


"""

def main():
    cleanedDF = load_data()

    #show relationship between age and income in two different visualizations
    cleanedDF = drop_df(cleanedDF, columns= ['fnlwgt', 'capital_gain', 'capital_loss'])
    # graph all labels. Exit graph to load next graph
    for col in cleanedDF.head():
        create_hist(cleanedDF, col)


def drop_df(df, columns=['fnlwgt', 'capital_gain', 'capital_loss']):
    """remove column from df
    ['fnlwgt', 'capital_gain', 'capital_loss'] These columns dont seem that useful.
    Args:
        df (dataframe): data
        columns (list): list of columns to remove

    Returns:
        dataframe: data
    """
    return df.drop(columns=columns)
#create boxplot visualization between given column and income
def create_box(df, column):

    df.boxplot(column = [column], by = 'income')
    plt.show()
    
def create_line(df,column):
    """works best with hours worked"""
    bins = 15
    less_50k = df[df['income'] == 0]
    more_50k = df[df['income'] == 1]
    # print(less_50k)
    # print(more_50k)
    data=np.array(less_50k[column])
    y,binEdges=np.histogram(data,bins=bins)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    plt.plot(bincenters,y,'-',label='<=50K', color = "#e5ac77")

    data=np.array(more_50k[column])
    y,binEdges=np.histogram(data,bins=bins)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    plt.plot(bincenters,y,'-',label='>50K', color = "#3f4d53")
    plt.legend(loc='upper right')
    plt.xlabel(column) 
    plt.ylabel("Number of people")
    plt.grid(color='gray', linestyle='dashed')
    plt.show()  
   
 #Creates a double bar chart. Only tested with education. 
def create_bar(df, column):
    unique_val = pd.unique(df[column])
    condensedDF = df[['income', column]]
    high_inc = condensedDF.loc[df['income'] == 1]
    low_inc = condensedDF.loc[df['income'] == 0]
    high_inc_plot = []
    low_inc_plot = []
    colors = ['#e5ac77','#3f4d53']
    for val in unique_val:
        high_inc_plot.append(high_inc.loc[df[column] == val].shape[0])
        low_inc_plot.append(low_inc.loc[df[column] == val].shape[0])   
    
    newDF = pd.DataFrame({'More Than 50K': high_inc_plot, 'Less Than 50k': low_inc_plot}, index = unique_val)
    
    ax= newDF.plot.barh(width = 0.9, color = colors)
    ax.set_axisbelow(True)
    ax.xaxis.grid(color='gray', linestyle='dashed')
        

#create histogram visualization between given column and income
def create_hist(df, column):
    xaxis_label = 'income'
    condensedDF = df[[column, xaxis_label]]
    lessThanEqual50DF = condensedDF[condensedDF[xaxis_label] == 0]
    lessThanEqual50DF = lessThanEqual50DF[[column]]
    moreThan50DF = condensedDF[condensedDF[xaxis_label] == 1]
    moreThan50DF = moreThan50DF[[column]]
    bins = 25
    plt.hist(lessThanEqual50DF, bins, alpha=0.5, label='<=50K')
    plt.hist(moreThan50DF, bins, alpha=0.5, label='>50K')
    # plt.title('IDK')
    plt.xlabel(column) 
    plt.ylabel("Number of people")
    plt.xticks(rotation=0) # 90 is good for country label
    plt.legend(loc='upper right')
    plt.show()

#todo: generalize function
def create_pie_gender(df):
    low_income_bracket = df.loc[df['income'] == 0]
    high_income_bracket = df.loc[df['income'] == 1]
    
    low_inc_fem = low_income_bracket.loc[df['sex'] == 'Female']
    low_inc_male = low_income_bracket.loc[df['sex'] == 'Male']
    
    high_inc_fem = high_income_bracket.loc[df['sex'] == 'Female']
    high_inc_male = high_income_bracket.loc[df['sex'] == 'Male']
    
    low_inc_fem_count = low_inc_fem.shape[0]
    low_inc_male_count = low_inc_male.shape[0]
    
    high_inc_fem_count = high_inc_fem.shape[0]
    high_inc_male_count = high_inc_male.shape[0]
    
    total_females = low_inc_fem_count + high_inc_fem_count
    total_males = low_inc_male_count + high_inc_male_count
    
    plt.pie([low_inc_fem_count, high_inc_fem_count], labels = ['Less Than 50k', 'More Than 50K'], colors = ['#3f4d53', '#f3f3f3'], shadow = False, autopct='%.2f%%')
    plt.title("Female Income")
    plt.show()
    
    plt.pie([low_inc_male_count, high_inc_male_count], labels = ['Less Than 50k', 'More Than 50K'], colors = ['#e5ac77', '#f3f3f3'], shadow = False, autopct='%.2f%%')
    plt.title("Male Income")
    plt.show()
    




def load_data():
    # https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
    # local adult.data
	data = np.genfromtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", delimiter=', ', dtype=str)

	columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

	df = pd.DataFrame(data, columns=columns)

	df = df.astype({"age": np.int64, "education_num": np.int64, "hours_per_week": np.int64})

    #Other/Unknown = 0
    #Government = 2
    #Self-Employed = 1
	df = df.replace({'workclass': {'Without-pay': 0, 'Never-worked': 0,
																'Federal-gov': 2, 'State-gov': 2, 'Local-gov': 2,
																'Self-emp-not-inc': 1, 'Self-emp-inc': 1,
																'?': 0}})

    #White-Collar = 0
    #Blue-Collar = 1
    #Service = 2
    #Professional = 3
    #Other/Unknown = 4
	df = df.replace({'occupation': {'Adm-clerical': 0, 
																	'Craft-repair': 1,
																	'Exec-managerial': 4,'Farming-fishing': 1,
																	'Handlers-cleaners': 1,
																	'Machine-op-inspct': 1,'Other-service':2,
																	'Priv-house-serv':2,
																	'Prof-specialty': 3,'Protective-serv':2,
																	'Tech-support':2,
																	'Transport-moving': 1, 'Unknown': 0,
																	'Armed-Forces': 0, '?': 0}})

    #Single = 0
    #Married = 1
	df = df.replace({'marital_status': {'Married-civ-spouse': 1, 'Married-AF-spouse': 1, 'Married-spouse-absent': 1,
									'Never-married': 0}})

	df = df.replace({'income': {'<=50K': 0, '>50K': 1}})

    #Associate = 0
    #School = 1
	df = df.replace({'education': {'Assoc-voc': 'Associate', 'Assoc-acdm': 'Associate',
									'11th':'School', '10th':'School', '7th-8th':'School', '9th':'School',
									'12th':'School', '5th-6th':'School', '1st-4th':'School', 'Preschool':'School'}})

    #Other/Unknown = 0
    #United-States = 1
    #Cambodia = 2
    #England = 3 
    #Canada = 4 
    #Germany = 5 
    #India = 6 
    #Japan = 7 
    #Greece = 8 
    #China = 9
    #Cuba = 10
    #Iran = 11 
    #Honduras = 12
    #Philippines = 13 
    #Italy = 14 
    #Poland = 15 
    #Jamaica = 16 
    #Vietnam = 17 
    #Mexico = 18 
    #Portugal = 19 
    #Ireland = 20 
    #France = 21 
    #Dominican-Republic = 22 
    #Laos = 23 
    #Ecuador = 24 
    #Taiwan = 25  
    #Haiti = 26 
    #Columbia = 27 
    #Hungary = 28  
    #Guatemala = 29  
    #Nicaragua = 30  
    #Scotland = 31  
    #Thailand = 32 
    #Yugoslavia = 33  
    #El-Salvador = 34  
    #Trinadad&Tobago = 35  
    #Peru = 36 
    #Hong = 37
    #Holand-Netherlands = 38
	df = df.replace({'native_country': {'Outlying-US(Guam-USVI-etc)': 'United-States', '?': 'Other/Unknown', 'South': 'Other/Unknown', 'Puerto-Rico': 'United-States'}})

	return df

if __name__ == '__main__':
	main()