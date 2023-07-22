import pandas as pd
import numpy as np
#import plotly.express as px

from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
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
    #for col in cleanedDF.head():
    #    create_hist(cleanedDF, col)

    # for col in cleanedDF.head():
    #     create_bar(cleanedDF, col)

    #create_sunburst_race(cleanedDF)
    #create_sunburst_gender(cleanedDF)
    #create_pie(cleanedDF, 'race')
    #model(cleanedDF)

def model(cleanedDF):
    cleaned_onehot = cleanedDF.copy()
    
    #convert categorical columns into binary versions
    cleaned_onehot = pd.get_dummies(cleaned_onehot, columns=['workclass'], prefix = ['workclass'])
    cleaned_onehot = pd.get_dummies(cleaned_onehot, columns=['occupation'], prefix = ['occupation'])
    cleaned_onehot = pd.get_dummies(cleaned_onehot, columns=['marital_status'], prefix = ['marital_status'])
    cleaned_onehot = pd.get_dummies(cleaned_onehot, columns=['relationship'], prefix = ['relationship'])
    cleaned_onehot = pd.get_dummies(cleaned_onehot, columns=['race'], prefix = ['race'])
    cleaned_onehot = pd.get_dummies(cleaned_onehot, columns=['native_country'], prefix = ['native_country'])
    cleaned_onehot = pd.get_dummies(cleaned_onehot, columns=['education'], prefix = ['education_name'])
    
    labels = cleaned_onehot.columns.tolist()
    labels.remove('income')
    column_labels = cleaned_onehot[labels]
    class_labels = cleaned_onehot.income
    column_train, column_test, class_train, class_test = train_test_split(column_labels, class_labels, test_size=0.2, random_state=1)

    clf = DecisionTreeClassifier()
    # Train Decision Tree Classifer
    clf = clf.fit(column_train,class_train)
    #Predict the response for test dataset
    class_pred = clf.predict(column_test)
    
    #print accuracy of model
    print("Accuracy:",metrics.accuracy_score(class_test, class_pred))
    
    #importance of each column in the model
    importance_dict = dict(zip(labels, clf.feature_importances_))
    #print(importance_dict)  
    
    #combine importance for categorical columns together
    workclass_importance = calc_importance(importance_dict, 'workclass')
    occupation_importance = calc_importance(importance_dict, 'occupation')
    marital_status_importance = calc_importance(importance_dict, 'marital_status')
    relationship_importance = calc_importance(importance_dict, 'relationship')
    race_importance = calc_importance(importance_dict, 'race')
    native_country_importance = calc_importance(importance_dict, 'native_country')
    education_importance = calc_importance(importance_dict, 'education_name')

    #importance of each feature varies from run to run but by no more than 0.01
    importance = {'marital status': marital_status_importance, 'age': importance_dict['age'], 'education number': importance_dict['education_num'], 
                'hours per week': importance_dict['hours_per_week'], 'occupation': occupation_importance, 
                'workclass': workclass_importance, 'relationship': relationship_importance, 'native country': native_country_importance, 
                'race': race_importance, 'education': education_importance, 'gender': importance_dict['gender']}
                
    #top 8 in order are marital_status, age, education_num, hours_per_week, occupation, workclass, relationship, native_country
    print(importance)
    
    # Data to plot
    labels = []
    values = []

    for i, j in importance.items():
        labels.append(i)
        values.append(j)

    colors = ["#e5ac77", "#3f4d53", "#fec15c", "#c5c5c6", "#e1e18d", "#7d807e", "#e8afaf", "#fef9f0", "#f77059", "#fbef10", "#9c8187"]
    plt.pie(values, labels=labels, colors = colors, shadow = False, autopct = '%.2f%%')
    plt.title('Factor Importance in Percentages')
    plt.show()
    
def calc_importance(dict, column):
    importance = 0
    for i in dict:
        if i.startswith(str(column) + '_'):
            importance += dict[i]
    return importance

def drop_df(df, columns=['fnlwgt', 'capital_gain', 'capital_loss', 'education']):
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
    plt.show()
        

#create histogram visualization between given column and income
def create_hist(df, column):
    xaxis_label = 'income'
    condensedDF = df[[column, xaxis_label]]
    lessThanEqual50DF = condensedDF[condensedDF[xaxis_label] == 0]
    lessThanEqual50DF = lessThanEqual50DF[[column]]
    moreThan50DF = condensedDF[condensedDF[xaxis_label] == 1]
    moreThan50DF = moreThan50DF[[column]]
    bins = 25
    plt.style.use('seaborn-whitegrid') # nice and clean grid
    plt.hist(lessThanEqual50DF, bins,  label='<=50K', facecolor='#3f4d53', edgecolor="#3f4d53")
    plt.hist(moreThan50DF, bins,  label='>50K', facecolor = '#e5ac77', edgecolor="#e5ac77")
    # plt.title('IDK')
    plt.xlabel(column) 
    plt.ylabel("Number of people")
    plt.xticks(rotation=0) # 90 is good for country label
    plt.legend(loc='upper right')
    plt.show()


#creates pie chart comparing high vs low income for each member of the population    
def create_pie(df, column):
    unique_val = pd.unique(df[column])
    condensedDF = df[['income', column]]
    high_inc = condensedDF.loc[df['income'] == 1]
    low_inc = condensedDF.loc[df['income'] == 0]
    colors = ['#e5ac77','#3f4d53']
    for val in unique_val:
        high_inc_count = high_inc.loc[df[column] == val].shape[0]
        low_inc_count = low_inc.loc[df[column] == val].shape[0]
        plt.pie([low_inc_count, high_inc_count], labels = ['Less Than 50k', 'More Than 50K'], colors = colors, shadow = False, autopct ='%.2f%%') 
        plt.title(val + ' Income')
        plt.show()

#creates sunburst graph by race
def create_sunburst_race(df):
    dist = df[['race', 'income', 'age']]
    dist = dist.replace({'income': {0: '<=50K', 1: '>50K'}})
    dist = dist.groupby(by=['race', 'income']).count()[["age"]].rename(columns={'age':'count'})
    dist['data'] = 'US Census'
    dist = dist.reset_index()
    dist.head()

    fig = px.sunburst(dist,
        path=['data', 'race', 'income'],
        values='count',
        title='US Census Count Distribution [Race, Income]',
        color='count',
        color_continuous_scale="RdBu_r",
        width=1000, height=1000)
    fig.write_html('sunburst_race_income.html')
    # fig.write_image('sunburst_race_income.png')

    fig.show()


#creates sunburst graph by gender
def create_sunburst_gender(df):
    dist = df[['gender', 'income', 'age']]
    dist = dist.replace({'income': {0: '<=50K', 1: '>50K'}})
    dist = dist.replace({'gender': {0: 'Female', 1: 'Male'}})
    dist = dist.groupby(by=['gender', 'income']).count()[["age"]].rename(columns={'age':'count'})
    dist['data'] = 'US Census'
    dist = dist.reset_index()
    dist.head()

    fig = px.sunburst(dist,
        path=['data', 'gender', 'income'],
        values='count',
        title='US Census Count Distribution [Gender, Income]',
        color='count',
        color_continuous_scale="RdBu_r",
        width=1000, height=1000)
    fig.write_html('sunburst_gender_income.html')
    # fig.write_image('sunburst_race_income.png')

    fig.show()


def load_data():
    # https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
    # local adult.data
    try:
        data = np.genfromtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", delimiter=', ', dtype=str)
    except :
        data = np.genfromtxt("adult.data", delimiter=', ', dtype=str)

    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'gender', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

    df = pd.DataFrame(data, columns=columns)

    df = df.astype({"age": np.int64, "education_num": np.int64, "hours_per_week": np.int64})

    df = df.replace({'gender': {'Female': 0, 'Male': 1}})

    df = df.replace({'workclass': {'Without-pay': 'Other/Unknown', 'Never-worked': 'Other/Unknown',
                                                                'Federal-gov': 'Government', 'State-gov': 'Government', 'Local-gov':'Government',
                                                                'Self-emp-not-inc': 'Self-Employed', 'Self-emp-inc': 'Self-Employed',
                                                                '?': 'Other/Unknown'}})

    df = df.replace({'occupation': {'Adm-clerical': 'White-Collar', 
                                                                    'Craft-repair': 'Blue-Collar',
                                                                    'Exec-managerial':'White-Collar','Farming-fishing':'Blue-Collar',
                                                                    'Handlers-cleaners':'Blue-Collar',
                                                                    'Machine-op-inspct':'Blue-Collar','Other-service':'Service',
                                                                    'Priv-house-serv':'Service',
                                                                    'Prof-specialty':'Professional','Protective-serv':'Service',
                                                                    'Tech-support':'Service',
                                                                    'Transport-moving':'Blue-Collar','Unknown':'Other/Unknown',
                                                                    'Armed-Forces':'Other/Unknown','?':'Other/Unknown'}})

    df = df.replace({'marital_status': {'Married-civ-spouse': 'Married', 'Married-AF-spouse': 'Married', 'Married-spouse-absent':'Married',
                                    'Never-married':'Single'}})

    df = df.replace({'income': {'<=50K': 0, '>50K': 1}})

    df = df.replace({'education': {'Assoc-voc': 'Associate', 'Assoc-acdm': 'Associate',
                                    '11th':'School', '10th':'School', '7th-8th':'School', '9th':'School',
                                    '12th':'School', '5th-6th':'School', '1st-4th':'School', 'Preschool':'School'}})

    df = df.replace({'native_country': {'Outlying-US(Guam-USVI-etc)': 'United-States', '?': 'Other/Unknown', 'South': 'Other/Unknown', 'Puerto-Rico': 'United-States'}})

    return df

if __name__ == '__main__':
    main()