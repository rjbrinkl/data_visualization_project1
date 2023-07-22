import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def main():
    cleanedDF = load_data()

    #show relationship between age and income in two different visualizations
    create_box(cleanedDF, 'age')
    
    create_hist(cleanedDF, 'age')
    
#create boxplot visualization between given column and income
def create_box(df, column):

    df.boxplot(column = [column], by = 'income')
    plt.show()
    
#create histogram visualization between given column and income
def create_hist(df, column):
    condensedDF = df[[column, 'income']]
    lessThanEqual50DF = condensedDF[condensedDF['income'] == 0]
    lessThanEqual50DF = lessThanEqual50DF[[column]]
    moreThan50DF = condensedDF[condensedDF['income'] == 1]
    moreThan50DF = moreThan50DF[[column]]
    bins = 25
    plt.hist(lessThanEqual50DF, bins, alpha=0.5, label='<=50K')
    plt.hist(moreThan50DF, bins, alpha=0.5, label='>50K')
    plt.legend(loc='upper right')
    plt.show()


def load_data():
	data = np.genfromtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", delimiter=', ', dtype=str)

	columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

	df = pd.DataFrame(data, columns=columns)

	df = df.astype({"age": np.int64, "education_num": np.int64, "hours_per_week": np.int64})

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
