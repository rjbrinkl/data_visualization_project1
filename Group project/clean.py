'''Documentation'''
# imports
import pandas as pd

# functions
def clean_data(filename='adult.data.txt'):
    '''
    None removed 32562
    all ? removed 30163

    TODO figure out if Enums are good

    Parameters
        ----------
        filename : str
            The name of file to read
    Returns
        -------
        list
            a list of lists
    '''


    # format parmeters
    conv = {
    'Private' : 0, 'Self-emp-not-inc':1, 'Self-emp-inc':2, 'Federal-gov':3, 'Local-gov':4, 'State-gov':5, 'Without-pay':6, 'Never-worked':7,
    'Preschool':0,'1st-4th':1,'5th-6th':2,'7th-8th':3,'9th':4,'10th':5,'11th':6,'12th':7,'HS-grad':8,'Some-college':9,'Assoc-acdm':10,'Assoc-voc':11,'Prof-school':12,'Bachelors':13,'Masters':14,'Doctorate':15,
    'Married-civ-spouse':0, 'Divorced':1, 'Never-married':2, 'Separated':3, 'Widowed':4, 'Married-spouse-absent':5, 'Married-AF-spouse':6,
    'Tech-support':0, 'Craft-repair':1, 'Other-service':2, 'Sales':3, 'Exec-managerial':4, 'Prof-specialty':5, 'Handlers-cleaners':6, 'Machine-op-inspct':7, 'Adm-clerical':8, 'Farming-fishing':9, 'Transport-moving':10, 'Priv-house-serv':11, 'Protective-serv':12, 'Armed-Forces':13,
    'Wife':0, 'Own-child':1, 'Husband':2, 'Not-in-family':3, 'Other-relative':4, 'Unmarried':5,
    'White':0, 'Asian-Pac-Islander':1, 'Amer-Indian-Eskimo':2, 'Other':3, 'Black':4,
    'Female':0, 'Male':1,
    'United-States':0, 'Cambodia':1, 'England':2, 'Puerto-Rico':3, 'Canada':4, 'Germany':5, 'Outlying-US(Guam-USVI-etc)':6, 'India':7, 'Japan':8, 'Greece':9, 'South':10, 'China':11, 'Cuba':12, 'Iran':13, 'Honduras':14, 'Philippines':15, 'Italy':16, 'Poland':17, 'Jamaica':18, 'Vietnam':19, 'Mexico':20, 'Portugal':21, 'Ireland':22, 'France':23, 'Dominican-Republic':24, 'Laos':25, 'Ecuador':26, 'Taiwan':27, 'Haiti':28, 'Columbia':29, 'Hungary':30, 'Guatemala':31, 'Nicaragua':32, 'Scotland':33, 'Thailand':34, 'Yugoslavia':35, 'El-Salvador':36, 'Trinadad&Tobago':37, 'Peru':38, 'Hong':39, 'Holand-Netherlands':40,
    '<=50K':0,'>50K':1
    }
    #open file and data

    file = open(filename, 'r')
    data = list()
    #read file
    for line in file:
        # remove ? from data
        if '?' in line:
            continue

        line = line.strip()
        # if not None
        if line:
            age, workc, fnlwgt, edu, edu_num, marital_s, occ, rel, race, sex, cap_gain, cap_loss, hours_week, native_c,income = line.strip().replace(' ', '').split(',')
            t =[int(age), conv[workc], int(fnlwgt), conv[edu], int(edu_num), conv[marital_s], conv[occ], conv[rel], conv[race], conv[sex], int(cap_gain), int(cap_loss), int(hours_week), conv[native_c],conv[income] ]
            data.append(t)
    #print(data)
    
    #create dataframe from cleaned data
    resultsDF = pd.DataFrame(data, columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours-per-week', 'native-country', 'income >50K?']) 
    
    #write results to file without index or header
    resultsDF.to_csv(r'./Cleaned_Data.csv', index = False)    
    return resultsDF
# main
if __name__ == '__main__':
    clean_data()