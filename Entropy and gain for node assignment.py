"""
Contact:
Zachary Hunt
hunt.590@buckeyemail.osu.edu
"""

# Implementing entropy and gain functions for a given dataset (CSV file, last attribute is class label)

import pandas as pd
import numpy as np
path = input('Please enter the path to your csv file: ')
#path = 'cars.csv'
def entropy(x):
    return -(x) * np.log2(x)

def determination(path):
    
    data = pd.read_csv(path)
    df = pd.DataFrame(data)
    df = df.dropna()
    class_label = df.columns[-1]
    class_df = df[class_label]
    unique_classes = class_df.unique()
    num_classes = len(unique_classes)
    column_list = []
    for columns in df.columns:
        column_list.append(columns)

    column_list = column_list[:-1]
    num_columns = len(column_list)
    
    class_count = class_df.value_counts().tolist()
    summ = np.sum(class_count)
    class_entropy = 0
    for i in class_count:
        prop = i/summ
        ent = entropy(prop)
        class_entropy+=ent


    entropies = []
    for column in column_list:
        grouped = df.groupby(column)[class_label].value_counts()
        sum_val=0
        mapping = dict()
        for i, row in grouped.iteritems():
            if not (i[0] in mapping):
                mapping[i[0]] = []
            mapping.get(i[0]).append(row)

        attr_ent = 0
        tot_sum = 0
        for key, value in mapping.items():
            temp_sum = np.sum(value)
            tot_sum += temp_sum
            sum_entrop = 0
            for v in value:
                prop = v/temp_sum
                ent = entropy(prop)
                sum_entrop+=ent
            attr_ent += sum_entrop*temp_sum

        column_entr = float(attr_ent/tot_sum)
        entropies.append(column_entr)
    [print('The entropy for column', i+1, 'is', entropies[i]) for i in range(len(entropies))]
    
    [print('The gain for column', i+1, 'is', (class_entropy - entropies[i])) for i in range(len(entropies))]
    
    gain_list = []
    for i in range(len(entropies)):
        gain_list.append(class_entropy - entropies[i])
    
    max_gain = max(gain_list)
    
    print('The maximum gain obtained is', max_gain, 'which means the column that corresponds to this value should be used as the root node')
    
    

determination(path)

"""
Please enter the path to your csv file: cars.csv
The entropy for column 1 is 0.7507717108679162
The entropy for column 2 is 0.8709516529812099
The gain for column 1 is 0.14897204783034645
The gain for column 2 is 0.028792105717052707
The maximum gain obtained is 0.14897204783034645 which means the column that corresponds to this value should be used as the root node
"""
    
