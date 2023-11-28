'''CY4 h11 symbolic regression -- reproducing section 4.1.2 of the paper'''
#Import libraries
import numpy as np
import gzip
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from gplearn.genetic import SymbolicRegressor
#from sympy import sympify

#Define path to data
path = './Data/5dTransWH.all.gz' #...note this data is read directly from the zip file

#Import data
with gzip.open(path, 'rb') as file:
    weights, h11 = [], []
    for line in file.readlines():
        line_str = str(line).replace('b\'','').replace(']\\n\'','').replace('=d','').replace(':',',').replace('[','').replace(' ',',').split(',')
        weights.append(line_str[:6])     
        h11.append(line_str[-4])

weights = np.array(weights,dtype='int')
h11 = np.array(h11,dtype='int')
del(file,line,line_str)

#%% #Data setup
ML_data = [[weights[index],h11[index]] for index in range(len(h11))]
s = int(np.floor(0.8*len(h11)))
np.random.shuffle(ML_data)
Training_data =   np.array([datapoint[0] for datapoint in ML_data[:s]])
Training_labels = np.array([datapoint[1] for datapoint in ML_data[:s]])
Testing_data =    np.array([datapoint[0] for datapoint in ML_data[s:]])
Testing_labels =  np.array([datapoint[1] for datapoint in ML_data[s:]])
del(ML_data)

#%% #Define and fit the Sregressor
#Choose functions from ['add','sub','mul','div','neg','sqrt','log','abs','inv','max','min','sin','cos','tan']
SR = SymbolicRegressor(population_size=1000, function_set=['add','sub','mul','div'], metric='mean absolute error', generations=20, stopping_criteria=0.01, const_range=(-10,10),
                       p_crossover=0.8, p_subtree_mutation=0.01, p_hoist_mutation=0.1, p_point_mutation=0.01,
                       max_samples=1, verbose=1, parsimony_coefficient=0.99)#, random_state=1)

SR.fit(Training_data, Training_labels)

#%% #Test the Sregressor
prediction = SR.predict(Testing_data)
Score = SR.score(Testing_data, Testing_labels)
print('R^2:\t',Score)
print('MAE:\t',MAE(Testing_labels,prediction))
print('MAPE:\t',MAPE(Testing_labels,prediction))

#Output the final equation ---> needs sympy
converter = {
    'add' : lambda x, y : x + y,
    'sub' : lambda x, y : x - y,
    'mul' : lambda x, y : x*y,
    'div' : lambda x, y : x/y,
    'neg' : lambda x    : -x,
    'sqrt': lambda x    : x**0.5,
    'log' : lambda x    : log(x),
    'abs' : lambda x    : abs(x),
    'inv' : lambda x    : 1/x,
    'max' : lambda x    : max(x),
    'min' : lambda x    : min(x),
    'sin' : lambda x    : sin(x),
    'cos' : lambda x    : cos(x),
    'tan' : lambda x    : tan(x)
}
Eq = str(SR._program)
#Eq = sympify(str(SR._program), locals=converter) 
print('Equation:',Eq)


#%% #Compute trial functions R^2 scores on test data
from sklearn.metrics import r2_score

#On test dataset
predictions = np.array([0.75*i[1]+i[2]+i[3]*0.375 for i in Training_data]) ### --> hardcode trial functions here
print(r2_score(Training_labels,predictions))
print(MAPE(Training_labels,predictions))

#On full dataset
#predictions = np.array([i[1]+i[2]+i[4]/6 for i in weights]) ### --> hardcode trial functions here
#print(r2_score(h11,predictions))

#%% #Plot the predictions
import matplotlib.pyplot as plt
plt.figure('Difference')
plt.scatter(Testing_labels,prediction,alpha=0.1)
#plt.scatter(Testing_labels,abs(Testing_labels-prediction),alpha=0.1)
#plt.scatter(Testing_labels,abs((Testing_labels-prediction)/Testing_labels),alpha=0.1)
plt.xlabel('Parameter value')
plt.ylabel('Prediction difference')
#plt.xscale('log')
#plt.yscale('log')
plt.grid()
#plt.savefig('./....pdf')
