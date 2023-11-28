"""
Import the four-folds dataset, and perform machine-learning with a simply-connected
regressor neural network to predict the cohomological data and the Euler number.
The code below reproduces section 4.1 of the paper.
"""

#Import libraries
# General ones
import numpy as np
import gzip
from math import floor

# For ML
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_percentage_error as MAPE

#Define path to data
path = './Data//5dTransWH.all.gz'

# Import data
# The data contain the three Hodge numbers h^{1,1}, h^{1,2}, h^{1,3} and the Euler number \Chi.
# h^{2,2} can be obtained by 4 + 2 h^{1,1} - 4 h{1,2} + 2 h{1,3} + h^{2,2} = \chi
with gzip.open(path, 'rb') as file:
    weights, chi, h11, h12, h13, h22, triple_h = [], [], [], [], [], [], []
    for line in file.readlines():
        line_str = str(line).replace('b\'','').replace(']\\n\'','').replace('=d','').replace(':',',').replace('[','').replace(' ',',').split(',')
        chi_ = int(line_str[-1])
        h_11 = int(line_str[-4])
        h_12 = int(line_str[-3])
        h_13 = int(line_str[-2])
        h_22 = -(4 + 2*h_11 - 4*h_12 + 2*h_13 - chi_)
        weights.append(line_str[:6])     
        chi.append(chi_)
        h11.append(h_11)
        h12.append(h_12)
        h13.append(h_13)
        h22.append(h_22)
        # Can be alternatively calcualted as:
        # h22.append(1/3*(84 + chi_ + 6*h_11 + 6*h_13))
        
        triple_h.append([h_11, h_13, h_22])
        # Given the extra constraint, coming from the Atiyah-Singer theorem, 
        # i.e. -4 h^{1,1} + 2 h^{2,1} - 4 h^{3,1} + h^{2,2} - 44 = 0,
        # only 3 numbers out of the (Hodges, Euler) are really independent.
        # Here we choose those three because they are not zero, so they give
        # a well-defined MAPE when investigating ML performances.

# Bring the variables into a suitable format.
weights = np.array(weights,dtype='int')
chi = np.array(chi)
h11=np.array(h11)
h12=np.array(h12)
h13=np.array(h13)
h22=np.array(h22)
triple_h = np.array(triple_h)

chi=chi.astype(float)
h11=h11.astype(float)
h12=h12.astype(float)
h13=h13.astype(float)
h22=h22.astype(float)
triple_h=triple_h.astype(float)

#%% --------------------------------------------------------------------------------------
#   ML investigations are performed below.

#%% Prepare the data end perform ML; we select a large random sample to speed up computations.

size=150000
indices=np.arange(0,round(len(chi)))
# This investigates the whole dataset; alternatively, we also performed these 
# tests on the two halves: indices=np.arange(0,round(len(chi)/2)) for the first half
# and indices=np.arange(round(len(chi)/2),round(len(chi))) for the second half.

rand_indices=np.random.choice(indices, size, replace=False)

# Take random sample
CY_rand, chi_rand, h11_rand, h12_rand, h13_rand, h22_rand, triple_h_rand = [], [], [], [], [], [], []
for i in range(size):
    CY_rand.append(weights[rand_indices[i],:])
    chi_rand.append(chi[rand_indices[i]])
    h11_rand.append(h11[rand_indices[i]])
    h12_rand.append(h12[rand_indices[i]])
    h13_rand.append(h13[rand_indices[i]])
    h22_rand.append(h22[rand_indices[i]])
    triple_h_rand.append(triple_h[rand_indices[i]])
    
# Bring the variables into a suitable format.
CY_rand=np.array(CY_rand)
chi_rand=np.array(chi_rand)
h11_rand=np.array(h11_rand)
h12_rand=np.array(h12_rand)
h13_rand=np.array(h13_rand)
h22_rand=np.array(h22_rand)
triple_h_rand=np.array(triple_h_rand)
CY_rand=CY_rand.astype(float) 
chi_rand=chi_rand.astype(float)
h11_rand=h11_rand.astype(float)
h12_rand=h12_rand.astype(float)
h13_rand=h13_rand.astype(float)
h22_rand=h22_rand.astype(float)
triple_h_rand=triple_h_rand.astype(float)

investigation = 1     
# ...choose what property to ML from: [Euler number, h11, h12, h13, h22, 
# "Triple"(=[h11,h13,h22] by default) ] with the respective index.
k = 5                  
# ...number of k-fold cross-validations to perform (k = 5 => 80(train) : 
# 20(test) splits approx.)

if   investigation == 0: outputs = chi_rand
elif investigation == 1: outputs = h11_rand
elif investigation == 2: outputs = h12_rand
elif investigation == 3: outputs = h13_rand
elif investigation == 4: outputs = h22_rand
elif investigation == 5: outputs = triple_h_rand

#Zip input and output data together
data_size = len(CY_rand)
ML_data = [[CY_rand[index],outputs[index]] for index in range(data_size)]

#Shuffle data ordering
np.random.shuffle(ML_data)
s = int(floor(data_size/k)) #...number of datapoints in each validation split

#Define data lists, each with k sublists with the relevant data for that cross-validation run
Train_inputs, Train_outputs, Test_inputs, Test_outputs = [], [], [], []
for i in range(k):
    Train_inputs.append([datapoint[0] for datapoint in ML_data[:i*s]]+[datapoint[0] for datapoint in ML_data[(i+1)*s:]])
    Train_outputs.append([datapoint[1] for datapoint in ML_data[:i*s]]+[datapoint[1] for datapoint in ML_data[(i+1)*s:]])
    Test_inputs.append([datapoint[0] for datapoint in ML_data[i*s:(i+1)*s]])
    Test_outputs.append([datapoint[1] for datapoint in ML_data[i*s:(i+1)*s]])

#Run NN train & test 
#Define measure lists
MSEs, MAPEs, Rsqs = [], [], []    #...lists of regression measures
seed = 2                          #...select a random seeding (any integer) for regressor initialisation
hist = [] #record histroy and plot over training

#Loop through each cross-validation run
for i in range(k):
    #Define & Train NN Regressor directly on the data
    nn_reg = MLPRegressor((16,32,16),activation='relu',solver='adam',random_state=seed, max_iter=250, n_iter_no_change=20)  
    #...can edit the NN structure here; default max_iter=200 and default n_iter_no_change=10
    nn_reg.fit(Train_inputs[i], Train_outputs[i])
    
    #Compute NN predictions on test data, and calculate learning measures
    Test_pred = nn_reg.predict(Test_inputs[i])
    Rsqs.append(nn_reg.score(Test_inputs[i],Test_outputs[i]))
    MSEs.append(MSE(Test_outputs[i],Test_pred,squared=True))             #...True -> mse, False -> root mse
    check = Test_outputs[i] - Test_pred
    if investigation != 0 and investigation != 3: MAPEs.append(MAPE(Test_outputs[i],Test_pred)) #...note not defined for learning euler or h12, as true value can be 0

#Output averaged learning measures with standard errors
print('####################################')
print('Average Measures (investigation '+str(investigation)+'):')
print('R^2: ',sum(Rsqs)/k,'\pm',np.std(Rsqs)/np.sqrt(k))
print('MSE: ',sum(MSEs)/k,'\pm',np.std(MSEs)/np.sqrt(k))
if investigation != 0 and investigation != 2: print('MAPE:',sum(MAPEs)/k,'\pm',np.std(MAPEs)/np.sqrt(k)) 
# MAPE is not well-defined for \chi and h^{1,2}, which can be zeroes.

#%% Extract the key pieces of information (i.e. min, max and mean values)   
# for the data under investigation, calculated over the two halves of the dataset.  

w_tot = []
# Define an array with the sum of the weights for each system.
for i in range(len(weights)):
    w_tot.append(sum(weights[i,:]))

w_tot = np.array(w_tot).astype(float)

investigation = 5   # Choose which feature to focus on.

if   investigation == 0: outputs = chi
elif investigation == 1: outputs = h11
elif investigation == 2: outputs = h12
elif investigation == 3: outputs = h13
elif investigation == 4: outputs = h22
elif investigation == 5: outputs = w_tot

print("Over the first half of the dataset:")
print("Min: ", min(outputs[0:round(len(outputs)/2)]) )
print("Max: ", max(outputs[0:round(len(outputs)/2)]) )
print("Mean: ", sum(outputs[0:round(len(outputs)/2)])/round(len(outputs)/2) )

print("Over the first half of the dataset:")
print("Min: ", min(outputs[round(len(outputs)/2):len(outputs)]) )
print("Max: ", max(outputs[round(len(outputs)/2):len(outputs)]) )
print("Mean: ", sum(outputs[round(len(outputs)/2):len(outputs)])/round(len(outputs)/2) )



