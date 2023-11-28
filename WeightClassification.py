'''Script to perform NN classification between WPS5 weight vectors in various partitions -- reproducing section 4.2 of the paper'''
'''
To run: Cells are delineated by '#%%', cells individually tailored and run sequentially to reproduce quoted results.
    Cell 1 - Import libraries and data (note to unzip the PartitionData first).
    Cell 2 - Data setup. Select the investigation: this is done either by setting 'multiclassification'' to True then the classification amongst all the classes is run, otherwise setting it to False and then deciding the individual binary classification to perform by setting 'dataset_selection' to one of the above row's list. There is also additional functionality to track the saliency of the ML.
    Cell 3 - Perform the ML, hyperparameters can be tailored at the start of the cell.
    Cell 4 - Plot performance throughout training.
    Cell 5 - Output and format the averaged confusion matrix.
'''
#Import libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras, Variable, GradientTape

#Define path to data
partitionroot_path = './Data/Partition/'
partition_filenames = ['CnIPnD.txt','DnIP.txt','IPnRnD.txt','IPRnD.txt','DnR.txt','DR.txt','CYnR.txt','CYR.txt']

#Import data
all_data = []
for filename in partition_filenames:
    weights = []
    with open(partitionroot_path+filename,'r') as file:
        for line in file.readlines():
            weights.append(eval(line))
    all_data.append(np.array(weights))
    print(f'...imported {filename[:-4]}')
    
CnIPnD, DnIP, IPnRnD, IPRnD, DnR, DR, CYnR, CYR = all_data
print(f'Datasizes: {list(map(len,all_data))}')
del(file,filename,line,weights)

#%% #Setup the data
multiclassification = True    #...choose whether to run the multiclassification investigation between all classes, or a binary classification with the selected class below
saliency = False              #...choose whether to compute saliency over the test data

if multiclassification: #...multiclassification between all datasets
    inputs = [ws for dataset in all_data for ws in dataset]
    values = [dataset_idx for dataset_idx in range(len(all_data)) for ws in all_data[dataset_idx]]
    number_classes = len(all_data)
    class_weights = {i:w for i,w in enumerate(np.sum(list(map(len,all_data)))/np.array(list(map(len,all_data))))}
else:                   #...select two subsets to binary classify between
    #IP: [[0,1],[2,3,4,5,6,7]]
    #D: [[0,2,3],[1,4,5,6,7]]
    #R: [[2,4,6],[3,5,7]]
    #CY: [[0,1,2,3,4,5],[6,7]]
    #CYR: [[7],[6]]
    dataset_selection = [[7],[6]] ### --> update to one of above
    number_classes = 2
    inputs = [ws for binaryset in dataset_selection for dataset_idx in binaryset for ws in all_data[dataset_idx]]
    values = [binary_label for binary_label in [0,1] for dataset_selection_idx in dataset_selection[binary_label] for ws in all_data[dataset_selection_idx]]
    class_weights = {0: len(values)/(len(values)-np.sum(values)), 1: len(values)/np.sum(values)}
    print('Binary split:',[len(values)-np.sum(values),np.sum(values)])
    
ML_data = [[inputs[index],values[index]] for index in range(len(values))]

#Number of k-fold cross-validations to perform
k = 5   #... k = 5 => 80(train) : 20(test) splits approx.

#Shuffle data ordering
np.random.shuffle(ML_data)
#Parition the data into (train,test)
Training_data, Training_labels, Testing_data, Testing_labels = [], [], [], []
if k == 1 and not saliency:
    s = int(np.floor(0.8*len(values)))
    Training_data.append(  [datapoint[0] for datapoint in ML_data[:s]])
    Training_labels.append([datapoint[1] for datapoint in ML_data[:s]])
    Testing_data.append(   [datapoint[0] for datapoint in ML_data[s:]])
    Testing_labels.append( [datapoint[1] for datapoint in ML_data[s:]])
elif k > 1 and not saliency:
    s = int(np.floor(len(values)/k)) 
    for i in range(k):
        Training_data.append(  [datapoint[0] for datapoint in ML_data[:i*s]]+[datapoint[0] for datapoint in ML_data[(i+1)*s:]])
        Training_labels.append([datapoint[1] for datapoint in ML_data[:i*s]]+[datapoint[1] for datapoint in ML_data[(i+1)*s:]])
        Testing_data.append(   [datapoint[0] for datapoint in ML_data[i*s:(i+1)*s]])
        Testing_labels.append( [datapoint[1] for datapoint in ML_data[i*s:(i+1)*s]])
elif saliency:
    s = int(np.floor(0.8*len(values)))
    for i in range(k):
        np.random.shuffle(ML_data)
        Training_data.append(  [datapoint[0] for datapoint in ML_data[:s]])
        Training_labels.append([datapoint[1] for datapoint in ML_data[:s]])
        Testing_data.append(   [datapoint[0] for datapoint in ML_data[s:]])
        Testing_labels.append( [datapoint[1] for datapoint in ML_data[s:]])
Training_data, Training_labels, Testing_data, Testing_labels = np.array(Training_data), np.array(Training_labels), np.array(Testing_data), np.array(Testing_labels)
del(i,ML_data,inputs,values)

#%% #Create, Train, & Test NN
#Define NN hyper-parameters
def act_fn(x): return keras.activations.relu(x,alpha=0.01) #...leaky-ReLU activation
number_of_epochs = 20           #...number of times to run training data through NN
size_of_batches = 200           #...number of datapoints the NN sees per iteration of optimiser (high batch means more accurate param updating, but less frequently) 
layer_sizes = [16,32,16]        #...number and size of the dense NN layers

#Define lists to record training history and learning measures
hist_data = []                              #...training data (output of .fit(), used for plotting)
cm_list, acc_list, mcc_list = [], [], []    #...lists of measures
average_gradients = []                      #...track saliency 

#Train k independent NNs for k-fold cross-validation (learning measures then averaged over)
for i in range(k):
    #Setup NN
    model = keras.Sequential()
    for layer_size in layer_sizes:
        model.add(keras.layers.Dense(layer_size, activation=act_fn))
        model.add(keras.layers.Dropout(0.05)) #...dropout layer to reduce chance of overfitting to training data
    model.add(keras.layers.Dense(number_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    
    #Train NN
    hist_data.append(model.fit(Training_data[i], Training_labels[i], batch_size=size_of_batches, epochs=number_of_epochs, shuffle=1, validation_split=0., class_weight=class_weights, verbose=1))
    
    #Test NN - Calculate learning measures: confusion matrix, accuracy, MCC
    cm = np.zeros(shape=(number_classes,number_classes)) #...initialise confusion matrix
    predictions = np.argmax(model.predict(Testing_data[i]),axis=1)
    for check_index in range(len(Testing_labels[i])):
        cm[int(Testing_labels[i][check_index]),int(predictions[check_index])] += 1      #...row is actual class, column is predicted class
    cm_list.append(cm)
    print()
    print('Normalised Confusion Matrix:') 
    print(cm/len(Testing_labels[i])) 
    print('Row => True class, Col => Predicted class')
    print()
    acc_list.append(sum([cm[n][n] for n in range(number_classes)])/len(Testing_labels[i])) #...sum of cm diagonal entries gives accuracy
    mcc_denom = np.sqrt((np.sum([np.sum([np.sum([cm[m,n] for n in range(number_classes)]) for m in list(range(number_classes)[:k])+list(range(number_classes)[k+1:])])*np.sum([cm[k,l] for l in range(number_classes)]) for k in range(number_classes)])) * (np.sum([np.sum([np.sum([cm[n,m] for n in range(number_classes)]) for m in list(range(number_classes)[:k])+list(range(number_classes)[k+1:])])*np.sum([cm[l,k] for l in range(number_classes)]) for k in range(number_classes)])))
    if mcc_denom != 0:
        mcc_list.append(np.sum([np.sum([np.sum([cm[k,k]*cm[l,m]-cm[k,l]*cm[m,k] for k in range(number_classes)]) for l in range(number_classes)]) for m in range(number_classes)]) / mcc_denom)
    else: 
        mcc_list.append(np.nan)
    
    #Track NN saliency
    if saliency:
        image = Variable(Testing_data[i],dtype='float') 
        with GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(image)
            saliency_predictions = model(image)
            loss = saliency_predictions
            
        #Compute the gradient of the loss wrt the input image (and convert to numpy)
        gradient = tape.gradient(loss, image)
        gradient = gradient.numpy()
        avg_grad = np.absolute(np.mean(gradient,axis=0))
        avg_grad = avg_grad/np.linalg.norm(avg_grad)
        average_gradients.append(avg_grad)

#Output averaged learning measures
print('####################################')
print('Average measures:')
print('Accuracy:',sum(acc_list)/k,'\pm',np.std(acc_list)/np.sqrt(k))
print('MCC:',sum(mcc_list)/k,'\pm',np.std(mcc_list)/np.sqrt(k))
print('CM:',np.mean(cm_list,axis=0).tolist())

#Compute the average saliency feature analysis
if saliency:
    #Average over the runs and plot
    average_gradients = np.array(average_gradients)
    average_gradients = np.mean(average_gradients,axis=0)
    print('Average Gradients:\n',average_gradients)
    plt.axis('off')
    plt.imshow([average_gradients],vmin=0,vmax=1)
    #plt.savefig('./SaliencyImage###.pdf',bbox_inches='tight')

#%% #Plot History Data
#Note history data is taken during training (not as good as test data statistics)
hist_label = range(1,len(hist_data)+1)

plt.figure('Accuracy History')
for i in range(len(hist_data)):
    plt.plot(range(1,number_of_epochs+1),hist_data[i].history['accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xlim(0,number_of_epochs+1)
plt.ylim(0,1)
plt.xticks(range(number_of_epochs+1))
plt.yticks(np.arange(0,1.1,0.1))
plt.legend(hist_label, loc='best')
plt.grid()
plt.show()

plt.figure('Loss History')
for i in range(len(hist_data)):
    plt.plot(range(1,number_of_epochs+1),hist_data[i].history['loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xlim(0,number_of_epochs+1)
plt.ylim(0)
plt.xticks(range(number_of_epochs+1))
#plt.yticks(np.arange(0,1.3,0.1))
plt.legend(hist_label, loc='best')
plt.grid()
plt.show()

#%% #Plot the confusion matrix
cm_list = np.array(cm_list)
cm_avg = np.mean(cm_list,axis=0)
print('CM:',cm_avg.tolist())
plt.axis('off')
plt.imshow(cm_avg/np.sum(cm_avg),vmin=0.) #vmax=1.
plt.tight_layout()
#plt.savefig('multiclassification_avg_cm.pdf')

#Normalise
cm_avg /= np.sum(cm_avg)
print(cm_avg)
#Output in a latex processible format
cmstr = str(np.around(cm_avg,3).tolist())
cmstr = cmstr.replace('[[','\\begin{pmatrix} ').replace(']]',' \\end{pmatrix}').replace('], [',' \\\\ ').replace(',',' &')
print(cmstr)

