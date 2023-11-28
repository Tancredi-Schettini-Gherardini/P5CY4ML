'''Script to PCA the WPS5 weight data -- reproducing section 3.3 of the paper'''
#Import libraries
import numpy as np
import matplotlib.pyplot as plt

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
    
datasizes = list(map(len,all_data))
print(f'Datasizes: {datasizes}')
#CnIPnD, DnIP, IPnRnD, IPRnD, DnR, DR, CYnR, CYR = all_data
del(file,filename,line,weights)

#%% #Perform PCA on all the data
#Flatten data
all_data_flat = np.transpose(np.array([ww for dataset in all_data for ww in dataset]))

#Centering --> unnnecessary as covariance already relative to mean (explicitly checked the covariance matrices are the same)
#all_data_flat = all_data_flat - np.mean(all_data_flat,axis=1).reshape(-1,1) 

#Compute eigendecomposition
covariance = np.cov(all_data_flat)
eig_values, eig_vectors = np.linalg.eigh(covariance) #...as covariance manifestly symmetric use the faster 'eigh' routine

#Sort eigenspectrum according to decreasing eigenvalues
eidx_sort = np.argsort(eig_values)[::-1]
eig_values = eig_values[eidx_sort]
eig_vectors = eig_vectors[:,eidx_sort]
explained_variance = eig_values/np.linalg.norm(eig_values)
print(f'Explained_variance: {explained_variance}')

#Project onto first component (significantly dominant)
projection = np.matmul(eig_vectors[0],all_data_flat)
projection_partitions = [projection[sum(datasizes[:i]):sum(datasizes[:i+1])] for i in range(len(datasizes))]
projection_partition_means = np.array([np.mean(dataset) for dataset in projection_partitions])
projection_partition_stds = np.array([np.std(dataset) for dataset in projection_partitions])

#%% #Histogram the 1d projections for each partition
plt.figure('Histogram')
for dataset_idx in range(len(projection_partitions)):
    rounded_data = np.round(projection_partitions[dataset_idx])
    x, y = np.unique(rounded_data,return_counts=True)
    print(min(x),max(x))
    plt.scatter(x,y,label=str(partition_filenames[dataset_idx][:-4]),alpha=0.2,zorder=10)
plt.xlabel('PCA 1d Projection')
plt.ylabel('Frequency')
plt.xscale('log')
plt.yscale('log')
leg=plt.legend(loc='best')
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.grid()
plt.tight_layout()
#plt.savefig('PCA_histogramLL.png')
#plt.savefig('PCA_histogramLL.pdf')
del(rounded_data,x,y,lh)

#%% #Generic PCA projections of each dataset individually
dataset_idx = 7
projection_size = 2

#Compute eigendecomposition
covariance = np.cov(np.transpose(all_data[dataset_idx]))
eig_values, eig_vectors = np.linalg.eigh(covariance) #...as covariance manifestly symmetric use the faster 'eigh' routine

#Sort eigenspectrum according to decreasing eigenvalues
eidx_sort = np.argsort(eig_values)[::-1]
eig_values = eig_values[eidx_sort]
eig_vectors = eig_vectors[:,eidx_sort]
explained_variance = eig_values/np.linalg.norm(eig_values)
print(f'Data index: {dataset_idx}\nExplained_variance: {explained_variance}')

#Project onto selected component 
projection = np.matmul(eig_vectors[:projection_size],np.transpose(all_data[dataset_idx]))

#Plot the 2d PCA
plt.figure('2d PCA')
plt.scatter(projection[0,:],projection[1,:],alpha=1)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid()
plt.tight_layout()
#plt.savefig(f'2dPCA_{dataset_idx}.png')
#plt.savefig(f'2dPCA_{dataset_idx}.pdf')

#%% #Generic PCA projections of both CY datasets simultaneously
projection_size = 2
CY_data = np.transpose(np.array([ww for ww in all_data[-2]]+[ww for ww in all_data[-1]]))
#...with centering
#CY_data = CY_data - np.mean(CY_data,axis=1).reshape(-1,1) 

#Compute eigendecomposition
covariance = np.cov(CY_data)
eig_values, eig_vectors = np.linalg.eigh(covariance) #...as covariance manifestly symmetric use the faster 'eigh' routine

#Sort eigenspectrum according to decreasing eigenvalues
eidx_sort = np.argsort(eig_values)[::-1]
eig_values = eig_values[eidx_sort]
eig_vectors = eig_vectors[:,eidx_sort]
explained_variance = eig_values/np.linalg.norm(eig_values)
print(f'Explained_variance: {explained_variance}')

#Project onto selected component 
CYnR_projection = np.matmul(eig_vectors[:projection_size],np.transpose(all_data[-2]))
CYR_projection  = np.matmul(eig_vectors[:projection_size],np.transpose(all_data[-1]))
#...with centering
#CYnR_projection = np.matmul(eig_vectors[:projection_size],CY_data[:,:len(all_data[-2])])
#CYR_projection  = np.matmul(eig_vectors[:projection_size],CY_data[:,len(all_data[-2]):])

#Plot the 2d PCA
plt.figure('2d PCA')
plt.scatter(CYR_projection[0,:],CYR_projection[1,:],label='CYR',c='orangered',alpha=1,zorder=10)
plt.scatter(CYnR_projection[0,:],CYnR_projection[1,:],label='CYnR',c='steelblue',alpha=0.1,zorder=10)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
leg=plt.legend(loc='best')
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.grid()
plt.tight_layout()
#plt.savefig('2dPCA_CY.png')
#plt.savefig('2dPCA_CY.pdf')
