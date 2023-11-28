"""
Import the four-folds dataset, produce plots to show the interesting features 
of the invariants and perform a clustering analysis.
The code below reproduces section 3.1 of the paper.
"""

#Import libraries
import numpy as np
import gzip
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter

#Define path to data
path = './Data/5dTransWH.all.gz'

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
#   We plot the histograms to show the distributions below.

#%% h11 Histogram

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.5)
plt.ticklabel_format(axis='both', style='sci', scilimits=(5,5))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.hist(h11, bins=500,log=True, zorder = 10)
plt.xlabel(r'$h^{1,1}$', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)
#plt.savefig('h11_hist_.jpg', dpi=1200, bbox_inches='tight')

print("Maximumx is ", max(h11))
print("Minimum is ", min(h11))
print("Mean is ", sum(h11)/len(h11))

#%% h12 Histogram

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.5)
plt.ticklabel_format(axis='both', style='sci', scilimits=(3,3))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.hist(h12, bins=500,log=True, zorder = 10)
plt.xlabel(r'$h^{1,2}$', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)
#plt.savefig('h12_hist_.jpg', dpi=1200, bbox_inches='tight')

print("Maximumx is ", max(h12))
print("Minimum is ", min(h12))
print("Mean is ", sum(h12)/len(h12))

#%% h13 Histogram

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.5)
plt.ticklabel_format(axis='both', style='sci', scilimits=(5,5))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.hist(h13, bins=500,log=True, zorder = 10)
plt.xlabel(r'$h^{1,3}$', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)
#plt.savefig('h13_hist_.jpg', dpi=1200, bbox_inches='tight')

print("Maximumx is ", max(h13))
print("Minimum is ", min(h13))
print("Mean is ", sum(h13)/len(h13))

#%% h22 Histogram

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.5)
plt.ticklabel_format(axis='both', style='sci', scilimits=(6,6))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.hist(h22, bins=500,log=True, zorder = 10)
plt.xlabel(r'$h^{2,2}$', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)
#plt.savefig('h22_hist_.jpg', dpi=1200, bbox_inches='tight')

print("Maximumx is ", max(h22))
print("Minimum is ", min(h22))
print("Mean is ", sum(h22.astype(float))/len(h22))

#%% Euler Histogram

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.5)
plt.ticklabel_format(axis='both', style='sci', scilimits=(6,6))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.hist(chi, bins=500,log=True, zorder = 10)
plt.xlabel(r'$\chi$', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)
#plt.savefig('chi_hist_.jpg', dpi=1200, bbox_inches='tight')

print("Maximumx is ", max(chi))
print("Minimum is ", min(chi))
print("Mean is ", sum(chi.astype(float))/len(chi))


#%% --------------------------------------------------------------------------------------
#   We produce scatter plots for some of the Hodge data below.

#%% Classic Mirror Plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.5)
plt.ticklabel_format(axis='both', style='sci', scilimits=(5,5))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.scatter(h11-h13, h11+ h13, zorder = 10, s = 0.2)

x = np.arange(0, 3.15*10**5, 0.1*10**5 )
plt.plot(x, x , linewidth = 5, color="orangered", alpha = 0.5, label = r"$h^{1,3} = 0$")


x = np.arange(-3.15*10**5, 0, 0.1*10**5 )
plt.plot(x, -x , linewidth = 5, color="orangered", alpha = 0.5, label = r"$h^{1,1} = 0$")


plt.xlabel(r'$h^{1,1} - h^{1,3}$', fontsize = 15)
plt.ylabel(r'$h^{1,1} + h^{1,3}$', fontsize = 15)

#plt.savefig('Mirror_plot.jpg', dpi=1200, bbox_inches = "tight")

#%% Two highest Hodges plot

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.5)
plt.ticklabel_format(axis='both', style='sci', scilimits=(5,5))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.scatter(h13, h22, zorder = 10, s = 0.3)
x = np.arange(0, 3.15*10**5, 0.1*10**5 )
plt.plot(x, 4*x + 82, linewidth = 3, color="orangered", alpha = 0.5, label = r"$h^{2,2} = 4 \, h^{1,3}$")

plt.xlabel(r'$h^{1,3}$', fontsize = 15)
plt.ylabel(r'$h^{2,2} $', fontsize = 15)

plt.legend(loc="center right")   

#plt.savefig('Mirror_plot_2.jpg', dpi=1200, bbox_inches = "tight")

#%% 3-d Hodges plot

fig = plt.figure(figsize=(9, 6))
ax = plt.axes(projection='3d')

ax.scatter3D(h11, h12, h13, color='steelblue', s=0.4)
ax.ticklabel_format(axis='x', style='sci', scilimits=(5,5))
plt.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
plt.ticklabel_format(axis='z', style='sci', scilimits=(5,5))
ax.view_init(elev=17, azim = 65)
ax.set_xlabel(r"$h^{1,1}$ (1e5) ", fontsize = 12) 
ax.set_ylabel(r"$h^{1,2}$ (1e3)", fontsize = 12) 
ax.set_zlabel(r"$h^{1,3}$ (1e5)", fontsize = 12)

#plt.savefig('3dplot4_.jpg', dpi=1000, bbox_inches = "tight")

#%% h^{1,1} vs highest weight plot 

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.5)
plt.ticklabel_format(axis='both', style='sci', scilimits=(5,5))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel(r"$h^{1,1}$", fontsize = 15)
plt.xlabel(r"$w_{max}$", fontsize = 15)
#plt.scatter(weights[:,5],h11[:], c="steelblue", s=0.2)


#%% --------------------------------------------------------------------------------------
#   We now focus on the clustering behaviour, which is evident from the scatter
#   plot above.

#%% Perform clustering analysis on the data for which the linear forking is most
#   evident, i.e. the systems with large weights.

h11_clust, high_weight_clust = [], []
for i in range(len(h11)):
    # This is just a way of selecting those points in the h^{1,1}-w_{max} plane
    # that nicely show linear clustering. We identified the middle cluster, and
    # used the line perpendicular to it as a lower bound to select the data.
    # Simply choosing all the points with w_{max}>3*10^5 would also be an option.
    if h11[i] + (1/0.1925)*weights[i,5] > 20*10**5: 
        
        h11_clust.append(h11[i])
        high_weight_clust.append(weights[i,5])
        
# Get them into the right format
h11_clust = np.array(h11_clust)
high_weight_clust = np.array(high_weight_clust)

preset_number_clusters=8        # Chose this just by inspection.
ratio_data=np.array(h11_clust/high_weight_clust).reshape(-1,1)
kmeans = KMeans(n_clusters=preset_number_clusters).fit(ratio_data)


#Compute clustering over the full ratio data (irrespective of whether full or outer used to identify clusters)
transformed_full_data = kmeans.transform(ratio_data)                  #...data transformed to list distance to all centres
kmeans_labels = np.argmin(transformed_full_data,axis=1)                   #...identify the closest cluster centre to each datapoint
full_data_inertia = np.sum([min(x)**2 for x in transformed_full_data])    #...compute the inertia over the full dataset
cluster_sizes = Counter(kmeans_labels)                                    #...compute the frequencies in each cluster
print('\nCluster Centres: '+str(kmeans.cluster_centers_.flatten())+'\nCluster sizes: '+str([cluster_sizes[x] for x in range(10)])+'\n\nInertia: '+str(full_data_inertia)+'\nNormalised Inertia: '+str(full_data_inertia/len(h11_clust))+'\nNormalised Inertia / range: '+str((full_data_inertia/(len(h11_clust)*(max(ratio_data)-min(ratio_data))))[0]))

#%% Visualise the data that we have selected for the investigation as a histogram.

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

inner=500000
plt.hist(h11_clust/high_weight_clust,bins=50, zorder = 10, color= "steelblue")

plt.xlabel(r"$h^{1,1}/ w_{max}$", fontsize = 15)
plt.ylabel("Frequency", fontsize = 15)

#plt.savefig('clust_hist.jpg', dpi=1000, bbox_inches = "tight")

#%% Visualise the data and the clusters that were found above on the same plot.

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.xlim(-100000,2000000)
plt.ylim(-10000,400000)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(alpha=0.5)

plt.scatter(high_weight_clust,h11_clust, c='steelblue', s=0.1, zorder = 10, label="Data")
plt.plot(np.linspace(0,2000000,2),kmeans.cluster_centers_.flatten()[0]*np.linspace(0,2000000,2),color='black',lw=0.5, alpha=0.5, zorder = 15, label="Cluster lines")
for grad in kmeans.cluster_centers_.flatten():
    plt.plot(np.linspace(0,2000000,2),grad*np.linspace(0,2000000,2),color='black',lw=0.5, alpha=0.5, zorder = 15)


plt.ylabel(r'$h^{1,1}$', fontsize = 15)
plt.xlabel(r'$w_{max}$', fontsize = 15 )

plt.legend(loc="upper left", fontsize = 11.5)

#plt.savefig('clust_lines_overlaid.jpg', dpi=1000, bbox_inches = "tight")
