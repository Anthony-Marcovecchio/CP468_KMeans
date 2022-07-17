import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.spatial import distance

# convert to dataframe
df = pd.read_csv('/Users/anthonymarcovecchio/Downloads/kmeans.csv')
print(df.to_string())

# PART A
df.plot.scatter(x='f1',y='f2')

# PART B
def kmeans(k, df):
    
    # initialize centroids and cluster label lists
    centroids = [[0 for i in range(2)] for j in range(k)]
    clust_label = [0 for i in range(len(df))]
    
    # choose first k data points as initial centroids
    for i in range(k):
        centroids[i][0] = df.iat[i,0]
        centroids[i][1] = df.iat[i,1]
        
    # assign each observation to closest centroid
    for obs in range(len(df)):
        min_dist = sys.maxsize
        for cent in range(len(centroids)):
            obs_points = df.iloc[obs].values.tolist()
            curr_dist = distance.euclidean(obs_points, centroids[cent])
            # update cluster label to new closest centroid
            if curr_dist < min_dist:
                min_dist = curr_dist
                clust_label[obs] = cent
                
    # compute new centroids
    new_centroids = [[0 for i in range(2)] for j in range(k)]
    for obs in range(len(df)):
        curr_obs = df.iloc[obs].values.tolist()
        curr_clust = clust_label[obs]
        new_centroids[curr_clust][0] += curr_obs[0]
        new_centroids[curr_clust][1] += curr_obs[1]
        
    # compute averages for new_centroids
    for i in range(len(new_centroids)):
        new_centroids[i][0] = round(new_centroids[i][0]/(clust_label.count((i))), 2)
        new_centroids[i][1] = round(new_centroids[i][1]/(clust_label.count((i))), 2)
        
    # repeat until centroid averages don't change
    while new_centroids != centroids:
        centroids = new_centroids
        
        # assign each observation to closest centroid
        for obs in range(len(df)):
            min_dist = sys.maxsize
            for cent in range(len(centroids)):
                obs_points = df.iloc[obs].values.tolist()
                curr_dist = distance.euclidean(obs_points, centroids[cent])
                # update cluster label to new closest centroid
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    clust_label[obs] = cent

        # compute new centroids
        new_centroids = [[0 for i in range(2)] for j in range(k)]
        for obs in range(len(df)):
            curr_obs = df.iloc[obs].values.tolist()
            curr_clust = clust_label[obs]
            new_centroids[curr_clust][0] += curr_obs[0]
            new_centroids[curr_clust][1] += curr_obs[1]

        # compute averages for new_centroids
        for i in range(len(new_centroids)):
            new_centroids[i][0] = round(new_centroids[i][0]/(clust_label.count((i))), 2)
            new_centroids[i][1] = round(new_centroids[i][1]/(clust_label.count((i))), 2)
            
    return clust_label, centroids

k = 2
clust_label, centroids = kmeans(k, df)
print(clust_label)


# PART C
# append label column to dataframe
df2 = df.assign(label=clust_label)
# scatterplot with clusters coloured coded
plt.scatter(df2.f1, df2.f2, c=df2.label, alpha = 0.6, s=25)


# PART D/E
clust1 = clust_label.count(0)
clust2 = clust_label.count(1)
print(f'Size of purple cluster: {clust1}')
print(f'Size of yellow cluster: {clust2}')
print()
print(f'Final centroids: {centroids}')
