import pandas as pd
import numpy as np

def dist(p0, p1):
# Description: Calculate the euclidian distance between [p0] and [p1].
    return ((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)**0.5

def kmean(data, K):
# Description: Clustering the [data] into [K] clusters using K-mean clustering.
    
    data_out = data.copy()

    # -- |Centroid Initialization| --
    centroid_x = min(data_out['x']) + np.random.rand(1,K)*(max(data_out['x']) - min(data_out['x']))
    centroid_y = min(data_out['y']) + np.random.rand(1,K)*(max(data_out['y']) - min(data_out['y']))
    centroid = pd.DataFrame({'x':centroid_x[0], 'y':centroid_y[0]})

    while True:
        # -- |Assign Sample points to Centroid| --
        for sp, i in zip(zip(data_out['x'],data_out['y']),range(data_out.shape[0])):
            sp_ct_dist = []
            # Calculate distance between sample point and centroids
            for ct in zip(centroid['x'],centroid['y']):
                sp_ct_dist.append(dist(sp,ct))
            # Assign sample point to the same label with the closest centroid
            data_out.at[i, 'labels'] = np.argmin(sp_ct_dist)
        
        # -- |Centroid Re-Initialization if some of cluster have no members| --
        while len(set(data_out['labels'])) != K:
            # -- |Assign Sample points to Centroid| --
            centroid_x = min(data_out['x']) + np.random.rand(1,K)*(max(data_out['x']) - min(data_out['x']))
            centroid_y = min(data_out['y']) + np.random.rand(1,K)*(max(data_out['y']) - min(data_out['y']))
            centroid = pd.DataFrame({'x':centroid_x[0], 'y':centroid_y[0]})

            # -- |Assign Sample points to Centroid| --
            for sp, i in zip(zip(data_out['x'],data_out['y']),range(data_out.shape[0])):
                sp_ct_dist = []
                # Calculate distance between sample point and centroids
                for ct in zip(centroid['x'],centroid['y']):
                    sp_ct_dist.append(dist(sp,ct))
                # Assign sample point to the same label with the closest centroid
                data_out.at[i, 'labels'] = np.argmin(sp_ct_dist)

        # -- |Assure that the labels will be integer| --
        data_out['labels'] = data_out['labels'].astype(int)

        # -- |Keep the pre-update centroid| --
        centroid_last = centroid.copy()

        # -- |Centroid Update| --
        for ct in range(K):
            centroid.at[ct, 'x'] = np.sum(data_out[data_out['labels'] == ct]['x'])/len(data_out[data_out['labels'] == ct]['x'])
            centroid.at[ct, 'y'] = np.sum(data_out[data_out['labels'] == ct]['y'])/len(data_out[data_out['labels'] == ct]['y'])

        # -- |Check if the centroid changed| --
        if centroid.compare(centroid_last).empty:
            return data_out, centroid # if the centroids don't changed, break the assign and update loop
        
def WSS(labeled_data, centroid):
# Description: Calculate within cluster sums of squares of given [labeled_data] and it's [centroid]s.
    wss = 0
    for ct in range(len(centroid['x'])):
        for sp in zip(labeled_data[labeled_data['labels'] == ct]['x'],labeled_data[labeled_data['labels'] == ct]['y']):
            wss = wss + dist(sp,(centroid['x'].iloc[ct],centroid['y'].iloc[ct]))**2
    return wss

def BSS(labeled_data, centroid):
# Description: Calculate between cluster sums of squares of given [labeled_data] and it's [centroid]s.
    bss = 0

    # -- |Average of all data calculation| --
    mean_x = 0
    mean_y = 0
    for sp in zip(labeled_data['x'],labeled_data['y']):
        mean_x = mean_x + sp[0]
        mean_y = mean_y + sp[1]
    mean_x = mean_x/len(labeled_data['x'])
    mean_y = mean_y/len(labeled_data['y'])

    # -- |BSS Calculation| --
    for ct in range(len(centroid['x'])):
        bss = bss + len(labeled_data[labeled_data['labels'] == ct]['x'])*dist((centroid['x'].iloc[ct],centroid['y'].iloc[ct]), (mean_x,mean_y))**2
    return bss

def silhouette(labeled_data, centroid):
# Description: Calculate silhouette coefficient of given [labeled_data] and it's [centroid]s. 
    return (BSS(labeled_data, centroid) - WSS(labeled_data, centroid))/max(BSS(labeled_data, centroid),WSS(labeled_data, centroid))

def FVE(labeled_data, centroid):
# Description: Calculate fraction of explained variance of given [labeled_data] and it's [centroid]s.    
     
    # -- |Average of all data calculation| --
    mean_x = 0
    mean_y = 0
    for sp in zip(labeled_data['x'],labeled_data['y']):
        mean_x = mean_x + sp[0]
        mean_y = mean_y + sp[1]
    mean_x = mean_x/len(labeled_data['x'])
    mean_y = mean_y/len(labeled_data['y'])

    # -- |All-data variance| --
    adv = 0
    for sp in zip(labeled_data['x'],labeled_data['y']):
        adv = adv + dist(sp, (mean_x,mean_y))**2

    # -- |BSS Calculation| --
    bss = 0
    for ct in range(len(centroid['x'])):
        bss = bss + len(labeled_data[labeled_data['labels'] == ct]['x'])*dist((centroid['x'].iloc[ct],centroid['y'].iloc[ct]), (mean_x,mean_y))**2
    
    return bss/adv