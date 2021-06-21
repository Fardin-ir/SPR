import numpy as np
import pandas as pd
import json
from functools import reduce

def convert(string): 
    li = list(string.split(" ")) 
    return li 

data = pd.read_json('tweets.json', lines=True)[['text','id']]
texts = data['text'].apply(convert).values
ids = data['id']
initials = pd.read_csv('initial_centroids.txt',header=None)
print(initials)
initials = initials.astype(object)

def jaccard_distance(a,b):
    return 1 - len(np.intersect1d(a,b))/len(np.union1d(a,b))

#print(jaccard_distance(text[0],text[1]))

def add_to_nearest_cluster(means,id):
    #print('dsdsds')
    x = texts[np.where(ids==id)][0]
    means_text = means[0].apply(lambda i: texts[np.where(ids==int(i))][0])
    c = np.argmin([jaccard_distance(x,mean) for mean in means_text])
    temp = np.append(means.loc[c,1],str(id))
    #print(temp)
    temp = np.delete(temp,np.where(temp=='nan'))
    means.at[c,1] = temp

#add_to_nearest_cluster(initials,ids[0])
#print(initials)

def find_centroid(means):
    for index, row in means.iterrows():
        print('cc')
        min_dis = 100000
        min_id = 0
        for id1 in row[1]:
            ave_dist = 0
            for id2 in row[1]:
                ave_dist += jaccard_distance(texts[np.where(ids==int(id1))][0],texts[np.where(ids==int(id2))][0])
            ave_dist = ave_dist/len(row[1])
            if ave_dist<min_dis:
                min_dis = ave_dist
                min_id = id1
                print(min_id)
        row[0] = min_id
        


def k_means(initials):
    means = initials.copy()
    while(True):
        print('********************************************************')
        old_meeans = means.copy()
        for i in ids:
            add_to_nearest_cluster(means,i)
        find_centroid(means)
        print(means)
        output = means.copy()
        for index, row in means.iterrows():
            row[1] = 'nan'
        if (old_meeans[0] == means[0]).all():
            break
        return output

means = k_means(initials)
print(means)
with open("output.txt", "w") as txt_file:
    for i in range(len(means[0])):
        txt_file.write(str(means.iloc[i,0]) + ":")     
        txt_file.write(str((means.iloc[i,1])) + "\n")    
        txt_file.write("********************************************\n\n")    
