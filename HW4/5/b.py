import numpy as np
import pandas as pd
import json
from functools import reduce
np.set_printoptions(suppress=False)

def convert(string): 
    li = list(string.split(" ")) 
    return li 

data = pd.read_json('tweets.json', lines=True)[['text','id']]
texts = data['text'].apply(convert).values
ids = data['id']


def jaccard_distance(a,b):
    return 1 - len(np.intersect1d(a,b))/len(np.union1d(a,b))


def find_next_initial(temp_id,means,output):
    print('cc')
    max_dis = 0
    min_id = 0
    for id in temp_id:
        ave_dist = 0
        for mean in means:
            ave_dist += jaccard_distance(texts[np.where(ids==int(id))][0],texts[np.where(ids==mean)][0])
        ave_dist = ave_dist/len(means)
        if ave_dist>max_dis:
            min_dis = ave_dist
            min_id = id
    output=np.append(output,min_id)
    return output

output = np.array([])
temp_id = ids.copy()
output=find_next_initial(temp_id,temp_id,output)
print(output)
for k in range(24):
    temp_id = ids.copy()
    output=find_next_initial(np.setdiff1d(temp_id,output),output,output)

with open("part_b.txt", "w") as txt_file:
    for line in output:
        txt_file.write(str(int(line)) + ",\n")      



