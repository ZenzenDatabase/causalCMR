import numpy as np
import pandas as pd

import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from causallearn.search.ScoreBased.GES import ges
from causallearn.search.PermutationBased.GRaSP import grasp

from causallearn.utils.GraphUtils import GraphUtils
import torch
import pickle

az = np.load("embeddings/prediction_500.npz")
audio_data  = az["audio"][-1564:]
visual_data = az["visual"][-1564:]
 
class_list = ['Clock', 'Motorcycle', 'Train horn', 'Bark', 'Cat', 'Bus', \
              'Rodents/rats', 'Toilet flush', 'Acoustic guitar', \
             'Frying (food)', 'Chainsaw', 'Horse', 'Helicopter', 'Infant cry',\
             'Truck']
audio_name  = ["audio_"+lab_name for lab_name in class_list]
visual_name = ["visual_"+lab_name for lab_name in class_list]

combined_array = np.column_stack((audio_data, visual_data))

df = pd.DataFrame(combined_array, columns=[audio_name+visual_name])

#grasp
Record = grasp(df, score_func="local_score_BIC")
pyd = GraphUtils.to_pydot(Record, labels=audio_name+visual_name)
tmp_png = pyd.create_png(f="png")
fp = io.BytesIO(tmp_png)
img = mpimg.imread(fp, format='png')
plt.axis('off')
plt.imshow(img)
plt.show()
pyd.write_png("grasp_500.png")

pyd = GraphUtils.to_pydot(Record, labels=audio_name + visual_name)
edges_list = [(edge.get_source(), edge.get_destination()) for edge in pyd.get_edges()]


_df = pd.DataFrame(edges_list, columns=["Cause", "Effect"])

labels = [
    'audio_Clock', 'audio_Motorcycle', 'audio_Train horn', 'audio_Bark', 'audio_Cat',
    'audio_Bus', 'audio_Rodents/rats', 'audio_Toilet flush', 'audio_Acoustic guitar', 'audio_Frying (food)',
    'audio_Chainsaw', 'audio_Horse', 'audio_Helicopter', 'audio_Infant cry', 'audio_Truck',
    'visual_Clock', 'visual_Motorcycle', 'visual_Train horn', 'visual_Bark', 'visual_Cat',
    'visual_Bus', 'visual_Rodents/rats', 'visual_Toilet flush', 'visual_Acoustic guitar', 'visual_Frying (food)',
    'visual_Chainsaw', 'visual_Horse', 'visual_Helicopter', 'visual_Infant cry', 'visual_Truck'
]

_df.replace({i: labels[i] for i in range(len(labels))}, inplace=True)
_df.to_csv("causal_500.csv")

