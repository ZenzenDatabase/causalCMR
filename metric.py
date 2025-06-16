import os
import numpy as np
from scipy.spatial.distance import cdist
import scipy


def compute_map_metric(audio_emb, visual_emb, lab, similarity="COS"):
    map_1 = MAP(audio_emb, visual_emb, lab, k = 0, dist_method=similarity)
    map_2 = MAP(visual_emb, audio_emb, lab, k = 0, dist_method=similarity)
    average_map = (map_1+map_2)*0.5
#     print("Test process result:")
#     print("---The MAP of the audio-visual:\n Audio2visual = {:.2f}%; \n Visual2audio ={:.2f}%; \n Average = {:.2f}%.".format(map_1*100, map_2*100, average_map*100))
    return map_1, map_2, average_map


def MAP(embed_1, embed_2, label, k = 0, dist_method='COS'):
    if dist_method == 'L2':
        dist = scipy.spatial.distance.cdist(embed_1, embed_2, 'euclidean')
    elif dist_method == 'COS':
        dist = scipy.spatial.distance.cdist(embed_1, embed_2, 'cosine')
    ord = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
        k = numcases
    res = []
    for i in range(numcases):
        order = ord[i]
        p = 0.0
        r = 0.0
        for j in range(k):
            if label[i] == label[order[j]]:
                r += 1
                p += (r / (j + 1))
            if r > 0:
                res += [p / r]
            else:
                res += [0]
    return np.mean(res)