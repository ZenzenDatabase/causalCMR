import pandas as pd
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_causal_weight(image_onehot, text_onehot, Causality_A_image, Causality_A_text, Causality_A_cross_1, Causality_A_cross_2):
    causal_weight = 1.0
    
    image_label = torch.argmax(image_onehot).item()
    text_label  = torch.argmax(text_onehot).item()
    
    # Audio-to-audio causality
    if Causality_A_image.iloc[image_label, image_label] == 1:
        causal_weight *= 0.1    
    
    # Visual-to-visual causality
    if Causality_A_text.iloc[text_label, text_label] == 1:
        causal_weight *= 0.1    
    
    # Audio-to-visual causality
    if Causality_A_cross_1.iloc[image_label, text_label] == 1:
        causal_weight *= 0.4  
    
    # Visual-to-audio causality
    if Causality_A_cross_2.iloc[text_label, image_label] == 1:
        causal_weight *= 0.4  

    return causal_weight

def pairwise_distance(x, y):
    return torch.norm(x - y, p=2)

def causal_loss_func(image_feats, text_feats, audio_onehots, visual_onehots, 
                         Causality_A_image, Causality_A_text, Causality_A_cross_1, Causality_A_cross_2):
    total_causal_loss = 0
    
    for i in range(image_feats.size(0)):
        weight = compute_causal_weight(audio_onehots[i], visual_onehots[i], 
                                       Causality_A_image, Causality_A_text, 
                                       Causality_A_cross_1, Causality_A_cross_2)
        
        dist = pairwise_distance(image_feats[i], text_feats[i])
        total_causal_loss += dist * weight  
        
    return total_causal_loss


def causal_loss(view1_predict, view2_predict, labels):
    class_list = ['Clock', 'Motorcycle', 'Train horn', 'Bark', 'Cat', 'Bus', 
                  'Rodents/rats', 'Toilet flush', 'Acoustic guitar', 
                  'Frying (food)', 'Chainsaw', 'Horse', 'Helicopter', 'Infant cry',
                  'Truck']

    audio_name  = ["audio_" + lab_name for lab_name in class_list]
    visual_name = ["visual_" + lab_name for lab_name in class_list]

    Causality_A_image   = pd.DataFrame(np.zeros((len(audio_name), len(audio_name))), 
                                     index=audio_name, columns=audio_name)
    Causality_A_text    = pd.DataFrame(np.zeros((len(visual_name), len(visual_name))), 
                                     index=visual_name, columns=visual_name)
    Causality_A_cross_1 = pd.DataFrame(np.zeros((len(audio_name), len(visual_name))), 
                                       index=audio_name, columns=visual_name)
    Causality_A_cross_2 = pd.DataFrame(np.zeros((len(visual_name), len(audio_name))), 
                                       index=visual_name, columns=audio_name)

    df = pd.read_csv("../causal_500.csv")

    
    for _, row in df.iterrows():
        cause, effect = row['Cause'], row['Effect']
    
        # Audio-to-audio causality
        if cause.startswith('audio_') and effect.startswith('audio_'):
            Causality_A_image.at[cause, effect] = 1

        # Visual-to-visual causality
        elif cause.startswith('visual_') and effect.startswith('visual_'):
            Causality_A_text.at[cause, effect] = 1

        # Audio-to-visual causality
        elif cause.startswith('audio_') and effect.startswith('visual_'):
            Causality_A_cross_1.at[cause, effect] = 1

        # Visual-to-audio causality
        elif cause.startswith('visual_') and effect.startswith('audio_'):
            Causality_A_cross_2.at[cause, effect] = 1
        
    causal_loss_val = causal_loss_func(view1_predict, view2_predict, labels, labels, 
                                   Causality_A_image, Causality_A_text, 
                                   Causality_A_cross_1, Causality_A_cross_2)
    return causal_loss_val

def causality_matrix_gen(margin):
    _df = pd.read_csv("../grasp_causal_baseline.csv")
    class_list = ['Clock', 'Motorcycle', 'Train horn', 'Bark', 'Cat', 'Bus', 
                  'Rodents/rats', 'Toilet flush', 'Acoustic guitar', 
                  'Frying (food)', 'Chainsaw', 'Horse', 'Helicopter', 'Infant cry', 'Truck']

    audio_name  = ["audio_" + lab_name for lab_name in class_list]
    visual_name = ["visual_" + lab_name for lab_name in class_list]

    label_to_index = {label: i for i, label in enumerate(class_list)}

    causality_matrix = np.full((15, 15), margin)

    for _, row in _df.iterrows():
        cause, effect = row['Cause'], row['Effect']
        
        if cause.startswith('audio_') and effect.startswith('visual_'):
            cause_label  = cause.split("_", 1)[1]
            effect_label = effect.split("_", 1)[1]

            if cause_label in label_to_index and effect_label in label_to_index:
                i = label_to_index[cause_label]  # Row index
                j = label_to_index[effect_label]  # Column index
                causality_matrix[i, j] = 0.5  # Set causality value

    causality_df = pd.DataFrame(causality_matrix, index=class_list, columns=class_list)
    return causality_df
