# -*- coding: utf-8 -*-
import pickle

with open('../cutted_umbalanced_data/cutted_umbalanced_data.pkl', 'rb') as f:
    df = pickle.load(f)

print(df[df['score']==0])

