import pandas as pd
import gzip
import pickle

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

df = getDF('reviews_Electronics_5.json.gz')


texts = df['reviewText'].tolist()
labels = df['overall'].tolist()



with open("./raw_data/texts.pkl", 'wb') as handle:
    pickle.dump(texts, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("./raw_data/label.pkl", 'wb') as handle:
    pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)