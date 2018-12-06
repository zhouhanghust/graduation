import pickle
from Gra_En.doExperiment.cleanAndClear import clean_text


with open("./raw_data/texts.pkl","rb") as f:
    texts = pickle.load(f)

with open("./raw_data/label.pkl","rb") as f:
    label = pickle.load(f)

label = list(map(lambda x:int(x>3),label))
texts = list(map(lambda x:clean_text(x),texts))


with open("./cuted_unbalanced_data/texts.pkl", 'wb') as handle:
    pickle.dump(texts, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("./cuted_unbalanced_data/label.pkl", 'wb') as handle:
    pickle.dump(label, handle, protocol=pickle.HIGHEST_PROTOCOL)








