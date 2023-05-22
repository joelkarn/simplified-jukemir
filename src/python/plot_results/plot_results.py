import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_theme(style="whitegrid")

# load data from 'data/datasizes/run_i.json' where i is the run number


data = []
go = True
i = 0
data = []
while go:
    try:
        with open(f"data/svm_datasizes/auc/inclusive-exclusive/run_{i}.json", 'r') as f:
            model_data = json.load(f)
            for model_data_point in model_data['data']:
                #create a datapoint with the model name
                data_point = {
                    'data_size': model_data_point['n_samples'],
                    'auc_roc_score': model_data_point['auc_roc_score'],
                    'model': model_data['model']}

                data.append(data_point)
            i += 1
    except:
        go = False

columns = ['data_size', 'auc_roc_score', 'model']
df = pd.DataFrame(data, columns=columns)
print(df)
df = df.set_index(['model']).apply(pd.Series.explode).reset_index()

# #plot data using seaborn

plt.figure(1, figsize=(10, 10))
sns.lineplot(x='data_size',
             y='auc_roc_score',
             ci=100,
             hue='model',
             data=df)
#only plot x values that are in the range of 100-1400
#plt.xlim(100, 587)
#plt.ylim(0.63, .88)
plt.title(r'Inclusive vs. Exclusive')
plt.xlabel(r'Data size')
plt.ylabel(r'AUC')
plt.tight_layout()
plt.savefig('data/plots/svm_datasizes_3.pdf', format='pdf')
plt.show()