import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_theme(style="whitegrid")

# load data from 'data/datasizes/run_i.json' where i is the run number

data = []
go = True
i = 0
while go:
    try:
        with open(f"data/datasizes/run_{i}.json", 'r') as f:
            run_data = json.load(f)
            data.append(run_data)
            i += 1
    except:
        go = False


columns = ['data_sizes', 'f1_scores', 'model']
df = pd.DataFrame(data, columns=columns)
df = df.set_index(['model']).apply(pd.Series.explode).reset_index()

#plot data using seaborn


plt.figure(1, figsize=(10, 10))
sns.lineplot(x='data_sizes',
             y='f1_scores',
             ci=100,
             hue='model',
             data=df)
#only plot x values that are in the range of 100-1400
plt.xlim(100, 1400)
plt.ylim(0.6, 1)
plt.title(r'F1 score for different data sizes')
plt.xlabel(r'Data size')
plt.ylabel(r'F1 score')
plt.tight_layout()
plt.savefig('data/plots/datasizes.pdf', format='pdf')
plt.show()