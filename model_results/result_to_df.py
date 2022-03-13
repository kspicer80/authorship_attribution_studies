import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv('results.txt', header=None, sep=',')
#print(results.head())
results.rename(columns={0: 'prediction', 1: 'text'}, inplace=True)
#print(results.head())
results['test'] = results['prediction'].str.extract(r'((?<=\[).*?(?=\]))')
results['cleaned_prediction'] = results['test'].astype(float)
#print(results.head())
#print(results.dtypes)
#print(results.shape)
cather_df = results.loc[results.cleaned_prediction <= 0.5]
jewett_df = results.loc[results.cleaned_prediction >= 0.5]
print(cather_df.shape)
print(jewett_df.shape)
print((cather_df.shape[0])/(results.shape[0]))
print((jewett_df.shape[0])/(results.shape[0]))

predictions = results['cleaned_prediction']
plt.plot(predictions)
plt.show()
