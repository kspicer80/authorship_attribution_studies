with open(r'C:\Users\KSpicer\Documents\GitHub\cather_jewett_comparisons\testing_data\c_archbishop.txt', encoding='utf-8') as f:
    data = f.read()
    data = data.lower().split()
    
with open(r'C:\Users\KSpicer\Documents\GitHub\cather_jewett_comparisons\training_data\c_my_antonia.txt', encoding='utf-8') as f:
    antonia = f.read()
    antonia = antonia.lower().split()

with open(r'C:\Users\KSpicer\Documents\GitHub\cather_jewett_comparisons\training_data\c_o_pioneers.txt', encoding='utf-8') as f:
    pioneers = f.read()
    pioneers = pioneers.lower().split()
    
print(len(data), len(antonia), len(pioneers))
