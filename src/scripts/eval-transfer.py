import itertools
import os
import pandas as pd
from os.path import join

job_string_base = '''
#!/bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -l gpu 

source activate transformer
'''

transfer_datasets = ["SICK-E", "AddOneRTE", "DPR", "SPRL", "FNPLUS", "JOCI", "MPE", "SciTail",\
    "GLUEDiagnostic", "QQP", "MNLIMatched", "MNLIMismatched", "SNLI", "SNLIHard"]

results_dir = 'paper_results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# we need to find the best transfer dev results and show the results on the test set.
# method1
dev_results = {}
test_results = {}
for task in transfer_datasets:
    path = join(results_dir, task+"_m1.csv")
    df = pd.read_csv(path, sep=";")
    dev_results[task] = df.loc[df['dev'].idxmax(), 'dev']
    test_results[task] = df.loc[df['dev'].idxmax(), 'test']

print("Method 1")
print("Dev results")
print(dev_results)
print("Test results")
print(test_results)

# method2
dev_results = {}
test_results = {}
for task in transfer_datasets:
    path = join(results_dir, task+"_m2.csv")
    df = pd.read_csv(path, sep=";")
    dev_results[task] = df.loc[df['dev'].idxmax(), 'dev']
    test_results[task] = df.loc[df['dev'].idxmax(), 'test']

print("Method 2")
print("Dev results")
print(dev_results)
print("Test results")
print(test_results)
