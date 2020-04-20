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

# All parameters that need to be set. 
data_to_path = {
    'SNLI': "/idiap/temp/rkarimi/resources/SNLI",
    'SNLIHard': "/idiap/temp/rkarimi/datasets/SNLIHard/",
    'MNLIMatched': "/idiap/temp/rkarimi/datasets/MNLIMatched/", 
    'MNLIMismatched': "/idiap/temp/rkarimi/datasets/MNLIMismatched/", 
    'JOCI': "/idiap/temp/rkarimi/datasets/JOCI",
    'SICK-E': "/idiap/temp/rkarimi/resources/SICK-E",
    'AddOneRTE': "/idiap/temp/rkarimi/datasets/AddOneRTE",
    'DPR': "/idiap/temp/rkarimi/datasets/DPR", 
    'FNPLUS': "/idiap/temp/rkarimi/datasets/FNPLUS",
    'SciTail': "/idiap/temp/rkarimi/datasets/SciTail", 
    'SPRL': "/idiap/temp/rkarimi/datasets/SPRL",
    'MPE': "/idiap/temp/rkarimi/datasets/MPE", 
    'QQP': "/idiap/temp/rkarimi/datasets/QQP",
    'GLUEDiagnostic':  "/idiap/temp/rkarimi/datasets/GLUEDiagnostic"
}
results_dir = 'paper_results'
outpath_base_m1 = '/idiap/temp/rkarimi/robust-nli/method1/'
outpath_base_m2 = '/idiap/temp/rkarimi/robust-nli/method2/'
glove_path = "/idiap/temp/rkarimi/resources/GloVe/glove.840B.300d.txt"



def submit_job(curr_job, filename, outpath_base):
    job_name = "template.job"
    with open(job_name, "w") as f:
       f.write(curr_job)
    os.system("qsub -V -N {0} -e {1}.err -o {1}.out template.job".format(filename, os.path.join(outpath_base, filename)))


transfer_datasets = ["SICK-E", "AddOneRTE", "DPR", "SPRL", "FNPLUS", "JOCI", "MPE", "SciTail",\
    "GLUEDiagnostic", "QQP", "MNLIMatched", "MNLIMismatched", "SNLI", "SNLIHard"]

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# computing the transfer performance for all datasets for all trained models in method 1.
if not os.path.exists(outpath_base_m1):
    os.makedirs(outpath_base_m1)
job_string = job_string_base + '''python eval.py --embdfile {8} --model {0}/alpha_{1}_beta_{2}/model.pickle  --test_path {3}  --outputfile {4} --train_path{5} --test_data {6} --train_data {7}'''
alphas = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
betas = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
for task in transfer_datasets:
    for alpha in alphas:
        for beta in betas:
            curr_job = job_string.format(outpath_base_m1, alpha, beta, data_to_path[task],\
                 join(results_dir, task+"_m1.csv"), data_to_path["SNLI"], task, "SNLI", glove_path)
            submit_job(curr_job, "eval"+str(alpha)+str(beta)+task, outpath_base)


# computing the transfer performance for all datasets for all trained models in method 2.
if not os.path.exists(outpath_base_m2):
    os.makedirs(outpath_base_m2)
job_string = job_string_base + '''python eval.py --embdfile {8} --model {0}/alpha_{1}_beta_{2}/model.pickle  --test_path {3} --outputfile {4} --train_path {5} --test_data {6} --train_data {7}'''
alphas = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
betas = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
for task in transfer_datasets:
    for alpha in alphas:
        for beta in betas:
            curr_job = job_string.format(outpath_base_m2, alpha, beta, data_to_path[task],\
                join(results_dir, task+"_m2.csv"), data_to_path["SNLI"], task, "SNLI", glove_path)
            submit_job(curr_job, "eval"+str(alpha)+str(beta)+task, outpath_base)


