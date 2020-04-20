import itertools
import os

job_string_base = '''
#!/bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -l gpu 

source activate transformer
'''

# All parameters that need to be set.
outpath_base_m1 = '/idiap/temp/rkarimi/robust-nli/method1/'
outpath_base_m2 = '/idiap/temp/rkarimi/robust-nli/method2/'
glove_path = "/idiap/temp/rkarimi/resources/GloVe/glove.840B.300d.txt"
snli_path = "/idiap/temp/rkarimi/datasets/SNLI"


def submit_job(curr_job, filename, outpath_base):
    job_name = "template.job"
    with open(job_name, "w") as f:
       f.write(curr_job)
    os.system("qsub -V -N {0} -e {1}.err -o {1}.out template.job".format(filename, os.path.join(outpath_base, filename)))


# Hyper-parameter tuning for method 1.
if not os.path.exists(outpath_base_m1):
    os.makedirs(outpath_base_m1)
job_string = job_string_base + '''python train.py --embdfile {3} --outputdir {0}/alpha_{1}_beta_{2}  --pool_type max --nlipath {4} --n_classes 3  --adv_lambda {2} --adv_hyp_encoder_lambda {1} --nli_net_adv_hyp_encoder_lambda 0 --random_premise_frac 0 --enc_lstm_dim 512''' 
alphas = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
betas = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
for alpha in alphas: 
   for beta in betas:   
       curr_job = job_string.format(outpath_base_m1, alpha, beta, glove_path, snli_path)
       submit_job(curr_job, "m1"+str(alpha)+str(beta), outpath_base)


# Hyper-parameter tuning for method 2
if not os.path.exists(outpath_base_m2):
    os.makedirs(outpath_base_m2)
job_string = job_string_base + '''python train.py --embdfile {3} --outputdir {0}/alpha_{1}_beta_{2}  --pool_type max --nlipath {4} --n_classes 3 --adv_lambda 0 --adv_hyp_encoder_lambda 0 --nli_net_adv_hyp_encoder_lambda {2} --random_premise_frac {1} --enc_lstm_dim 512''' 
alphas = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
betas = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
for alpha in alphas: 
   for beta in betas:   
       curr_job = job_string.format(outpath_base_m2, alpha, beta, glove_path, snli_path)
       submit_job(curr_job, "m2"+str(alpha)+str(beta), outpath_base)
