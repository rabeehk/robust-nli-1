# Description

In this repository, we fixed the bugs in InferSent model and training used in repository [robust-nli](https://github.com/azpoliak/robust-nli/tree/master),
as explained in Karimi et al, ACL, 2019 and provide the scripts to compute the transfer performance reported in Karimi et al, ACL, 2019.

## Reproduce Transfer performance

To reproduce the transfer performances reported in Karimi et al, ACL, 2019, updates 
the required paths in the scripts and run the following commands:

```
cd src
python scripts/run-finetune.py
python scripts/run-transfer.py
python scripts/eval-transfer.py
```

## Robust NLI Using Adversarial Learning

Training NLU models robustly to ignore annotations artificats
that allow hypothesis only models to outperform majority baselines.
The goal is to be able to train NLI models on datasets with annotation artifcats
and then perform well on different datasets that do not contain those artifacts.

### Requirements
All code in the repo relies on python2.7 and `anaconda2`.

To create a conda enviornment with all required packages, run `conda env create -f environment.yml`

This project relies on [pytorch](http://pytorch.org/) and is based on [InferSent](https://github.com/facebookresearch/InferSent). 

### Data
We provide a bash script that can be used to downlod all data used in our experiments. The script also cleans and processes the data.
To get and process the data, go in `data` and run `./get_data.sh`.  

### Training

To train a hypothesis-only NLI model, use `src/train.py`.

All command line arguments are initialized with default values. If you ran `get_data.sh` as described above, all of the paths will be set directly and you can just run `src/train.py`. 

The most useful command line arguments are:

- `embdfile` - File containin the word embeddings
- `outputdir` - Output directory to store the model after training
- `train_lbls_file` NLI train data labels file
- `train_src_file`  NLI train data source file
- `val_lbls_file`  NLI validation (dev) data labels file
- `val_src_file`   NLI validation (dev) data source file
- `test_lbls_file` NLI test data labels file
- `test_src_file`  NLI test data source file 
- `remove_dup` 1 to remove duplicate hypothesis from train, 0 to keep them in. 0 is the default

#### Adversarial Learning Hyper-parameters
- `adv_lambda` Controls the loss weight of the hypothesis only classifier.  
- `adv_hyp_encoder_lambda` Controls the adversarial weight for the hypothesis only encoder
- `nli_net_adv_hyp_encoder_lambda` Controls the adversarial weight for the hypothesis encoder in NLI net
- `random_premise_frac` Controls the fraction of randome premises to use in NLI net

##### Mapping to hyper-parameters in the papers
In *Don't Take the Premise for Granted (ACL)*:
  - alpha refers to `adv_hyp_encoder_lambda` (Method 1) and `random_premise_frac` (Method 2)
  - beta refers to `adv_lambda` (Method 1) and `nli_net_adv_hyp_encoder_lambda` (Method 2)
  
In *On Adversarial Removal of Hypothesis-only Bias (StarSem)*:
  - λLoss refers to `adv_lambda`
  - λEnc refers to `adv_hyp_encoder_lambda`
  - λRand refers to `random_premise_frac`
  - λRandAdv refers to `nli_net_adv_hyp_encoder_lambda`

To see a description of more command line arguments, run `src/train.py --help`.

##### Hyper-parameters for transfer experiments
These are the hyper-parameter values for the transfer experiments reported in table 2 of our ACL paper:

| test set | adv_lambda | adv_hyp_encoder_lambda | random_premise_frac | nli_net_adv_hyp_encoder_lambda |
| --- | --- | --- | --- | --- |
| SNLI test | 0.1 | 0.2 | 0.05 | 0.05 |
| SNLI hard | 0.1 | 0.2 | 0.05 | 0.05 |
| GLUE | 1 | 0.05 | 0.1 | 0.05 |
| MNLI mismatched | 1 | 0.05 | 0.1 | 0.05 |
| MNLI matched | 0.4 | 0.1 | 0.1 | 0.05 |
| JOCI test | 0.8 | 0.05 | 0.05 | 0.05 |
| MPE test | 0.1 | 1 | 0.05 | 0.2 |
| SICK test | 0.1 | 1 | 0.1 | 0.05 |
| ADD-ONE-RTE test | 0.8 | 0.4 | 0.8 | 1 |
| SCITAIL test | 0.05 | 0.8 | 0.1 | 0.1 |
| DPR test | 1 | 0.2 | 0.05 | 0.4 |
| SPRL test | 1 | 1 | 1 | 1 |
| FNPLUS test | 0.8 | 1 | 0.2 | 0.2 |


## Bibligoraphy
If you use this repo, please cite the three following papers:

```
@inproceedings{karimi-etal-2019-endtoend,
    title = "End-to-End Bias Mitigation by Modelling Biases in Corpora",
    author = "Karimi Mahabadi, Rabeeh and Belinkov, Yonatan  and Henderson, James",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    publisher = "Association for Computational Linguistics",
}


@inproceedings{belinkov-etal-2019-dont,
    title = "Don{'}t Take the Premise for Granted: Mitigating Artifacts in Natural Language Inference",
    author = "Belinkov, Yonatan  and Poliak, Adam  and Shieber, Stuart  and Van Durme, Benjamin  and Rush, Alexander",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1084"
}

@inproceedings{belinkov-etal-2019-adversarial,
    title = "On Adversarial Removal of Hypothesis-only Bias in Natural Language Inference",
    author = "Belinkov, Yonatan  and Poliak, Adam  and Shieber, Stuart  and Van Durme, Benjamin  and Rush, Alexander",
    booktitle = "Proceedings of the Eighth Joint Conference on Lexical and Computational Semantics (*{SEM} 2019)",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/S19-1028"
}
```
