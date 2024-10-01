# FLORES200 dataset

You need to add the FLORES200 dataset inside this folder, which you can get [here](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#download).

This folder should have the following structure:
```
data/flores200
    -> augmented_data
        -> eng_Latn_augmented.devtest
        -> eng_Latn_errtype.devtest.json
    -> dev
        -> ace_Arab.dev
        -> ace_Latn.dev
        ...
    -> devtest
        -> ace_Arab.devtest
        -> ace_Latn.devtest
        ...
    metadata_dev.tsv
    metadata_devtest.tsv
```