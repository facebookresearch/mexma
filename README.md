# MEXMA: Token-level objectives improve sentence representations

This repository contains the code necessary to train MEXMA, as presented in the [MEXMA paper](https://arxiv.org/abs/2409.12737).
![MEMXA architecture](/assets/MEXMA.png)


## Setup
### Packages and requirements
Python version 3.11.9

1) conda create --name mexma python=3.9.11
2) conda activate mexma
3) git clone git@github.com:facebookresearch/mexma.git
4) cd mexma
5) pip install -r requirements.txt

### Training data
More details about the training data are present in [data/train_data](/data/train_data/README.md).

### Evaluation
You need to add the [xsim file](https://github.com/facebookresearch/LASER/blob/main/source/xsim.py) to evaluation/xsim, in order to be able to evaluate on it during training.

Additionally, you also need to add the FLORES200 dataset inside data/flores200, which you can get [here](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#download).

## Training the model
In order to train the model, simply launch:
``` 
torchrun main.py \
    --encoder xlm-roberta-large \
    --max_model_context_length 200 \
    --checkpoint None \
    --mlm_loss_weight 1 \
    --cls_loss_weight 1 \
    --koleo_loss_weight 0.01 \
    --number_of_linear_layers 0 \
    --linear_layers_inputs_dims None  \
    --linear_layers_outputs_dims None  \
    --number_of_transformer_layers_in_head 6 \
    --number_of_transformer_attention_heads_in_head 8 \
    --initialization_method torch_default \
    --train_data_file None  \
    --test_data_file None  \
    --hf_dataset_directory [YOUR_DIRECTORY_HERE]  \
    --batch_size 150 \
    --workers 12 \
    --device cuda \
    --lr 0.0001 \
    --epochs 3 \
    --start_epoch 0 \
    --src_mlm_probability 0.4 \
    --trg_mlm_probability 0.4 \
    --number_of_iterations_to_accumulated_gradients 2 \
    --testing_frequency 5000000 \
    --saving_frequency 2000 \
    --mixed_precision_training  \
    --clip_grad_norm 1.2  \
    --wd None  \
    --lr_scheduler_type cosineannealinglr  \
    --lr_warmup_percentage 0.3 \
    --lr_warmup_method linear \
    --lr_warmup_decay 0.1 \
    --print_freq 10 \
    --save_model_checkpoint 50000 \
    --no_wandb \
    --flores_200_src_languages acm_Arab aeb_Arab afr_Latn amh_Ethi ary_Arab  arz_Arab asm_Beng azb_Arab azj_Latn bel_Cyrl ben_Beng bos_Latn bul_Cyrl cat_Latn ces_Latn ckb_Arab cym_Latn dan_Latn deu_Latn ell_Grek epo_Latn est_Latn eus_Latn fin_Latn fra_Latn gla_Latn gle_Latn glg_Latn guj_Gujr hau_Latn heb_Hebr hin_Deva hrv_Latn hun_Latn hye_Armn ind_Latn isl_Latn ita_Latn jav_Latn jpn_Jpan kan_Knda kat_Geor kaz_Cyrl khm_Khmr kir_Cyrl kor_Hang lao_Laoo mal_Mlym mar_Deva mkd_Cyrl mya_Mymr nld_Latn nno_Latn nob_Latn npi_Deva pol_Latn por_Latn ron_Latn rus_Cyrl san_Deva sin_Sinh slk_Latn slv_Latn snd_Arab som_Latn spa_Latn srp_Cyrl sun_Latn swe_Latn swh_Latn tam_Taml tel_Telu tha_Thai tur_Latn uig_Arab ukr_Cyrl urd_Arab vie_Latn xho_Latn zho_Hant
```

## Loading the pretrained model
In order to use the MEXMA model, follow these steps:
1) ```wget https://dl.fbaipublicfiles.com/mexma/MEXMA.zip```
2) ```unzip MEXMA.zip```
3) 
```
from transformers import XLMRobertaModel
import torch
model = XLMRobertaModel.from_pretrained('MEXMA')
input_ids = torch.randint(low=0, high=25000, size=(5,10))
outputs = model(input_ids)
cls_embeddings = outputs.last_hidden_state[:,0,:]
assert cls_embeddings.shape == torch.Size([5,1024]), "There was an issue loading the model, the shapes are incorrect"
```

## License
MEXMA is MIT licensed. See the [LICENSE](LICENSE) file for details. However portions of the project are available under separate license terms: backbone/block_diagonal_roberta.py and losses/koleo.py are licensed under the Apache-2.0 license.

## Citation
```
@misc{janeiro2024mexma,
    title={MEXMA: Token-level objectives improve sentence representations}, 
    author={João Maria Janeiro and Benjamin Piwowarski and Patrick Gallinari and Loïc Barrault},
    year={2024},
    eprint={2409.12737},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2409.12737}, 
}
```
