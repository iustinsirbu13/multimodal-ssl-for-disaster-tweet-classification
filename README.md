# multimodal-ssl-for-disaster-tweet-classification

This repository contains the code and data used for training the models described in [Multimodal Semi-supervised Learning for Disaster Tweet Classification](https://aclanthology.org/2022.coling-1.239) (Sirbu et al., COLING 2022)


The code is based on the [FixMatch-pytorch](https://github.com/kekmodel/FixMatch-pytorch) implementation, combined with [MMBT](https://github.com/facebookresearch/mmbt) for multimodal support. Other repos used for pre-processing the tweets, augmenting them with EDA or Fairseq (back-translation) are also provided in the [Code References](#code-references) section.

## Running instructions

The main file is code/FixMatch-pytorch/myTrain.py. Some example scripts are also provided inside the FixMatch-pytorch folder (e.g. train_fixmatch_mmbt_humanitarian_ls2.cmd). 

The most relevant arguments would be: 
* --mu - the ratio between labeled and unlabeled data
* --k-img - the size of the labeled data, only used for time estimation purposes
* --text_soft_aug and --text_hard_aug - the soft and the hard augmentation used for text (name of the corresponding fields as they are used in the dataset file); the augmentation for the image is always RandAugment.
* --text_prob_aug - by default 1, in order to use text augmentation. If set to 0, one could reproduce the experiments where the text augmentation is not used; equivalent to setting both augmentations to 'none'.
* --lambda-u - the weight between labeled and unlabeled losses. 
* --train_file and --unlabeld_dataset - used to specify the name of the files to be used during training (e.g. in case there are different files containing the EDA and BT augmentations)

Setups: 
* For classical FixMatch training, the flag --lambda-u is set to a positive constant (usually 1); it may be 0 for supervised training.
* For FixMatchLS training, the flags "--linear_lu --distil unlabeled" have to be used. Also, --lambda-u shoud be set to C*num_epochs (i.e. the weight will vary linearly between 0 and this value, increasing by C every epoch). 

Modes:
* For training a new model, --out should be set to the output directory that will be created for the experiment (e.g. <local_path>/<experiment_name>)
* For resuming the training of a model from the last checkpoint, --resume should be set to <local_path>/<experiment_name>/checkpoint.pth.tar
* For evaluating a model at it's best checkpoint, the flag --eval_only has to be set; also, --out has to be set to <local_path>/<experiment_name>/model_best.pth.tar

Data format:
The data format used by this code is the following:
* .jsonl files (labeled, unlabeled, dev and test)
* {"label": "informative", "image": "path/to/image.png", "text": "tweet text example", "eda_01": ["eda augmentation 1", "eda augmentation 2", "eda augmentation 3", ...], "eda_02": ["eda augmentation 20% instead of 10%", "eda augmentation 20% 2", ...], "back_translate": ["back translation 1", "back translation 2", ...], ...}
* The names of dict keys for the augmented versions are not relevant, as long as the same names are used by the --text_soft_aug and --text_hard_aug flags. Only the augmentations being used are required to exist in the file.
* Only the --train_file and the --unlabeld_dataset have to contain the text augmentations used, as no augmentation is applied for dev.jsonl and test.jsonl. 
* A simple way of adding text augmentation is by using a script similar to scripts_cmd/add_eda_to_jsonl.cmd.

## Code References

[FixMatch-pytorch](https://github.com/kekmodel/FixMatch-pytorch)

[MMBT](https://github.com/facebookresearch/mmbt)

[modules/aidrtokenize.py](https://github.com/firojalam)

[modules/eda.py](https://github.com/jasonwei20/eda_nlp)

[Fairseq](https://github.com/facebookresearch/fairseq)

## Citations

```
@inproceedings{sirbu2022multimodal,
  title={Multimodal Semi-supervised Learning for Disaster Tweet Classification},
  author={Sirbu, Iustin and Sosea, Tiberiu and Caragea, Cornelia and Caragea, Doina and Rebedea, Traian},
  booktitle={Proceedings of the 29th International Conference on Computational Linguistics},
  pages={2711--2723},
  year={2022}
}
```