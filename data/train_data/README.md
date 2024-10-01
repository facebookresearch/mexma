# Train data

For our experiments we created a huggingface dataset, using the [datasets](https://huggingface.co/docs/datasets/en/index), where the data is structured like this:
```
src,trg
[SENTENCE1],[TRANSLATION_OF_SENTENCE1]
```

If you wish to use HuggingFace datasets too, they should have the same columns, and you need to set the flag "--hf_dataset_directory" with the directory of the dataset.

If you instead wish to use a csv file with those same columns, you set the flag "--train_data_file".