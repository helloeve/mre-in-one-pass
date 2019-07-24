# Extracting Multiple-Relations in One-Pass with Pre-Trained Transformers

## Notes
This repo contains the implementation for the algorithm in:
```
@article{wang2019extracting,
  title={Extracting Multiple-Relations in One-Pass with Pre-Trained Transformers},
  author={Wang, Haoyu and Tan, Ming and Yu, Mo and Chang, Shiyu and Wang, Dakuo and Xu, Kun and Guo, Xiaoxiao and Potdar, Saloni},
  journal={arXiv preprint arXiv:1902.01030},
  year={2019}
}
```

The codes are modified based on the original BERT [repo](https://github.com/google-research/bert). Some unrelated modules from the original repo have been deleted in order to make it easy to understand.

## Data
We provide the processed Semeval2018 data along with the repo. For ACE dataset, we could not share it within this place due to the data policy.

## Training (MRE)
The following command will work for training the model on Semeval dataset. For other configurable arguments, please refer to `run_classifier.py`.

```
python run_classifier.py \
        --task_name=semeval \
        --do_train=true \
        --do_eval=false \
        --do_predict=false \
        --data_dir=$DATA_DIR/semeval2018/multi \
        --vocab_file=$BERT_BASE_DIR/vocab.txt \
        --bert_config_file=$BERT_BASE_DIR/bert_config.json \
        --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
        --max_seq_length=128 \
        --train_batch_size=4 \
        --learning_rate=2e-5 \
        --num_train_epochs=30 \
        --max_distance=2 \
        --max_num_relations=12 \
        --output_dir=<path to store the checkpoint>
```

## Prediting
The following command will work for using the trained model to inference on the test dataset.
```
python run_classifier.py \
        --task_name=semeval \
        --do_train=false \
        --do_eval=false \
        --do_predict=true \
        --data_dir=$DATA_DIR/semeval2018/multi \
        --vocab_file=$BERT_BASE_DIR/vocab.txt \
        --bert_config_file=$BERT_BASE_DIR/bert_config.json \
        --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
        --max_seq_length=256 \
        --max_distance=2 \
        --max_num_relations=12 \
        --output_dir=<path to the stored checkpoint>
```

## Training (SRE)
The following command will work for training the model on Semeval dataset. For other configurable arguments, please refer to `run_classifier.py`.

```
python run_classifier.py \
        --task_name=semeval \
        --do_train=true \
        --do_eval=false \
        --do_predict=false \
        --data_dir=$DATA_DIR/semeval2018/single \
        --vocab_file=$BERT_BASE_DIR/vocab.txt \
        --bert_config_file=$BERT_BASE_DIR/bert_config.json \
        --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
        --max_seq_length=128 \
        --train_batch_size=4 \
        --learning_rate=2e-5 \
        --num_train_epochs=15 \
        --max_distance=2 \
        --max_num_relations=1 \
        --output_dir=<path to store the checkpoint>
```

## Prediting
The following command will work for using the trained model to inference on the test dataset.
```
python run_classifier.py \
        --task_name=semeval \
        --do_train=false \
        --do_eval=false \
        --do_predict=true \
        --data_dir=$DATA_DIR/semeval2018/single \
        --vocab_file=$BERT_BASE_DIR/vocab.txt \
        --bert_config_file=$BERT_BASE_DIR/bert_config.json \
        --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
        --max_seq_length=256 \
        --max_distance=2 \
        --max_num_relations=1 \
        --output_dir=<path to the stored checkpoint>
```