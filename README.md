# Bert4Innovation_Classification
To judge the sentences in the paper's abstract wether to be the innovation of the whole paper

## Requirements
* tensorflow==1.11.0
* spacy==3.0.5
* pandas==1.1.5
* numpy==1.19.5
* torch>=0.4.1,<=1.2.0
* scikit-learn==0.24.2

## Run the code
### 1) Prepare the data set:
The pre-training data and the fine-tuning data are in the data dir. 
Run the ``generate_corpus_innovation.py`` get_rand_data_set function can get the fine-tuning datasets from ``innovation_fine_tune_data_x.x.csv`` evenly and randomly, and will save the datasets in ``data/innovationx.x``. 

### 2) Further Pre-Training
#### Generate Further Pre-Training Corpus
Run the ``generate_corpus_innovation.py`` get_corpus function can get the pre-training corpus, which will be saved in ``codes/further_pre_training/corpus/``.
#### Run Further Pre-Training
```shell
python create_pretraining_data.py \
  --input_file=./Innovation_corpus.txt \
  --output_file=tmp/tf_innovation.tfrecord \
  --vocab_file=./uncased_L-12_H-768_A-12/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
  
python run_pretraining.py \
  --input_file=./tmp/tf_innoavtion.tfrecord \
  --output_dir=./uncased_L-12_H-768_A-12_Innovation_pretrain \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=./uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=./uncased_L-12_H-768_A-12/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=100000 \
  --num_warmup_steps=10000 \
  --save_checkpoints_steps=10000 \
  --learning_rate=5e-5
```

### 3) Fine-Tuning
Convert Tensorflow checkpoint to PyTorch checkpoint
```shell
python convert_tf_checkpoint_to_pytorch.py \
  --tf_checkpoint_path ./uncased_L-12_H-768_A-12_Innovation_pretrain/model.ckpt-100000 \
  --bert_config_file ./uncased_L-12_H-768_A-12_Innovation_pretrain/bert_config.json \
  --pytorch_dump_path ./uncased_L-12_H-768_A-12_Innovation_pretrain/pytorch_model.bin
```

#### Fine-Tuning on downstream tasks
