## Jingju Baseline
It is a baseline of our project about Beijing opera script generation. Our baseline model is based on [gpt2-chinese-ancient](https://huggingface.co/uer/gpt2-chinese-ancient) which is pretrained with 1.5GB literary Chinese.Please refer to our [paper](https://github.com/zhengmidon/jingju_baseline/blob/master/Beijing%20Opera%20Script%20Generation%20with%20GPT-2%20as%20A%20Baseline.pdf) for details.
### Directory Annotation
```
jingju_baseline/
	|-- finetuning.py 	#the finetuning script
	|-- jingju_test.py 	#test script
	|-- preprocess.py 	#data preprocess script
	|-- config/ 		#model configuration files 
	|-- corpora/ 		#corpora files
	|-- models/ 		# vocab file, model checkpoints and some necessary files
	|-- scripts/ 		# several functional scripts
	|-- test/ 			#test files
	|-- uer/ 			#files from UER-py
```
### Environment Preparation
Our baseline model is fineturned with a pretraining framework [UER-py](https://github.com/dbiir/UER-py). Refer to the [part](https://github.com/dbiir/UER-py#requirements) for environment requirements.
### Finetuning

 1. Data preprocess
 ```Bash
 python3 preprocess.py --corpus_path corpora/jingju_train.txt\
		  --vocab_path models/vocab.txt \
		  --tokenizer bert \
		  --dataset_path corpora/jingju_train.pt \
		  --processes_num 32 --seq_length 1024 --target lm
 python3 preprocess.py --corpus_path corpora/jingju_dev.txt\
		  --vocab_path models/vocab.txt \
		  --tokenizer bert \
		  --dataset_path corpora/jingju_dev.pt \
		  --processes_num 32 --seq_length 1024 --target lm
 ```
 2. Finetuning
```Bash
export CUDA_VISIBLE_DEVICES=0
nohup python3 -u finetuning.py --dataset_path corpora/jingju_train.pt\
		 --devset_path corpora/jingju_dev.pt\
		 --vocab_path models/vocab.txt \
		 --config_path config/jingju_config.json \
		 --output_model_path models/finetuned_model.bin\
		 --pretrained_model_path models/uer-ancient-chinese.bin\
		 --world_size 1 --gpu_ranks 0  \
		 --total_steps 100000 --save_checkpoint_steps 50000\
		 --report_steps 1000 --learning_rate 5e-5\
		 --batch_size 5 --accumulation_steps 4 \
		 --embedding word_pos  --fp16 --fp16_opt_level O1 \
		 --remove_embedding_layernorm --encoder transformer \
		 --mask causal --layernorm_positioning pre \
		 --target lm --tie_weights > fineturning.log 2>&1 &
```
**Refer to [here](https://github.com/dbiir/UER-py/wiki/%E9%A2%84%E5%A4%84%E7%90%86) for  function of every argument.**

Specificly, you may change environment variable `CUDA_VISIBLE_DEVICES` and `--world_size` paired with `--gpu_ranks` option for multi-GPU training. To enable `--fp16` coordinated with `--fp16_opt_level`  needs [apex](https://github.com/NVIDIA/apex).
### Test
You can finetuning by yourself with instructions above, or download the [checkpoint](https://pan.baidu.com/s/1K5flY2Wex6aLkfSIKtfUcA)(extracting code: q0yn) to directory `./models`
Then run as follows:
```bash
python3 preprocess.py --corpus_path corpora/jingju_test.txt\
		  --vocab_path models/vocab.txt \
		  --tokenizer bert \
		  --dataset_path corpora/jingju_test.pt \
		  --processes_num 32 --seq_length 1024 --target lm
```
```bash
nohup python3 -u jingju_test.py --load_model_path models/finetuned_model.bin-100000 \
		--vocab_path models/vocab.txt \
		--beginning_path test/jingju_beginning.txt  \
		--reference_path test/jingju_reference.txt \
		--prediction_path test/jingju_candidates.txt \
		--test_path test/jingju_beginning.txt \
		--testset_path datasets/jingju_test.pt \
		--config_path config/jingju_config.json \
		--seq_length 1024 --embedding word_pos \
		--remove_embedding_layernorm \
		--encoder transformer --mask causal \
		--layernorm_positioning pre --target lm \
		--tie_weights > test_candidate_generation.log 2>&1 &
```
The automatic mertics(i.e., **F1, Perplexity, BLEU and Distinct**) will be displayed on _stdout_.
### Generation
```bash
nohup python3 -u scripts/generate_lm.py \
		--load_model_path models/finetuned_model.bin-100000 \
		--vocab_path models/vocab.txt \
		--test_path test/beginning.txt \
		--prediction_path test/generation.txt \
		--config_path config/jingju_config.json \
		--seq_length 1024 --embedding word_pos \
		--remove_embedding_layernorm --encoder transformer \
		--mask causal --layernorm_positioning pre \
		--target lm --tie_weights > generation_log.log 2>&1 &
```
Given the beginning, the model will generates script corresponding with it.
The `generate_lm.py` script only generates sequence no longer than 1024. If you want longer script, replace `scripts/generate_lm.py` with `scripts/long_generate_lm.py` and revise `--seq_length` to the length you desire. **Note** that the generation procedure employs auto-regressive fashion, so generating long sequence is a time-consuming process.
## Citation
```
@article{zhao2019uer,
  title={UER: An Open-Source Toolkit for Pre-training Models},
  author={Zhao, Zhe and Chen, Hui and Zhang, Jinbin and Zhao, Xin and Liu, Tao and Lu, Wei and Chen, Xi and Deng, Haotang and Ju, Qi and Du, Xiaoyong},
  journal={EMNLP-IJCNLP 2019},
  pages={241},
  year={2019}
}
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  year={2019}
}
```

