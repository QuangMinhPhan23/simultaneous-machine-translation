# About this repository

This is a fork of **EAST** (Fu et al., XMU, [biaofuxmu/EAST](https://github.com/biaofuxmu/EAST)), a
framework for simultaneous machine translation. I extended it with two projects of my own. The original
EAST README follows below.

**The base framework (not written by me):** `src/`, `examples/`, and the demo files in `data/` are the
upstream EAST code. My work builds on top of it. I removed the parts of the upstream repo this project
never uses (the ceval/cmmlu/mmlu benchmark data and the LLaMA-Factory utility scripts).

**My contribution.** Each folder has one README for instructions and one for results:

- **`east_scripts/`**: adapting EAST to spoken Egyptian Arabic (data building, chunking, DPO,
  evaluation). See [east_scripts/README.md](east_scripts/README.md) and
  [east_scripts/RESULTS.md](east_scripts/RESULTS.md).
- **`word_order_study/`**: does source-to-target word order drive quality loss in simultaneous MT?
  Tested across 5 languages (Vietnamese, MSA, Korean, Saudi, Egyptian). See
  [word_order_study/README.md](word_order_study/README.md) and
  [word_order_study/RESULTS.md](word_order_study/RESULTS.md).

`east_scripts/` is the bigger project, so it is split into `data/` (build the training data), `train/`
(fine-tune), `eval/` (score the result) and `jobs/` (cluster job scripts). `word_order_study/` is small
enough to stay flat. The datasets and model weights are not in this repo because they are too large: the
data is rebuilt by the scripts, and the trained models are on Hugging Face.

---

# LLMs Can Achieve High-quality Simultaneous Machine Translation as Efficiently as Offline
Source code for the paper: [LLMs Can Achieve High-quality Simultaneous Machine Translation as Efficiently as Offline](https://arxiv.org/abs/2504.09570)


## Contents
- [Requirements and Installation](#requirements-and-installation)
- [Preparation](#preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Checkpoints](#checkpoints)

## Requirements and Installation
- torch==2.1.0
- transformers==4.44.2
- deepspeed==0.14.0
- peft==0.11.1
- To install llamafactory and develop locally:
```shell script
pip install -e ./
```

## Preparation
#### Training data
The training data used in the paper can be downloaded:
| Huggingface | ModelScope  |
| ---- | ---- |
|[SiMT-De-En-660K 🤗](https://huggingface.co/datasets/biaofu-xmu/SiMT-De-En-660K) | [SiMT-De-En-660K 🤖](https://modelscope.cn/datasets/BiaoFuXMU/SiMT-De-En-660K) |
|[SiMT-Multi-90K 🤗](https://huggingface.co/datasets/biaofu-xmu/SiMT-Multi-90K) | [SiMT-Multi-90K 🤖](https://modelscope.cn/datasets/BiaoFuXMU/SiMT-Multi-90K) |


Preprocessing training data:

```shell script
python east_scripts/data/prepare_train_data.py
```
Then, add the following content to `data/dataset_info.json`
```json
  "simt_de_en_660k": {
    "file_name": "mt_data/train_data/SiMT-De-En-660K.json"
  },
  "simt_multi_90k": {
    "file_name": "mt_data/train_data/SiMT-Multi-90K.json"
  },
```
#### Test data
Preprocessing wmt test data or your data:

```shell script
python east_scripts/data/prepare_mt_test_data.py
```


## Training

#### 1. Stage I: Full Fine-Tuning

```shell script
sh east_scripts/train_simulmt_full_sft_stage1.sh
```

#### 2. Stage II: LoRA Fine-Tuning

```shell script
sh east_scripts/train_simumt_lora_sft_stage2.sh
```

#### 3. Merge Lora Weights to Base Model

```shell script
sh east_scripts/merge_lora.sh
```

## Evaluation
```shell script
sh east_scripts/simul_eval.sh
```

## Checkpoints

Our models are released here:
| Huggingface | ModelScope  |
| ---- | ---- |
|[EAST-8B 🤗](https://huggingface.co/biaofu-xmu/EAST-8B) | [EAST-8B 🤖](https://modelscope.cn/models/BiaoFuXMU/EAST-8B) |
|[EAST-Stage-1-8B 🤗](https://huggingface.co/biaofu-xmu/EAST-Stage-1-8B) | [EAST-Stage-1-8B 🤖](https://modelscope.cn/models/BiaoFuXMU/EAST-Stage-1-8B) |

## Acknowledgement
This repo benefits from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Thanks for their wonderful works.


## Citation

If you find this repo useful for your research, please consider citing the paper:
```
@misc{fu2025llmsachievehighqualitysimultaneous,
      title={LLMs Can Achieve High-quality Simultaneous Machine Translation as Efficiently as Offline}, 
      author={Biao Fu and Minpeng Liao and Kai Fan and Chengxi Li and Liang Zhang and Yidong Chen and Xiaodong Shi},
      year={2025},
      eprint={2504.09570},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.09570}, 
}
```