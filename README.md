# Efficient Autoregressive Audio Modeling via Next-Scale Prediction


<div align="center">

[![project page](https://img.shields.io/badge/AAR%20project%20page-lightblue)]()&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2404.02905-b31b1b.svg)](https://arxiv.org/pdf/2408.09027)&nbsp;
[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-yellow)](https://huggingface.co/qiuk6/AAR)&nbsp;

</div>
<p align="center" style="font-size: larger;">
  <a href="https://arxiv.org/pdf/2408.09027">Efficient Autoregressive Audio Modeling via Next-Scale Prediction</a>
</p>

<p align="center">
<img src="assets/pipeline.png" width=95%>
<p>

<be>
  
# Updates 
- (2024.08.22) Add SAT inference/training code. Codes about AAR will be available soon.
- (2024.08.20) Repo created. Code and checkpoints will be released this week.


# Installation

- Install all packages via ```pip3 install -r requirements.txt```.


# Dataset

We download our Audioset from the website https://research.google.com/audioset/ and collect it as 

```
AudioSet
├── audioset_unbalanced_train_mp3
├── unbalanced_train_segments.csv
└── audioset_eval_raw_mp3
```

---

# Scale-level audio tokenizer (SAT)

## Training

```
python3 train_SAT_mpi.py --config config/train/SAT.yaml --train_dir /path/to/audioset_unbalanced_train_mp3 --train_csv /path/to/csv --batch_size $bs --gpus $gpus --output_dir /path/to/save/ckpt --use_prefetcher True --resume latest
```

## Inference

```
python3 inference_SAT.py --config config/inference/SAT.yaml --resume /path/to/ckpt.pth --test_dir /path/to/audioset_eval_raw_mp3 --batch_size $bs
```

## Pre-trained model
We provide Audioset pre-trained SAT checkpoint as follows:
|   model    | # Scale | # Tokens |latent_dim| FAD | HF weights 🤗  |
|:----------:|:--------|:---------|:---------|:----|:-------------- |
|    SAT     |    16   |   455    |   64     | 1.09|[SAT.pth](https://huggingface.co/qiuk6/AAR/resolve/main/SAT_bs_1536_d1024_lat64.pth) |
|    SAT     |   16    |   455    |  128     | 1.40|(SAT.pth)()     |


# Acoustic AutoRegressive Modeling (AAR)

## Training

```
python3 train_AAR_mpi.py --config config/train/AAR.yaml --train_dir /path/to/audioset_unbalanced_train_mp3 --train_csv /path/to/csv --batch_size $bs --gpus $gpus --output_dir /path/to/save/ckpt --use_prefetcher True --resume latest --vqvae_pretrained_path /path/to/vae/ckpt --latent_dim $latent --dimension $dim 
```

## Inference

```
python3 inference_AAR.py --config config/inference/AAR.yaml --aar_pretrained_path /path/to/aar.pth --vqvae_pretrained_path /path/to/vqvae.pth --test_dir /path/to/audioset_eval_raw_mp3 --batch_size $bs --output_dir /path/to/save
```

## Pre-trained model
We provide Audioset pre-trained AAR checkpoint as follows:
|   model    | # Scale | # Tokens |latent_dim| FAD | HF weights 🤗  |
|:----------:|:--------|:---------|:---------|:----|:-------------- |
|    SAT     |   16    |   455    |  128     | 1.40|[SAT.pth](https://huggingface.co/qiuk6/AAR/resolve/main/SAT_d128_lat128.pth)     |
|    AAR     |    16   |   455    |   128     | 6.01|[AAR.pth](https://huggingface.co/qiuk6/AAR/resolve/main/AAR_d16_bs2048.pth) |

# Citation
```
@misc{qiu2024efficient,
    title={Efficient Autoregressive Audio Modeling via Next-Scale Prediction},
    author={Kai Qiu and Xiang Li and Hao Chen and Jie Sun and Jinglu Wang and Zhe Lin and Marios Savvides and Bhiksha Raj},
    year={2024},
    eprint={2408.09027},
    archivePrefix={arXiv},
    primaryClass={cs.SD}
}
```
