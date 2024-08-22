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
- (2024.08.22) Release training code and checkpoint for SAT, AAR will be available soon.
- (2024.08.21) Add inference code, will release checkpoint and training code soon.
- (2024.08.20) Repo created. Code and checkpoints will be released in this week.


# AAR ckpts
We provide AAR model trained with AudioSet for you, which are on  <a href='https://huggingface.co/qiuk6/AAR'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-AAR-yellow'></a> or can be downloaded from the following links:
|   model    | HF weights 🤗  
|:----------:|:------------------------------------------------------------------------------------|
|    SAT     |[SAT.pth](https://huggingface.co/qiuk6/AAR/resolve/main/SAT_bs_1536_d1024_lat64.pth) |


# Dataset

We download our Audioset from website https://research.google.com/audioset/ and collect it as 

```
AudioSet
├── audioset_unbalanced_train_mp3
├── unbalanced_train_segments.csv
└── audioset_eval_raw_mp3
```

# Installation

- Install all packages via ```pip3 install -r requirements.txt```.


# Training

```
python3 train_SAT_mpi.py --config config/train/SAT.yaml --train_dir /path/to/audioset_unbalanced_train_mp3 --train_csv /path/to/csv --batch_size $bs --gpus $gpus --output_dir /path/to/save/ckpt --use_prefetcher True --resume latest
```

# Inference

```
python3 inference_SAT.py --config config/inference/SAT.yaml --resume /path/to/ckpt.pth --test_dir /path/to/audioset_eval_raw_mp3 --batch_size $bs
```


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
