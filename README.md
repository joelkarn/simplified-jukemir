# Simplified JukeMIR (for Apple Silicon)

This repository Simplified JukeMIR made to work with Apple Silicon GPUs. If you want to use a regular GPU, or see how to set everything up, please see the original repo: 
```
https://github.com/ldzhangyx/simplified-jukemir
```

## Usage

### 1. Use JukeMIR to extract features

`python extract_features.py`

### 2. Use features to train the model for music genre classification 

`python classify_music.py`

## Citation

The Simplified JukeMIR was based on JukeMIR, so if you want to use any of these, please consider citing the original paper:

```
@inproceedings{castellon2021calm,
  title={Codified audio language modeling learns useful representations for music information retrieval},
  author={Castellon, Rodrigo and Donahue, Chris and Liang, Percy},
  booktitle={ISMIR},
  year={2021}
}
```

You are also welcome to add a footnote referring to the Simple JukeMIR project:

```
https://github.com/ldzhangyx/simplified-jukemir
```
