# Simplified JukeMIR: A minimum version for extracting features with Jukebox

This repository contains code is for extracting JukeMIR features in a simplified way on your server.

It generates a (4800, 1) vector for a 23.77s audio clip. The script will automatically cut or pad the waveform to this length.

It requires a single GPU with 12GB memory.

## Change Logs

- I remove files unrelated to the feature extraction, making the code that can be easily embedded into your own projects.
- I provide a guide that you can set up the environment without `docker` (since we usually do not have `sudo` permission in our servers).
- I fix the code file to make sure we can set up the pretrained models path manually.
- I remove the `mpi4py` dependency, which causes errors in the environment.
- I add a zero-padding code so music clips shorter than `23.44s` will be processed correctly.
- I fix the prior model loading error.
- etc.

## Installation Guide

### Step 1: Installing JukeMIR

The simplified JukeMIR package is basically the same as the original version, but I removed the `mpi4py` module, which is for parallel computing. `mpi4py` is difficult to be installed to the environment and has no effect on feature extraction. 

Run:

```
python -m pip install --no-cache-dir -e jukebox
```

### Step 2: Downloading pretrained models

Jukebox consists of two models: the VQ-VAE model and the prior model.

Run:

```
wget https://openaipublic.azureedge.net/jukebox/models/5b/vqvae.pth.tar
wget https://openaipublic.azureedge.net/jukebox/models/5b/prior_level_2.pth.tar
```

You can move the pretrained models to anywhere you like.

### Step 3: Changing the parameters

I recommend to precalculate the representations of your data in advance, since it takes really long.

The parameters you may change are in the beginning of `representation.py`:

```
JUKEBOX_SAMPLE_RATE = 44100
T = 8192
SAMPLE_LENGTH = 1048576 # ~23.77s, which is the param in JukeMIR
DEVICE='cuda'
VQVAE_MODELPATH = "models/5b/vqvae.pth.tar"
PRIOR_MODELPATH = "models/5b/prior_level_2.pth.tar"
INPUT_DIR = r"BUTTER_v2/westone-dataset/WAV/"
OUTPUT_DIR = r"BUTTER_v2/jukemir/output/"
os.environ["CUDA_VISIBLE_DEVICES"]="7"
PRIOR_DEPTH = 36
```

### Step 4: Run

Run:

```
python representation.py
```

and you will get the results.

## Citation

If you use this codebase in your work, please consider citing the original paper:

```
@inproceedings{castellon2021calm,
  title={Codified audio language modeling learns useful representations for music information retrieval},
  author={Castellon, Rodrigo and Donahue, Chris and Liang, Percy},
  booktitle={ISMIR},
  year={2021}
}
```

You are also welcome to add a footnote referring to this project:

```
https://github.com/ldzhangyx/simplified-jukemir
```
