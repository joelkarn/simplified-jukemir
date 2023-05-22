import pathlib
from argparse import ArgumentParser

# imports and set up Jukebox's multi-GPU parallelization
import jukebox
from jukebox.hparams import Hyperparams, setup_hparams
from jukebox.make_models import MODELS, make_prior, make_vqvae
from jukebox.utils.dist_utils import setup_dist_from_mpi
from tqdm import tqdm

import librosa as lr
import numpy as np
import torch
import os
import argparse


# --- MODEL PARAMS ---

JUKEBOX_SAMPLE_RATE = 44100
T = 8192
SAMPLE_LENGTH = 1048576  # ~23.77s, which is the param in JukeMIR
DEPTH = 36


# --------------------


def load_audio_from_file(fpath):
    audio, _ = lr.load(fpath, sr=JUKEBOX_SAMPLE_RATE)
    if audio.ndim == 1:
        audio = audio[np.newaxis]
    audio = audio.mean(axis=0)

    # normalize audio
    norm_factor = np.abs(audio).max()
    if norm_factor > 0:
        audio /= norm_factor

    return audio.flatten()


def audio_padding(audio, target_length):
    padding_length = target_length - audio.shape[0]
    padding_vector = np.zeros(padding_length)
    padded_audio = np.concatenate([audio, padding_vector], axis=0)
    return padded_audio


def get_z(audio, vqvae):
    # don't compute unnecessary discrete encodings
    audio = audio[: JUKEBOX_SAMPLE_RATE * 25]

    zs = vqvae.encode(torch.FloatTensor(audio[np.newaxis, :, np.newaxis]).to("mps"))

    z = zs[-1].flatten()[np.newaxis, :]

    if z.shape[-1] < 8192:
        raise ValueError("Audio file is not long enough")

    return z


def get_cond(hps, top_prior):
    sample_length_in_seconds = 62

    hps.sample_length = (
                                int(sample_length_in_seconds * hps.sr) // top_prior.raw_to_tokens
                        ) * top_prior.raw_to_tokens

    # NOTE: the 'lyrics' parameter is required, which is why it is included,
    # but it doesn't actually change anything about the `x_cond`, `y_cond`,
    # nor the `prime` variables
    metas = [
                dict(
                    artist="unknown",
                    genre="unknown",
                    total_length=hps.sample_length,
                    offset=0,
                    lyrics="""lyrics go here!!!""",
                ),
            ] * hps.n_samples

    labels = [None, None, top_prior.labeller.get_batch_labels(metas, "mps")]

    x_cond, y_cond, prime = top_prior.get_cond(None, top_prior.get_y(labels[-1], 0))

    x_cond = x_cond[0, :T][np.newaxis, ...]
    y_cond = y_cond[0][np.newaxis, ...]

    return x_cond, y_cond


def get_final_activations(z, x_cond, y_cond, top_prior):
    x = z[:, :T]

    # make sure that we get the activations
    top_prior.prior.only_encode = True

    # encoder_kv and fp16 are set to the defaults, but explicitly so
    out = top_prior.prior.forward(
        x, x_cond=x_cond, y_cond=y_cond, encoder_kv=None, fp16=False
    )

    return out


def get_acts_from_file(fpath, hps, vqvae, top_prior, meanpool):
    audio = load_audio_from_file(fpath)

    # zero padding
    if audio.shape[0] < SAMPLE_LENGTH:
        audio = audio_padding(audio, SAMPLE_LENGTH)

    # run vq-vae on the audio
    z = get_z(audio, vqvae)

    # get conditioning info
    x_cond, y_cond = get_cond(hps, top_prior)

    # get the activations from the LM
    acts = get_final_activations(z, x_cond, y_cond, top_prior)

    # postprocessing
    acts = acts.squeeze().type(torch.float32)  # shape: (8192, 4800)

    # average mode
    assert acts.size(0) % meanpool == 0, "The 'meanpool' param must be divisible by 8192. "
    acts = acts.view(meanpool, -1, 4800)
    acts = acts.mean(dim=1)
    acts = np.array(acts.cpu())
    return acts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from audio files.")
    parser.add_argument("input_dir", type=str, help="Path to music tagging parent audio directory.")
    parser.add_argument("output_dir", type=str, help="Path to the output parent directory.")
    args = parser.parse_args()

    # --- SETTINGS ---
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    print(DEVICE)

    # DEVICE = 'cuda'
    VQVAE_MODELPATH = "models/vqvae.pth.tar"
    PRIOR_MODELPATH = "models/prior_level_2.pth.tar"
    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    AVERAGE_SLICES = 1  # For average pooling. "1" means average all frames.
    #  Since the output shape is 8192 * 4800, the params bust can divide 8192.
    USING_CACHED_FILE = False
    model = "5b"  # might not fit to other settings, e.g., "1b_lyrics" or "5b_lyrics"

    # --- SETTINGS ---
    output_dir = pathlib.Path(OUTPUT_DIR)
    input_dir = pathlib.Path(INPUT_DIR)

    #with open('real_definite_paths.txt', 'r') as f:
    with open('test_paths.txt', 'r') as f:
        #input_paths = sorted([pathlib.Path(line.strip()) for line in f])
        input_paths = [pathlib.Path(line.strip()) for line in f]

    device = DEVICE
    # Set up VQVAE

    hps = Hyperparams()
    hps.sr = 44100
    hps.n_samples = 8
    hps.name = "samples"
    chunk_size = 32
    max_batch_size = 16
    hps.levels = 3
    hps.hop_fraction = [0.5, 0.5, 0.125]
    vqvae, *priors = MODELS[model]
    hps_1 = setup_hparams(vqvae, dict(sample_length=SAMPLE_LENGTH))
    hps_1.restore_vqvae = VQVAE_MODELPATH
    vqvae = make_vqvae(
        hps_1, device
    )

    # Set up language model
    hps_2 = setup_hparams(priors[-1], dict())
    hps_2["prior_depth"] = DEPTH
    hps_2.restore_prior = PRIOR_MODELPATH
    top_prior = make_prior(hps_2, vqvae, device)
    for input_path in tqdm(input_paths):
        # Check if output already exists
        output_path = pathlib.Path(output_dir, input_path.relative_to(input_dir).with_suffix(".npy"))
        print(output_path)

        if os.path.exists(str(output_path)):
            print(str(output_path) + " already exists. Skipping.")
        else:
            if not str(input_path) == "data/audio/music_tags/electronic-neither-expressive-neither-neither-technological-neither-modern-neither-youthful/IT0311500117.ogg":
                # Decode, resample, convert to mono, and normalize audio
                with torch.no_grad():
                    representation = get_acts_from_file(
                        input_path, hps, vqvae, top_prior, meanpool=AVERAGE_SLICES
                    )
                # Reshape representation to a 1D array
                representation = representation.reshape(representation.shape[-1])

                # Save representation
                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(output_path, representation)

