import torchopenl3
import soundfile as sf
import torch
import numpy as np
import argparse
import os
import pathlib
import librosa
from tqdm import tqdm

def audio_padding(audio, target_length):
    padding_length = target_length - audio.shape[0]
    padding_vector = np.zeros(padding_length)
    padded_audio = np.concatenate([audio, padding_vector], axis=0)
    return padded_audio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='read input and produce some output')

    parser.add_argument('songs_path', type=str,
                        help='Path to the raw audio directory')

    parser.add_argument('embeddings_path', type=str,
                        help='Path to the directory where the embeddings will be saved')

    args = parser.parse_args()

    folder_path_songs = pathlib.Path(args.songs_path)
    folder_path_embeddings = pathlib.Path(args.embeddings_path)
    file_names = sorted(list(folder_path_songs.glob("**/*.ogg")))#, reverse=True)
    SAMPLE_LENGTH = 1048576


    for file_name in tqdm(file_names):
        output_path = pathlib.Path(folder_path_embeddings, file_name.relative_to(folder_path_songs).with_suffix(".npy"))
        # Check if output parent directory already exists



        if not os.path.exists(output_path.parent):
            os.makedirs(output_path.parent)

        if os.path.exists(output_path):
            print("embedding already exists")

        else:
            audio, _ = sf.read(file_name, start=0, stop=SAMPLE_LENGTH) # Only read the first 1048576 samples of the audio file
            if audio.shape[0] < SAMPLE_LENGTH:
                audio = audio_padding(audio, SAMPLE_LENGTH)
            sr = 44100
            # audio = torch.from_numpy(audio).unsqueeze(0)
            # audio = audio.cpu().numpy()[0]

            emb, ts = torchopenl3.get_audio_embedding(torch.Tensor(audio), sr)

            # Max Pooling
            emb = torch.max(emb, dim=1)[0]

            # Get the right dimension
            emb = emb.reshape(emb.shape[-1])
            #print("emb shape: ", emb.shape)

            np.save(output_path, emb)