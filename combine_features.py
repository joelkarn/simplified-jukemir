import argparse
import pathlib
import os
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='combines two sets of features')

    parser.add_argument('audio_dir', type=str,
                        help='audio features directory')

    parser.add_argument('lyric_dir', type=str,
                        help='lyric features directory')

    parser.add_argument('output_dir', type=str,
                        help='audio and lyric feaures combined directory')




    args = parser.parse_args()

    audio_dir = pathlib.Path(args.audio_dir)
    lyric_dir = pathlib.Path(args.lyric_dir)
    output_dir = pathlib.Path(args.output_dir)

    audio_features = sorted(list(audio_dir.glob("**/*.npy")))
    lyric_features = sorted(list(lyric_dir.glob("**/*.npy")))

    for audio_feature, lyric_feature in zip(audio_features, lyric_features):
        print(audio_feature, lyric_feature)
        audio = np.load(audio_feature)
        lyric = np.load(lyric_feature)
        combined = np.concatenate((audio, lyric))
        # save the combined features to the output directory with the same path as the audio features but replace audio_dir with output_dir
        output_path = output_dir / audio_feature.relative_to(audio_dir)
        if not output_path.parent.exists():
            os.makedirs(output_path.parent)
        np.save(output_path, combined)

