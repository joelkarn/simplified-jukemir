import argparse
import pathlib


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from audio files.")

    parser.add_argument("input_dir", type=str, help="Path to feature parent directory.")
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)

    npy_paths = sorted(list(input_dir.glob("**/*.npy")))
    # get rid of first part of npy_paths, which is data/jukebox_features/music_tags:
    npy_paths = [str(path)[len('data/jukebox_features/music_tags/'):] for path in npy_paths]
    print(len(npy_paths))
    # add data/audio/music_tags/ to the beginning of each path
    npy_paths = ['data/lyrics/music_tags/' + path for path in npy_paths]
    # change from .npy to .ogg
    npy_paths = [path[:-4] + '.txt' for path in npy_paths]
    # save the selected file paths to a text file
    with open('lyrics_definite_paths.txt', 'w') as f:

        for path in npy_paths:
            f.write(str(path) + '\n')

