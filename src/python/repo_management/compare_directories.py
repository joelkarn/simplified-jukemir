import argparse
import pathlib


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from audio files.")

    parser.add_argument("input_dir", type=str, help="Path to feature parent directory.")
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)

    with open('definite_paths.txt', 'r') as f:
        input_paths = [line.strip() for line in f]

    npy_paths = sorted(list(input_dir.glob("**/*.npy")))
    # get rid of first part of npy_paths, which is data/jukebox_features/music_tags:
    npy_paths = [str(path)[len(args.input_dir + '/'):] for path in npy_paths]
    # compare npy_paths to input_paths
    count = 0
    for path in npy_paths:
        if path not in input_paths:
            count += 1
            print("NOT IN DEFINITE PATHS:")
            print(path)

    print(count)

