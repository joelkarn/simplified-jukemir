import argparse
import pathlib
import os
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='copies files from one directory to another')

    parser.add_argument('sources_dir', type=str,
                        help='Path to all files')

    parser.add_argument('dest_dir', type=str,
                        help='Path to the directory where 100 copies will be stored')
    parser.add_argument('extension', type=str,
                        help='extension of files to copy')

    args = parser.parse_args()

    source_path = pathlib.Path(args.sources_dir)
    dest_path = pathlib.Path(args.dest_dir)
    file_extension = args.extension

    file_names = sorted(list(source_path.glob("**/*." + file_extension)))


    # divide files into genres, example of a file path is 'data/audio_features/genre/Rnb/USSM11701732.npy'
    # the genre is the second to last folder in the path
    genres = {}

    for file_name in file_names:
        genre = file_name.parts[-2]
        if genre not in genres:
            genres[genre] = []
        genres[genre].append(file_name)

    # copy 100 files from each genre to the destination directory
    for genre in genres:
        genre_path = dest_path / genre
        if not genre_path.exists():
            os.makedirs(genre_path)
        # randomize the order of the files
        np.random.shuffle(genres[genre])
        for file_name in genres[genre][:100]:
            #print(file_name, genre_path)
            os.system("cp " + str(file_name) + " " + str(genre_path))

