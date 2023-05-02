import argparse
import pathlib
import os
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='copies files from one directory to another')

    parser.add_argument('isrc_dir', type=str,
                        help='Path to the directory where isrc are stored')

    parser.add_argument('sources_dir', type=str,
                        help='Path to all files')

    parser.add_argument('dest_dir', type=str,
                        help='Path to the directory where 100 copies will be stored')




    args = parser.parse_args()

    source_path = pathlib.Path(args.sources_dir)
    dest_path = pathlib.Path(args.dest_dir)
    isrc_path = pathlib.Path(args.isrc_dir)

    # get the isrc from the paths below by splitting the path and taking the last element
    # then remove the extension
    # also save the genre of the file, so that we can copy 100 files from each genre into the genre directory of the destination directory
    isrcs = []
    genres = {}
    for file_name in isrc_path.glob("**/*.ogg"):
        isrc = file_name.parts[-1].split(".")[0]
        genre = file_name.parts[-2]
        isrcs.append(isrc)
        genres[isrc] = genre

    # copy the files (with isrcs as file names) from the source directory to the destination directory
    for isrc in isrcs:
        file_name = isrc + ".npy"
        genre = genres[isrc]
        genre_path = dest_path / genre
        if not genre_path.exists():
            os.makedirs(genre_path)
        #print(str(source_path / genre / file_name), genre_path)
        os.system("cp " + str(source_path / genre / file_name) + " " + str(genre_path))

