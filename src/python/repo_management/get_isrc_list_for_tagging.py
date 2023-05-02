import numpy as np
import argparse
import pathlib
import random
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get equal amounts of samples from each tag.")
    parser.add_argument("input_dir", type=str, help="Path to parent directory of raw audio.")
    args = parser.parse_args()
    input_dir = pathlib.Path(args.input_dir)

    # all paths to ogg files, a path also contains the 10 different tags for that file
    file_paths = sorted(list(input_dir.glob("**/*.ogg")))

    # shuffle the paths
    random.seed(0)
    random.shuffle(file_paths)

    # tags dictionary, key is the path to the file, value is a list of 10 tags
    tags = {}

    for p in file_paths:
        tags[p] = os.path.split(p)[0].split('/')[-1].split('-')

    # count the number of occurrences of each tag in each dimension
    tag_counts = {}
    for dim in range(10):
        tag_counts[dim] = {}
        possible_tags = set([tags[p][dim] for p in tags])
        for tag in possible_tags:
            tag_counts[dim][tag] = 0


    for p in tags:
        for dim in range(10):
            tag = tags[p][dim]
            tag_counts[dim][tag] += 1

    # select 500 file paths for each tag in each dimension
    selected_file_paths = {}

    for p in tags:
        for dim in range(10):
            tag = tags[p][dim]
            if tag not in selected_file_paths:
                selected_file_paths[tag] = []
            if len(selected_file_paths[tag]) < 500:
                selected_file_paths[tag].append(p)

    # turn the dictionary into a list
    selected_file_paths_list = []
    for tag in selected_file_paths:
        selected_file_paths_list.extend(selected_file_paths[tag])

    # turn list into only unique paths
    selected_file_paths_list = list(set(selected_file_paths_list))
    print(len(selected_file_paths_list))

    # save the selected file paths to a text file
    with open('selected_file_paths.txt', 'w') as f:
        for path in selected_file_paths_list:
            f.write(str(path) + '\n')

