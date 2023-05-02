import argparse
import pathlib
import random
import os


if __name__ == "__main__":
    with open('old_new_new_selected_paths.txt', 'r') as f:
        already_selected = [pathlib.Path(line.strip()) for line in f]
    print(len(already_selected))
    neither_dim_5_paths_count = 210

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


    for p in tags:
        dim = 5
        tag = tags[p][dim]
        if tag == 'technological':
            "YES"
            if neither_dim_5_paths_count < 501:
                if p not in already_selected:
                    neither_dim_5_paths_count += 1
                    already_selected.append(p)


    # save the selected file paths to a text file
    with open('new_selected_paths.txt', 'w') as f:
        for path in already_selected:
            f.write(str(path) + '\n')

