import argparse
import pathlib
import random
import os

random.seed(0)

if __name__ == "__main__":
    with open('real_definite_paths.txt', 'r') as f:
        already_selected = [pathlib.Path(line.strip()) for line in f]

    with open('test_paths.txt', 'r') as f:
        big_list = [pathlib.Path(line.strip()) for line in f]

    # parser = argparse.ArgumentParser(description="Get equal amounts of samples from each tag.")
    # parser.add_argument("input_dir", type=str, help="Path to parent directory of raw audio.")
    # args = parser.parse_args()
    # input_dir = pathlib.Path(args.input_dir)
    # #path_not_to_include = "data/audio/music_tags/electronic-neither-expressive-neither-neither-technological-neither-modern-neither-youthful/IT0311500117.ogg"
    #
    #
    # # all paths to ogg files, a path also contains the 10 different tags for that file
    # file_paths = sorted(list(input_dir.glob("**/*.ogg")))

    count = 0
    for p in already_selected:
        if p in big_list:
            count += 1
    print(count, len(already_selected))
    # shuffle the paths

    #random.shuffle(file_paths)

    # tags dictionary, key is the path to the file, value is a list of 10 tags
    tags = {}

    for p in big_list:
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

    print(tag_counts)
    # # make a list of paths, where 1201 samples are selected from each tag
    # selected_paths = []
    # # add the already selected paths
    # selected_paths.extend(already_selected)
    #
    # # Initialize selected_tag_counts dictionary similar to tag_counts
    # selected_tag_counts = {}
    # for dim in range(10):
    #     selected_tag_counts[dim] = {}
    #     possible_tags = set([tags[p][dim] for p in tags])
    #     for tag in possible_tags:
    #         selected_tag_counts[dim][tag] = 0
    #
    # # Keep track of the number of selected songs for each tag
    # for p in already_selected:
    #     for dim in range(10):
    #         tag = tags[p][dim]
    #         selected_tag_counts[dim][tag] += 1
    # print(selected_tag_counts)
    # # Iterate through the shuffled list of songs
    # for p in file_paths:
    #     # Check if this song has already been selected
    #     if p in already_selected:
    #         continue
    #
    #     # Check if we have already reached the limit for each tag
    #     can_select = True
    #     for dim in range(10):
    #         tag = tags[p][dim]
    #         if selected_tag_counts[dim][tag] >= 1201:
    #             can_select = False
    #             break
    #         else:
    #             print("here we have something")
    #
    #     # If we haven't reached the limit, select this song and increase the count
    #     if can_select:
    #         selected_paths.append(p)
    #         for dim in range(10):
    #             tag = tags[p][dim]
    #             selected_tag_counts[dim][tag] += 1
    #
    # print(selected_tag_counts)

    #
    # # save the selected file paths to a text file
    # with open('really_real_new_selected_paths.txt', 'w') as f:
    #     for path in already_selected:
    #         f.write(str(path) + '\n')

