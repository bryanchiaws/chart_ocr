import os
import json
import argparse
from random import sample

class Preprocess_Data():

    def __init__(self, title_folder, data_folder, caption_folder, multicol_title_folder = None, multicol_data_folder = None, multicol_caption_folder = None) -> None:

        self.titles = []
        self.data = []
        self.captions = []

        title_files = os.listdir(title_folder)

        N = len(title_files)

        self.to_json = []

        for i in range(1, N+1):

            title = open(title_folder + '/' + str(i) + ".txt", 'r').readline().strip()
            caption = open(caption_folder + '/' + str(i) + ".txt", 'r').readline().strip()

            data = []

            with open(data_folder + '/' + str(i) + ".csv", 'r') as f:
                for line in f:
                    data += line.strip().split(',')

            self.to_json.append({"filename": "two_col-" + str(i) + ".txt", "data": title + ' <s> ' + " ".join(data), "label": caption})

        if multicol_title_folder:
            title_files = os.listdir(multicol_title_folder)

            N = len(title_files)

            for i in range(1, N+1):

                title = open(multicol_title_folder + '/' + str(i) + ".txt", 'r').readline().strip()
                caption = open(multicol_caption_folder + '/' + str(i) + ".txt", 'r').readline().strip()

                data = []

                with open(multicol_data_folder + '/' + str(i) + ".csv", 'r') as f:
                    for line in f:
                        data += line.strip().split(',')

                self.to_json.append({"filename": "multi_col-" + str(i) + ".txt", "data": title + ' <s> ' + " ".join(data), "label": caption})

    def split_data(self, dataset_save_folder, save_folder, prop = 1) -> None:
        
        train_mapping = dataset_save_folder + '/train_index_mapping.csv'
        val_mapping = dataset_save_folder + '/val_index_mapping.csv'
        test_mapping = dataset_save_folder + '/test_index_mapping.csv'

        train_map = set()
        val_map = set()
        test_map = set()

        train_json = []
        val_json = []
        test_json = []

        with open(train_mapping, 'r') as f:
            for line in f:
                train_map.add(line.strip())

        with open(val_mapping, 'r') as f:
            for line in f:
                val_map.add(line.strip())

        with open(test_mapping, 'r') as f:
            for line in f:
                test_map.add(line.strip())

        for eg in self.to_json:
            if eg['filename'] in train_map:
                train_json.append({'data': eg['data'], 'label': eg['label']})
            elif eg['filename'] in val_map:
                val_json.append({'data': eg['data'], 'label': eg['label']})
            elif eg['filename'] in test_map:
                test_json.append({'data': eg['data'], 'label': eg['filename'] + "; " + eg['label']})

        if prop < 1:
            train_json = sample(train_json, int(prop*len(train_json)))
            val_json = sample(val_json, int(prop*len(val_json)))
            test_json = sample(test_json, int(prop*len(test_json)))

        # Serializing json
        train_json = json.dumps(train_json)
        val_json = json.dumps(val_json)
        test_json = json.dumps(test_json)
        
        # Writing to sample.json
        with open(save_folder + "/train.json", "w") as outfile:
            outfile.write(train_json)
        with open(save_folder + "/validation.json", "w") as outfile:
            outfile.write(val_json)
        with open(save_folder + "/test.json", "w") as outfile:
            outfile.write(test_json)

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t", 
        "--title_folder", 
        help="Path to directory containing titles.",
        required=True,
    )

    parser.add_argument(
        "-c", 
        "--captions_folder", 
        help="Path to directory containing captions.",
        required=True,
    )

    parser.add_argument(
        "-d", 
        "--data_folder", 
        help="Path to directory containing data of tables/graphs.",
        required=True,
    )

    parser.add_argument(
        "-mt", 
        "--multicol_title_folder", 
        help="Path to multicolumn directory containing titles.",
        required=False,
    )

    parser.add_argument(
        "-mc", 
        "--multicol_captions_folder", 
        help="Path to multicolumn directory containing captions.",
        required=False,
    )

    parser.add_argument(
        "-md", 
        "--multicol_data_folder", 
        help="Path to multicolumn directory containing data of tables/graphs.",
        required=False,
    )

    parser.add_argument(
        "-ds", 
        "--dataset_split_folder", 
        help="Path to files that have dataset split.",
        required=True,
    )

    parser.add_argument(
        "-s", 
        "--save_folder", 
        help="Path to save json files for train, validation, and test data.",
        required=True,
    )

    parser.add_argument(
        "-p", 
        "--prop", 
        help="Percentage of data to sample.",
        type = float,
        required=True,
    )

    return parser

def main(args: argparse.Namespace):
    data = Preprocess_Data(
        args.title_folder, 
        args.data_folder, 
        args.captions_folder,
        args.multicol_title_folder, 
        args.multicol_data_folder, 
        args.multicol_captions_folder,
    )

    data.split_data(args.dataset_split_folder, args.save_folder, args.prop)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)