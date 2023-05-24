import os, glob
import json
import argparse
from random import sample
import pandas as pd
import pdb
import re

START_SYMBOL = "<s>"
END_SYMBOL = "</s>"
ROW_DIVIDER = "<r>"
NEXT_LINE = "<nl>"

def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]

class Preprocess_Data():

    def __init__(self, data_folder, labels_folder) -> None:

        self.to_json = {}

        for file in glob.glob(os.path.join(data_folder, "*/values.json")):
            self.to_json[file.split("/")[-2]] =  {"value_path": file}

        for file in glob.glob(os.path.join(data_folder, "*/label.json")):
            self.to_json[file.split("/")[-2]]["label_path"] = file

        for k, v in self.to_json.items():

            #CREATE LABEL
            label_path = os.path.join(labels_folder, k + ".csv")
            df = pd.read_csv(label_path, sep = ',', quotechar='"')
            #Do not add table titles
            #label = re.sub(rf"^(.+?)<r>", START_SYMBOL, re.sub(rf"{NEXT_LINE}(\s*?){ROW_DIVIDER}", NEXT_LINE , re.sub(r"\s{2,}", ROW_DIVIDER, df.to_string(index = False).replace('\n', NEXT_LINE))) + END_SYMBOL, count = 1).replace("<", " <").replace(">", "> ")
            #Adds titles
            label = START_SYMBOL + re.sub(rf"{NEXT_LINE}(\s*?){ROW_DIVIDER}", NEXT_LINE , re.sub(r"\s{2,}", ROW_DIVIDER, df.to_string(index = False).replace('\n', NEXT_LINE))) + END_SYMBOL

            #CREATE FEATURES
            self.to_json.append({"filename": label_path, "data": data, "label": label})

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
