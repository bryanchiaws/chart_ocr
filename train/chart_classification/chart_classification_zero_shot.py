import clip
import pandas as pd
import argparse
import pdb
import os
import numpy as np
import glob
import tqdm
from PIL import Image
import numpy as np
import cv2
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def predict_type(image, text_inputs):
    with torch.no_grad():
        image_input = preprocess(image).unsqueeze(0).to(device)
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs.to(device))

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(1)

    return indices, values

if __name__ == '__main__':
    # usage: main.py [data visualizations folder][output_folder]
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs_folder')
    parser.add_argument('output_path')
    args = parser.parse_args()

    # Detect graph type using CLIP
    model, preprocess = clip.load("ViT-B/32", device=device)
    clip_labels = ['Pie chart', 'Line chart', 'Bar chart']
    labels_mapping = {i: x for i, x in enumerate(clip_labels)}
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in labels_mapping.items()]).to(device)

    images = os.listdir(args.inputs_folder)
    categories = []

    for image_path in tqdm.tqdm(images):
        path = os.path.join(args.inputs_folder, image_path)
        image = Image.open(path)
        indices, values = predict_type(image, text_inputs)
        categories.append(labels_mapping[indices.item()])

    df = pd.DataFrame({"imgs": images, "category": categories})
    df['index'] = [int(x.split('.')[0]) for x in df['imgs']]
    df = df.set_index('index', drop = True).sort_values('index')
    
    df.to_csv(args.output_path + "/chart_categories_test.csv")