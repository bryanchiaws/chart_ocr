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
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
from datasets import load_metric, Dataset, DatasetDict, load_dataset
import datasets
from transformers import TrainingArguments
from transformers import Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name_or_path = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path, device = device)

#Statista: 1-9999 Train, 10000-19999 Test, 20000- Val
#Pew: 1-199 Train, 200-1199 Test, 1200- Val
#MultiColumn: 1-1000 Train, 2000-4999 Test, 5000- Val

#Code adapted from: https://huggingface.co/blog/fine-tune-vit

def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor([x.convert('RGB') for x in example_batch['image']], return_tensors='pt')
    inputs['labels'] = example_batch['label']
    return inputs

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

metric = load_metric("accuracy")

def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

if __name__ == '__main__':
    # usage: python train/chart_classification/chart_classification_few_shot.py data/statista_sample_cat data/statista_sample_cat output/chart_preds
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs_folder')
    parser.add_argument('infer_folder')
    parser.add_argument('output_dir')
    args = parser.parse_args()

    ds = load_dataset("imagefolder", data_dir=args.inputs_folder)
    prepared_ds = ds.with_transform(transform)
    labels = prepared_ds['train'].features['label'].names

    #Train model
    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
    )

    training_args = TrainingArguments(
        output_dir="./vit-base-beans",
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=4,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to='tensorboard',
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=prepared_ds["train"],
        eval_dataset=prepared_ds["validation"],
        tokenizer=feature_extractor,
    )

    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    metrics = trainer.evaluate(prepared_ds['validation'])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    test_results = trainer.predict(prepared_ds["test"])
    predicted_class = np.argmax(test_results.predictions, axis=1).astype(int)

    #Save prediction results
    filename = [ds['test'][x]['image'].filename for x in range(len(ds['test']))]
    mapping = {i: labels[i] for i in range(len(labels))}
    results_df = pd.DataFrame({"filename": filename, "pred": [mapping[int(x)] for x in list(predicted_class)]}).sort_values('filename')
    results_df.to_csv(os.path.join(args.output_dir, "classification_results.csv"))

    #np.savetxt(args.output_dir + "/classification_results.txt", predicted_class)
