import numpy as np
import torch
import os
from PIL import Image
from datasets import load_dataset
from datasets import load_metric
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer, pipeline
from torchvision.transforms import Compose, Normalize, ToTensor

os.environ["HF_HOME"] = "/scratch/e1310988"
os.environ['HF_DATASETS_CACHE'] = "/scratch/e1310988"

def fine_tune(model_checkpoint="vit-large-patch16-224-in21k-finetuned-captcha", batch_size=32):

    dataset = load_dataset("imagefolder", data_dir="processed_train_contours", drop_labels=False)
    metric = load_metric("accuracy")

    labels = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    processor = AutoImageProcessor.from_pretrained(model_checkpoint)

    transforms = Compose([ToTensor(), Normalize(mean=processor.image_mean, std=processor.image_std)])

    def preprocess_images(examples):
        examples['pixel_values'] = [transforms(image.convert("RGB")) for image in examples["image"]]
        return examples

    splits = dataset["train"].train_test_split(test_size=0.1)
    train_ds = splits["train"]
    val_ds = splits["test"]

    train_ds.set_transform(preprocess_images)
    val_ds.set_transform(preprocess_images)

    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes = True, 
    )

    model_name = model_checkpoint.split("/")[-1]

    args = TrainingArguments(
        f"{model_name}",
        remove_unused_columns=False,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=20,
        warmup_ratio=0.1,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    train_results = trainer.train(resume_from_checkpoint=True)
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    return model

def inference(repo_name, image_path):
    image_processor = AutoImageProcessor.from_pretrained(repo_name)
    model = AutoModelForImageClassification.from_pretrained(repo_name)
    pipe = pipeline("image-classification", 
            model=model,
            feature_extractor=image_processor)

    image = Image.open(image_path)

    encoding = image_processor(images=image, return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    predicted_class_idx = torch.argmax(logits).item()
    predicted_class = model.config.id2label[predicted_class_idx]

    return predicted_class

if __name__ == "__main__":
    model = fine_tune()

