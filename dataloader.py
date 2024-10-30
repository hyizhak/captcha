from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="processed_train", split=["train", "test"])  
dataset["train"][0]

dataset["train"][-1]