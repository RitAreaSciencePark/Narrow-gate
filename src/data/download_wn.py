import nltk
import os
from datasets import load_dataset

# # Define your custom directory
# custom_nltk_dir = '/orfeo/scratch/dssc/zenocosini/nltk_data'

# # Create the directory if it doesn't exist
# os.makedirs(custom_nltk_dir, exist_ok=True)

# # Set the new NLTK data path
# nltk.data.path.append(custom_nltk_dir)

# nltk.download('wordnet', download_dir=custom_nltk_dir)
os.environ["HF_DATASETS_CACHE"] = "/orfeo/scratch/dssc/zenocosini/"
split = "validation"
dataset = load_dataset("ILSVRC/imagenet-1k", split=split, cache_dir="/orfeo/scratch/dssc/zenocosini/")
