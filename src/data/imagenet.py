from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
from src.data.imagenet_graph import  SampledImageNetGraph, BalancedManualSampleImagenet
from src.data.imagenet_classes import IMAGENET2012_CLASSES
from easyroutine.logger import Logger
from tqdm import tqdm
import os
from src.utils import data_path
import random
import pandas as pd
from rich import print
import json
# set HF_CACHE to a custom directory

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
load_dotenv(dotenv_path="../.env")
from collections import defaultdict

def generate_random_dataset(
    graph: SampledImageNetGraph,
    max_samples_per_class:int,
    split: str = "train",
    seed: int = 42,
    num_proc: int = 10,
):
    # initialize the logger
    logger = Logger(logname="generate_dataset", log_file_path="logs.log")
    root_synset_name = graph._get_synset_from_offset(graph.root_nodes()[0]).name() # type: ignore
    # retrieve the dataset name
    dataset_name = f"imagenet_{root_synset_name}_{'tree' if graph.tree else 'graph'}_{split}_{max_samples_per_class}ImPerClass_{graph.n_imagenet_class}Class"

    # create mapping from label to id and viceversa (in the hf imagenet the labels are sorted by synset name, so we can use the index as id)
    id_to_label = {label: i for i, label in enumerate(IMAGENET2012_CLASSES)}
    label_to_id = {i: label for i, label in enumerate(IMAGENET2012_CLASSES)}

    class_count = {i: 0 for i in range(len(IMAGENET2012_CLASSES))}
    # get all the nodes in the graph
    all_nodes = list(graph.all_nodes())

    # check if in data/tmp folder there is already the dataset
    if os.path.exists(data_path(f"tmp/{dataset_name}")):
        logger.info(
            f"Dataset already present in data/tmp/{dataset_name}", std_out=True
        )
        dataset = load_from_disk(data_path(f"tmp/{dataset_name}/imagenet.arrow"))
        graph = SampledImageNetGraph.load(data_path(f"tmp/{dataset_name}/graph.nx"))
    else:
        # load the entire dataset
        dataset = load_dataset("ILSVRC/imagenet-1k", split=split)
        logger.info(f"Dataset loaded from ILSVRC/imagenet-1k", std_out=True)
        logger.info(
            f"Filtering dataset to keep only the nodes that are present in the graph. Number of nodes: {len(all_nodes)}. It can take a while",
            std_out=True,
        )
        
        # Step 1: Filter to include only the desired classes
        dataset = dataset.filter(lambda x: label_to_id[x["label"]] in all_nodes, num_proc=num_proc)
        
        def sample_dataset(dataset, graph, max_samples_per_class, seed, num_proc):
            random.seed(seed)
            class_samples = defaultdict(list)
            sampled_classes = set(graph.sampled_classes)
            
            # Shuffle the dataset first
            shuffled_dataset = dataset.shuffle(seed=seed)
            
            # Calculate shard size
            shard_size = max(1000, len(shuffled_dataset) // (num_proc * 10))  # Adjust as needed
            
            for shard in tqdm(shuffled_dataset.iter(batch_size=shard_size), total=len(shuffled_dataset)//shard_size, desc="Processing shards"):
                for id, label in enumerate(shard["label"]):
                    class_id = label_to_id[label]
                    if class_id in sampled_classes and len(class_samples[class_id]) < max_samples_per_class:
                        class_samples[class_id].append({"image": shard["image"][id], "label": label})
                
                if all(len(samples) >= max_samples_per_class for class_id, samples in class_samples.items() if class_id in sampled_classes):
                    break
            
            # Combine samples from all classes
            combined_samples = [sample for samples in class_samples.values() for sample in samples]
            
            return Dataset.from_list(combined_samples)

        combined_samples = sample_dataset(dataset, graph, max_samples_per_class, seed, num_proc)
        dataset = combined_samples

        # # Step 2: Sample from each class
        # sampled_datasets = []
        # for class_id in tqdm(graph.sampled_classes, desc="Sampling for each class"):
        #     class_dataset = dataset.filter(lambda x: label_to_id[x["label"]] == class_id, num_proc=num_proc)
        #     sampled_class_dataset = class_dataset.shuffle(seed=seed).select(range(min(max_samples_per_class, len(class_dataset))))
        #     sampled_datasets.append(sampled_class_dataset)

        # # Combine the sampled datasets
        # dataset = concatenate_datasets(sampled_datasets)

        # save the filtered dataset

        dataset.save_to_disk(data_path(f"tmp/{dataset_name}/imagenet.arrow"), num_proc=num_proc)
        graph.save(data_path(f"tmp/{dataset_name}/graph.nx"))
        logger.info(
            f"Dataset saved to {data_path(f'tmp/{dataset_name}')}", std_out=True
        )

    # add the synset and description columns
    mapping_dict = graph.get_mapping_dict()
    def get_info(x):
        return {"offset": label_to_id[x["label"]],
        "synset": mapping_dict[label_to_id[x["label"]]]["synset_name"],
        "definition": mapping_dict[label_to_id[x["label"]]]["definition"],
        "lemma_names": mapping_dict[label_to_id[x["label"]]]["lemma_names"],
        }
        

    dataset = dataset.map(
       get_info,
        num_proc=num_proc,
    )

    # save the dataset
    dataset.save_to_disk(data_path(f"datasets/{dataset_name}/imagenet.arrow"))
    #save the graph
    graph.save(data_path(f"datasets/{dataset_name}/graph.nx"))
    metadata = {
        "dataset_name": dataset_name,
        "max_sample_per_class": max_samples_per_class,
        "n_class": graph.n_imagenet_class,
    }
    json.dump(metadata, open(data_path(f"datasets/{dataset_name}/metadata.json"), "w"))
    
    return dataset
    
    
def generate_dataset_with_selected_class(
    selected_classes: dict,
    num_proc: int = 10,
    split: str = "train",
    concat_text: str | None = None,
    save: bool = True,
    push_to_hub: bool = False,
):
    
    # initialize the logger
    logger = Logger(logname="generate_dataset", log_file_path="logs.log")
   
    # retrieve the dataset name
    number_of_class = len(selected_classes.keys())
    number_img_per_class = selected_classes[list(selected_classes.keys())[0]] # we assume that all the classes have the same number of images
    # dataset_name = f"imagenet_manual_sample_{split}_alligator_crocodile"
    # dataset_name = f"imagenet_short_text_100_classes_x_100_samples"
    dataset_name = f"imagenet_text_50_classes_x_50_samples"

    # create mapping from label to id and viceversa (in the hf imagenet the labels are sorted by synset name, so we can use the index as id)
    id_to_label = {label: i for i, label in enumerate(IMAGENET2012_CLASSES)}
    label_to_id = {i: label for i, label in enumerate(IMAGENET2012_CLASSES)}



    # #check if in data/tmp folder there is already the dataset
    # if os.path.exists(data_path(f"tmp/{dataset_name}")) and save is True:
    #     logger.info(
    #         f"Dataset already present in data/tmp/{dataset_name}", std_out=True
    #     )
    #     dataset = load_from_disk(data_path(f"tmp/{dataset_name}/imagenet.arrow"))
    #     graph = BalancedManualSampleImagenet.load(data_path(f"tmp/{dataset_name}/graph.nx"))
    if True:
    # else:
        # load the entire dataset
        graph = BalancedManualSampleImagenet(high_level_classes=selected_classes)
        dataset = load_dataset("ILSVRC/imagenet-1k", split=split, num_proc=num_proc)
        # get all the nodes in the graph
        all_nodes = list(graph.all_nodes())
        
        logger.info(f"Dataset loaded from ILSVRC/imagenet-1k", std_out=True)
        logger.info(
            f"Filtering dataset to keep only the nodes that are present in the graph. Number of nodes: {len(all_nodes)}. It can take a while",
            std_out=True,
        )
        

        # Step 1: Filter to include only the desired classes
        dataset = dataset.filter(lambda x: label_to_id[x["label"]] in all_nodes, num_proc=num_proc)
        image_count_per_class = graph.get_leaf_image_counts(return_offset=True)
        
        def sample_dataset(dataset, graph, num_proc):
            class_samples = defaultdict(list)
            count_sampled_classes = {}

            
            # Calculate shard size
            shard_size = max(1000, len(dataset) // (num_proc * 10))  # Adjust as needed
            
            for shard in tqdm(dataset.iter(batch_size=shard_size), total=len(dataset)//shard_size, desc="Processing shards"):
                for id, label in enumerate(shard["label"]):
                    class_id = label_to_id[label]
                    if class_id in image_count_per_class:
                        if class_id not in count_sampled_classes:
                            count_sampled_classes[class_id] = 0
                        if count_sampled_classes[class_id] < image_count_per_class[class_id]:
                            class_samples[class_id].append({"image": shard["image"][id], "label": label})
                            count_sampled_classes[class_id] += 1

            
            # Combine samples from all classes
            combined_samples = [sample for samples in class_samples.values() for sample in samples]
            
            return Dataset.from_list(combined_samples)

        combined_samples = sample_dataset(dataset, graph, num_proc)
        dataset = combined_samples

        # # Step 2: Sample from each class
        # sampled_datasets = []
        # for class_id in tqdm(graph.sampled_classes, desc="Sampling for each class"):
        #     class_dataset = dataset.filter(lambda x: label_to_id[x["label"]] == class_id, num_proc=num_proc)
        #     sampled_class_dataset = class_dataset.shuffle(seed=seed).select(range(min(max_samples_per_class, len(class_dataset))))
        #     sampled_datasets.append(sampled_class_dataset)

        # # Combine the sampled datasets
        # dataset = concatenate_datasets(sampled_datasets)

        # save the filtered dataset
        if save:
            dataset.save_to_disk(data_path(f"tmp/{dataset_name}/imagenet.arrow"), num_proc=3)
            graph.save(data_path(f"tmp/{dataset_name}/graph.nx"))
            logger.info(
                f"tmp dataset saved to {data_path(f'tmp/{dataset_name}')}", std_out=True
            )

    # add the synset and description columns
    mapping_dict = graph.get_mapping_dict()
    def get_info(x):
        return {"offset": label_to_id[x["label"]],
        "synset": mapping_dict[label_to_id[x["label"]]]["synset_name"],
        "definition": mapping_dict[label_to_id[x["label"]]]["definition"],
        "lemma_names": mapping_dict[label_to_id[x["label"]]]["lemma_names"],
        }
        

    dataset = dataset.map(
       get_info,
        num_proc=num_proc,
    )

    if concat_text is not None:
        print("I'm here")
        def concat_text_fn(x):
            x["text"] = concat_text
            return x
        dataset = dataset.map(
            concat_text_fn,
            num_proc=num_proc,
        )
        
    def add_root_label(x):
        x["root_label"] = graph.map_leaf_to_high_level(leaf_offset=x["offset"])
        return x
        
    dataset = dataset.map(
        add_root_label,
        num_proc=1
    )



    # save the dataset
    if save:
        path_of_dataset = data_path(f"datasets/{dataset_name}/imagenet{'-text' if concat_text else ''}.arrow")
        dataset.save_to_disk(path_of_dataset)
        logger.info(
            f"Dataset saved to {data_path(f'datasets/{dataset_name}')}", std_out=True
        )
        #save the graph
        graph.save(data_path(f"datasets/{dataset_name}/graph.nx"))
        metadata = {
            "dataset_name": dataset_name,
            "max_sample_per_class": number_img_per_class,
            "n_class": graph.n_imagenet_class,
            "selected_classes": selected_classes,
            "graph_type": "BalancedManualSampleImagenet",
            "time": pd.Timestamp.now().isoformat()
        }
        json.dump(metadata, open(data_path(f"datasets/{dataset_name}/metadata.json"), "w"))

    if push_to_hub:
        dataset.push_to_hub(f"francescortu/{dataset_name}")
    
    return dataset
    
    

