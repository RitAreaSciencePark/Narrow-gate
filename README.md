# The Narrow Gate: Localized Image-Text Communication in Vision-Language Models
![Screenshot from 2024-12-02 20-52-38](https://github.com/user-attachments/assets/0b2ba585-9d34-449c-8ba9-b15bc330f219)

This repo contains the code for replicating the experiment of the paper ***The Narrow Gate: Localized Image-Text Communication in Vision-Language Models***. 


## Setup
First, install the required packages:
```bash
git submodule update --init --recursive
poetry install
cd transformers
poetry install
```

## Reproducing the figures
The scripts for reproducing the figures are in the `paper_experiments` directory. To reproduce the figures, run the following command:
```bash
python 1_cosineSimilarity_and_Homogeneity.py -m facebook/chameleon-30b -o out_dir
python 2_CrossAttention.py -m facebook/chameleon-30b -o out_dir
python 3_overlap_sec_4-2.py -m facebook/chameleon-30b -o out_dir
python 3.1_ablation_sec_4-3.py -m facebook/chameleon-30b -o out_dir 
python 4_AblationVQA.py -m facebook/chameleon-30b -o out_dir -md zero-shot -n 2000 -a {@end-image,@random-image, @last-image}
python 5_AblationCaptioning.py -m facebook/chameleon-30b -o out_dir -d {coco, flickr} -n 2000 -a {@end-image,@random-image, @last-image}
python 6_ActivationPatching.py -m facebook/chameleon-30b -o out_dir
```

The figures will be saved in the `out_dir` directory.
