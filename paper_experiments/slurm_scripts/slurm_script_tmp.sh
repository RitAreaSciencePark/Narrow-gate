#!/bin/sh
#SBATCH --job-name=gen_task
#SBATCH --partition=H100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=1-20:00:00
#SBATCH --gpus=1
#SBATCH -A lade
#SBATCH --output=/u/dssc/zenocosini/MultimodalInterp/paper_experiments/logs/bench_%j.out

SCRIPT="/u/dssc/zenocosini/MultimodalInterp/paper_experiments"

# MODEL="chameleon-7b"
# MODEL="chameleon-30b"
# MODEL="pixtral-12b"
# MODEL="Emu3-Gen"
# MODEL="janus"
# MODEL="llava-onevision-7b" 

# MODEL="deepseek-ai/Janus-1.3B"
# OUT_PATH=".data/janus"
# llava-onevision-7b
# "facebook/chameleon-7b"
# deepseek-ai/Janus-1.3B
export PYTHONPATH=$PYTHONPATH:/u/dssc/zenocosini/MultimodalInterp
cd /u/dssc/zenocosini/MultimodalInterp/paper_experiments

# echo "#################################################"
# echo "##### 3.1_overlap_ablation #####"
# echo "#################################################"
# # poetry run python $SCRIPT/3.1_overlap_ablation.py -m $MODEL -o $OUT_PATH/3_1_overlap_ablation

# echo "#################################################"
# echo "##### 2_cross_attention #####"
# echo "#################################################"
# poetry run python $SCRIPT/2_CrossAttention.py -m $MODEL -o $OUT_PATH/2_cross_attention --save

echo "#################################################"
echo "##### 5_AblationCaptioning #####"
echo "#################################################"

MODEL = "chameleon-7b"
echo "#################################################"
echo model: $MODEL
echo "#################################################"
OUT_PATH=".data/$MODEL"
poetry run python $SCRIPT/5_AblationCaptioning.py -n 1000 -m $MODEL -d coco -a @32 -o $OUT_PATH
poetry run python $SCRIPT/5_AblationCaptioning.py -n 1000 -m $MODEL -d flickr -a @32 -o $OUT_PATH

MODEL = "chameleon-30b"
echo "#################################################"
echo model: $MODEL
echo "#################################################"
OUT_PATH=".data/$MODEL"
poetry run python $SCRIPT/5_AblationCaptioning.py -n 1000 -m $MODEL -d coco -a @first-image -o $OUT_PATH
poetry run python $SCRIPT/5_AblationCaptioning.py -n 1000 -m $MODEL -d flickr -a @first-image -o $OUT_PATH


MODEL = "pixtral-12b"
echo "#################################################"
echo model: $MODEL
echo "#################################################"
OUT_PATH=".data/$MODEL"
poetry run python $SCRIPT/5_AblationCaptioning.py -n 1000 -m $MODEL -d coco -a @1025 -o $OUT_PATH
poetry run python $SCRIPT/5_AblationCaptioning.py -n 1000 -m $MODEL -d flickr -a @1025 -o $OUT_PATH





# poetry run python $SCRIPT -n 2000 -m 'deepseek-ai/Janus-1.3B'  -a @last-image  -o $OUT_PATH
# poetry run python $SCRIPT -n 2000 -m 'deepseek-ai/Janus-1.3B'  -a @random-image-10  -o $OUT_PATH
# poetry run python 1_CosineSimilarity_and_Homogeneity.py -m "deepseek-ai/Janus-1.3B" -o "cos_sim/janus"
# poetry run python $SCRIPT -m 'deepseek-ai/Janus-1.3B' -o $OUT_PATH

