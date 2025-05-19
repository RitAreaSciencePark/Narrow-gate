#!/bin/sh
#SBATCH --job-name=gen_task
#SBATCH --partition=H100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=10:00:00
#SBATCH --gpus=2
#SBATCH -A lade
#SBATCH --output=/u/dssc/zenocosini/MultimodalInterp/paper_experiments/logs/bench_%j.out

SCRIPT="/u/dssc/zenocosini/MultimodalInterp/paper_experiments"
export PYTHONPATH=$PYTHONPATH:/u/dssc/zenocosini/MultimodalInterp
cd /u/dssc/zenocosini/MultimodalInterp/paper_experiments


# MODEL="chameleon-7b"
# MODEL="chameleon-30b"
# MODEL="pixtral-12b"
# MODEL="Emu3-Gen"
# MODEL="janus"
# MODEL="llava-onevision-7b" 
MODEL="vila-u"
# MODEL="BAAI/Emu2"
OUT_PATH=".data"

poetry run python /u/dssc/zenocosini/MultimodalInterp/tests/old_test/tmp.py
# echo "#################################################"
# echo "##### Fig2_CosSim_Hom #####"
# echo "#################################################"
# poetry run python $SCRIPT/Fig2_CosSim_Hom.py \
#                             -m $MODEL \
#                             -o $OUT_PATH/Fig2_CosSim_Hom/$MODEL

# echo "#################################################"
# echo "##### Fig3_CrossAttention #####"
# echo "#################################################"
# poetry run python $SCRIPT/Fig3_CrossAttention.py \
#                             -m $MODEL \
#                             -o $OUT_PATH/Fig3_CrossAttention/$MODEL \
#                             -s


# echo "#################################################"
# echo "##### Fig4_OverlapImagenet #####"
# echo "#################################################"
# poetry run python $SCRIPT/Fig4_OverlapImagenet.py -m $MODEL -o $OUT_PATH/Fig4_OverlapImagenet

# echo "#################################################"
# echo "##### 2_cross_attention #####"
# echo "#################################################"
# poetry run python $SCRIPT/2_CrossAttention.py -m $MODEL -o $OUT_PATH/2_cross_attention --save

# echo "#################################################"
# echo "##### Tab1_AblationCaptioning #####"
# echo "#################################################"
# poetry run python $SCRIPT/Tab1_AblationCaptioning.py -n 1000 -m $MODEL -d coco -a @end-image -o $OUT_PATH/Tab1_AblationCaptioning/$MODEL
# # poetry run python $SCRIPT/Tab1_AblationCaptioning.py -n 1000 -m $MODEL -d coco -a @last-image-2 -o $OUT_PATH/Tab1_AblationCaptioning/$MODEL
# poetry run python $SCRIPT/Tab1_AblationCaptioning.py -n 1000 -m $MODEL -d coco  -o $OUT_PATH/Tab1_AblationCaptioning/$MODEL
# poetry run python $SCRIPT/Tab1_AblationCaptioning.py -n 1000 -m $MODEL -d coco -a-type "block-img-txt" -o $OUT_PATH/Tab1_AblationCaptioning/$MODEL

# poetry run python $SCRIPT/Tab1_AblationCaptioning.py -n 1000 -m $MODEL -d flickr -a @end-image -o $OUT_PATH/Tab1_AblationCaptioning/$MODEL
# # poetry run python $SCRIPT/Tab1_AblationCaptioning.py -n 1000 -m $MODEL -d flickr -a @last-image-2 -o $OUT_PATH/Tab1_AblationCaptioning/$MODEL
# poetry run python $SCRIPT/Tab1_AblationCaptioning.py -n 1000 -m $MODEL -d flickr -o $OUT_PATH/Tab1_AblationCaptioning/$MODEL
# poetry run python $SCRIPT/Tab1_AblationCaptioning.py -n 1000 -m $MODEL -d flickr -a-type "block-img-txt" -o $OUT_PATH/Tab1_AblationCaptioning/$MODEL

# poetry run python $SCRIPT/Tab1_AblationCaptioning.py -n 1000 -m $MODEL -d coco -o $OUT_PATH
# poetry run python $SCRIPT/Tab1_AblationCaptioning -n 1000 -m $MODEL -d coco -a @last-image -o $OUT_PATH
# poetry run python $SCRIPT/Tab1_AblationCaptioning.py -n 1000 -m $MODEL -d coco -a @random-image-10 -o $OUT_PATH
# poetry run python $SCRIPT/Tab1_AblationCaptioning.py -n 1000 -m $MODEL -d coco -a-type "block-img-txt" -o $OUT_PATH

# poetry run python $SCRIPT/Tab1_AblationCaptioning.py -n 1000 -m $MODEL -d flickr -a @end-image-emu -o $OUT_PATH
# poetry run python $SCRIPT/Tab1_AblationCaptioning.py -n 1000 -m $MODEL -d flickr -o $OUT_PATH
# poetry run python $SCRIPT/Tab1_AblationCaptioning.py -n 1000 -m $MODEL -d flickr -a @last-image -o $OUT_PATH
# poetry run python $SCRIPT/Tab1_AblationCaptioning.py -n 1000 -m $MODEL -d flickr -a @random-image-10 -o $OUT_PATH
# poetry run python $SCRIPT/Tab1_AblationCaptioning.py -n 1000 -m $MODEL -d flickr -a-type "block-img-txt" -o $OUT_PATH


# poetry run python $SCRIPT -n 2000 -m 'deepseek-ai/Janus-1.3B'  -a @last-image  -o $OUT_PATH
# poetry run python $SCRIPT -n 2000 -m 'deepseek-ai/Janus-1.3B'  -a @random-image-10  -o $OUT_PATH
# poetry run python 1_CosineSimilarity_and_Homogeneity.py -m "deepseek-ai/Janus-1.3B" -o "cos_sim/janus"
# poetry run python $SCRIPT -m 'deepseek-ai/Janus-1.3B' -o $OUT_PATH

# echo "#################################################"
# echo "##### Tab1_AblationCaptioning #####"
# echo "#################################################"
poetry run python $SCRIPT/Tab1_AblationVqQA.py -m $MODEL -o $OUT_PATH/Tab1_AblationVQA/$MODEL
poetry run python $SCRIPT/Tab1_AblationVqQA.py -m $MODEL -o $OUT_PATH/Tab1_AblationVQA/$MODEL -a @end-image
poetry run python $SCRIPT/Tab1_AblationVqQA.py -m $MODEL -o $OUT_PATH/Tab1_AblationVQA/$MODEL -a @last-image-2
poetry run python $SCRIPT/Tab1_AblationVqQA.py -m $MODEL -o $OUT_PATH/Tab1_AblationVQA/$MODEL -a-type "block-img-txt"

# MODE="end-image"
# MODE="image-text"
# poetry run python /u/dssc/zenocosini/MultimodalInterp/paper_experiments/3_overlap_sec_4-2.py --mode $MODE -m Emu3-Gen-Finetune -o .data/3_overlap_sec_4-2/$MODE



