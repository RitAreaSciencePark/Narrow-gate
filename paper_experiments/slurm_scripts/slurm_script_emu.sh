#!/bin/sh
#SBATCH --job-name=gen_task
#SBATCH --partition=H100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=150G
#SBATCH --time=1-20:00:00
#SBATCH --gpus=1
#SBATCH -A lade
#SBATCH --output=/u/dssc/zenocosini/MultimodalInterp/paper_experiments/logs/bench_%j.out

SCRIPT="/u/dssc/zenocosini/MultimodalInterp/paper_experiments/5_AblationCaptioning.py"
 
OUT_PATH=".data/emu3/capt/"
export PYTHONPATH=$PYTHONPATH:/u/dssc/zenocosini/MultimodalInterp
cd /u/dssc/zenocosini/MultimodalInterp/paper_experiments

echo "Emu3-Gen-hf"
echo "#################################################"
echo "##### 5_AblationCaptioning.py #####"
echo "#################################################"
echo "# FLICKR"
poetry run python $SCRIPT -n 1000 -m "Emu3-Gen" -o $OUT_PATH
poetry run python $SCRIPT -n 1000 -m "Emu3-Gen"  -at "block-img-txt" --a @end-image-emu -o $OUT_PATH
poetry run python $SCRIPT -n 1000 -m "Emu3-Gen"  -a @end-image-emu  -o $OUT_PATH


echo "# COCO"
poetry run python $SCRIPT -n 1000 -d coco -m "Emu3-Gen" -o $OUT_PATH
poetry run python $SCRIPT -n 1000 -d coco -m "Emu3-Gen"  -at "block-img-txt" --a @end-image-emu -o $OUT_PATH
poetry run python $SCRIPT -n 1000 -d coco -m "Emu3-Gen"  -a @end-image-emu  -o $OUT_PATH

# poetry run python $SCRIPT -n 2000 -m "Emu3-Gen-hf"  -a @last-image  -o $OUT_PATH
# poetry run python $SCRIPT -n 2000 -m "Emu3-Gen-hf"  -a @random-image-10  -o $OUT_PATH

# echo "#################################################"
# echo "##### 4_AblationVQA.py #####"
# echo "#################################################"
# SCRIPT="/u/dssc/zenocosini/MultimodalInterp/paper_experiments/4_AblationVQA.py"
# OUT_PATH=".data/emu3/vqa/"
# poetry run python $SCRIPT -n 2000 -m "Emu3-Gen-hf"  -a @end-image-emu  -o $OUT_PATH
# poetry run python $SCRIPT -n 2000 -m "Emu3-Gen-hf" -o $OUT_PATH
# poetry run python $SCRIPT -n 2000 -m "Emu3-Gen-hf"  -at "block-img-txt" -a @end-image-emu -o $OUT_PATH
# poetry run python $SCRIPT -n 2000 -m "Emu3-Gen-hf" -d coco -a @end-image-emu -o $OUT_PATH
# poetry run python $SCRIPT -n 2000 -m "Emu3-Gen-hf" -d coco -o $OUT_PATH
# poetry run python $SCRIPT -n 2000 -m "Emu3-Gen-hf" -d coco -a @last-image -o $OUT_PATH
# poetry run python $SCRIPT -n 2000 -m "Emu3-Gen-hf" -d coco -a @random-image-10 -o $OUT_PATH
# poetry run python $SCRIPT -n 2000 -m "Emu3-Gen-hf" -d coco -a-type "block-img-txt" -o $OUT_PATH


# poetry run python $SCRIPT -n 2000 -m 'deepseek-ai/Janus-1.3B'  -a @last-image  -o $OUT_PATH
# poetry run python $SCRIPT -n 2000 -m 'deepseek-ai/Janus-1.3B'  -a @random-image-10  -o $OUT_PATH
# poetry run python 1_CosineSimilarity_and_Homogeneity.py -m "deepseek-ai/Janus-1.3B" -o "cos_sim/janus"
# poetry run python $SCRIPT -m 'deepseek-ai/Janus-1.3B' -o $OUT_PATH

