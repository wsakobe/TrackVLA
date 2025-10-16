CHUNKS=30
NUM_PARALLEL=8
SAVE_PATH="exp_results/uninavid_all/at1"
MODEL_PATH="model_zoo/llama-vid-7b-full-224-video-fps-1-grid-2-panda-encoder-2025-10-10-all-data"

IDX=0
while [ $IDX -lt $CHUNKS ]; do
    for ((i = 0; i < NUM_PARALLEL && IDX < CHUNKS; i++)); do
        echo "Launching job IDX=$IDX on GPU=$((IDX % NUM_PARALLEL))"
        CUDA_VISIBLE_DEVICES=$((i)) PYTHONPATH="habitat-lab" python run.py \
            --split-num $CHUNKS \
            --split-id $IDX \
            --exp-config 'habitat-lab/habitat/config/benchmark/nav/track/track_infer_at.yaml' \
            --run-type 'eval' \
            --save-path $SAVE_PATH \
            --model-path $MODEL_PATH \
            --model-name 'uni-navid' &
        ((IDX++))
    done
    wait
done