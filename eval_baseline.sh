CHUNKS=30
NUM_PARALLEL=8
SAVE_PATH="exp_results/random_agent/stt"

IDX=0
while [ $IDX -lt $CHUNKS ]; do
    for ((i = 0; i < NUM_PARALLEL && IDX < CHUNKS; i++)); do
        echo "Launching job IDX=$IDX on GPU=$((IDX % NUM_PARALLEL))"
        CUDA_VISIBLE_DEVICES=$((i)) PYTHONPATH="habitat-lab" python run.py \
            --split-num $CHUNKS \
            --split-id $IDX \
            --exp-config 'habitat-lab/habitat/config/benchmark/nav/track/track_infer_stt.yaml' \
            --run-type 'eval' \
            --model-name 'baseline' \
            --save-path $SAVE_PATH &
        ((IDX++))
    done
    wait
done