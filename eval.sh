CHUNKS=8
SAVE_PATH="exp_results/random_agent/stt" 

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo $(( (IDX) % 8 ))
    CUDA_VISIBLE_DEVICES=$(( (IDX) % 8 )) PYTHONPATH="habitat-lab" python run.py \
    --split-num $CHUNKS \
    --split-id $IDX \
    --exp-config 'habitat-lab/habitat/config/benchmark/nav/track/track_infer_stt.yaml'\
    --run-type 'eval' \
    --save-path $SAVE_PATH &
    
done

wait