uv run python scripts/enrich_sft_nosearch.py \
    --uncertainty-json results/musique_parametric/musique_parametric_uncertainty_nemotron-3-nano_30b.json \
    --output-dir data/sft/musique_nosearch \
    --lookback-days 90 \
    --output-traces data/sft/musique_nosearch/raw_parametric_traces.json

uv run python scripts/curate_onpolicy_sft.py \
    --input-jsonl data/sft/musique_onpolicy/procedure1_onpolicy_sft.jsonl \
    --uncertainty-json results/musique_parametric/musique_parametric_uncertainty_nemotron-3-nano_30b.json \
    --natural-jsonl data/musique_train_natural.jsonl \
    --output-jsonl data/sft/musique_onpolicy/procedure1_onpolicy_sft_curated.jsonl

TORCH_EXTENSIONS_DIR=/work/torch_extensions \
CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch \
    --config_file configs/accelerate_zero3.yaml \
    --num_processes 2 \
    scripts/train_sft.py \
    --model-name nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --target-modules q_proj,k_proj,v_proj,o_proj,lm_head \
    --data-dir data/sft/musique_onpolicy \
    --extra-data data/sft/musique_nosearch/nosearch_sft.jsonl \
    --augment-formal-questions \
    --output-dir /work/models/nemotron-musique-lora-v3-aug \
    --hf-cache-dir /work/hf_cache \
    --deepspeed configs/deepspeed_zero2_2gpu.json \
    --batch-size 4 --grad-accum 4 \
    --epochs 3 \
    --learning-rate 1e-4 \
    --lora-rank 64 --lora-alpha 128 \
    --warmup-ratio 0.03 --save-steps 36 --logging-steps 5 \
    --logging-steps 1 > train.log 2>&1 &

PYTHONUNBUFFERED=1 HF_HOME=/work/hf_cache CUDA_VISIBLE_DEVICES=0,1 \
uv run python scripts/merge_lora.py \
    --adapter-dir /work/models/nemotron-musique-lora-v3-aug/checkpoint-108 \
    --output-dir  /work/models/nemotron-musique-lora-v3-aug_merged \
    --hf-cache-dir /work/hf_cache \
    --device cpu

uv run python scripts/run_qa_eval_experiment.py \
    --agent_type baseline --model_name nemotron-3-nano-musique-v3-aug:latest \
    --provider_name ollama --dataset musique-natural --num_examples 100 \
    > logs/nemotron_run.log 2>&1 &                          

uv run python scripts/download_traces.py \
    --model-name nemotron-3-nano-musique-v3-aug:latest \
    --agent-name baseline \
    --output-dir results/musique-natural \
    --eval-json results/musique-natural/musique-natural_baseline_nemotron-3-nano-musique-v3-aug:latest_run_1.json

uv run python scripts/stage_musique_natural_interplay.py \
    --model nemotron-3-nano-musique-v3-aug:latest \
    --uncertainty-json results/musique-natural/interplay_analysis_stage/musique_parametric_uncertainty_nemotron-3-nano_30b.json \
    --traces-json      results/musique-natural/musique_val_search_nemotron-3-nano-musique-v3-aug:latest_baseline_agent_run_1_traces_20260524_204617.json \
    --eval-json        results/musique-natural/musique-natural_baseline_nemotron-3-nano-musique-v3-aug:latest_run_1.json \
    --natural-jsonl    data/musique_val_natural.jsonl \
    --output-dir       results/musique-natural/interplay_analysis_tuned_val100_stage

uv run python scripts/analyze_parametric_search_interplay.py \
    --eval-dir   results/musique-natural/interplay_analysis_tuned_val100_stage \
    --output-dir results/musique-natural/interplay_analysis_tuned_val100 \
    --use-llm --llm-gold-check --attribution-model gpt-oss:120b \
    --attribution-provider ollama

uv run python scripts/run_musique_parametric_uncertainty.py \
    --model_name nemotron-3-nano:30b \
    --provider ollama \
    --num_runs 5 \
    --staleness_csv data/musique_train_staleness.csv \
    --output_dir results/musique_parametric \
    --skip-aggregate \
    --num_examples 1500 \
    --resume > musique_parametric_uncertainty_nemotron-3-nano_30b.log 2>&1

ssh -J dvirla@athena.technion.ac.il \
    -N \
    -L 127.0.0.1:8080:localhost:5000 \
    dvirla@n318

mlflow ui --host 0.0.0.0 --port 5000

# 1) Drop the tuned parametric uncertainty JSON into the canonical location
#    (slug-normalize the filename: ':' → '_')
cp results/musique-natural/interplay_analysis_tuned_stage/musique_parametric_uncertainty_nemotron-3-nano-musique-v3-aug:latest.json \
    results/musique_parametric/musique_parametric_uncertainty_nemotron-3-nano-musique-v3-aug_latest.json

# 2) Build the consolidated paper figures (replaces unified_interplay_analysis.py).
#    Reads the tuned natural slice (interplay_analysis_tuned) by default.
uv run python scripts/make_paper_figures.py --output-dir results/paper_figures