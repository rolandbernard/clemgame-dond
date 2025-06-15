#!/bin/bash

# Activate the virtual environment to have access to the clem cli.
source venv/bin/activate
export PYTHONPATH=.:$PYTHONPATH

mkdir -p logs

# Only run one game.
games=("dond")

# Run with all of the models we defined in `model_registry.json`.
models=(
    "gemma-3-27b-it"
    "mistral-small-2503"
    "llama-3-70b"
    "gpt-3.5-turbo-0125"
    "gemini-2.0-flash-001"
    "llama-4-maverick-17b-128e"
    "mistral-large-2411"
    "o4-mini-2025-04-16"
    "gpt-4.1-2025-04-16"
    "gemini-2.5-flash-preview-05-20"
    "gpt-4-turbo-2024-04-09"
)
# Removed the below four because I did not get them to work. They output raw
# thinking data which I didn't really know how to handle because it was being cut
# off.
# "qwen3-32b"
# "qwen-qwq-32b"
# "magistral-small-2506"
# "magistral-medium-2506"

languages=("en" "de" "it")

# Not that we skip the competitive scenario, because we can not evaluate it well
# if both models are the same. We could evaluate models by fixing the opponent.
modes=("coop" "semi")

echo "==================================================="
echo "RUNNING: Benchmark Run"
echo "==================================================="

for game in "${games[@]}"; do
    for mode in "${modes[@]}"; do
        for lang in "${languages[@]}"; do
            for model in "${models[@]}"; do
                if [ ! -e "results/${model}-t0.0--${model}-t0.0/${game}/0_${mode}_${lang}" ]; then
                    echo "Testing ${model} on ${game} (${mode}, ${lang})"
                    { time clem run -g "${game}" -m "${model}" -i "instances_${mode}_${lang}"; } 2>&1 \
                        | tee logs/runtime."${game}"."${mode}"."${lang}"."${model}".log
                fi
            done
        done
    done
    echo "Evaluating results for ${game}."
    { time clem transcribe -g "${game}"; } 2>&1 | tee logs/runtime.transcribe."${game}".log
    { time clem score -g "${game}"; } 2>&1 | tee logs/runtime.score."${game}".log
done
echo "Evaluating all results."
{ time clem eval; } 2>&1 | tee logs/runtime.score."${game}".log

echo "==================================================="
echo "FINISHED: Benchmark Run"
echo "==================================================="
