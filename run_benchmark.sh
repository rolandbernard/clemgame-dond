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
    "gemini-2.5-flash-preview-05-20"
    "gemini-2.0-flash-001"
    "o4-mini-2025-04-16"
    "gpt-3.5-turbo-0125"
    "gpt-4.1-2025-04-16"
    "mistral-large-2411"
    "mistral-small-2503"
    "magistral-medium-2506"
    "magistral-small-2506"
    "qwen-qwq-32b"
    "qwen3-32b"
    "llama-3-70b"
    "llama-4-maverick-17b-128e"
)

languages=("en" "de" "it")

# Not that we skip the competitive scenario, because we can not evaluate it well
# if both models are the same. We could evaluate models by fixing the opponent.
modes=("coop" "semi")

echo
echo "==================================================="
echo "RUNNING: Benchmark Run"
echo "==================================================="
echo

for mode in "${modes[@]}"; do
    for lang in "${languages[@]}"; do
        for game in "${games[@]}"; do
            for model in "${models[@]}"; do
                echo "Testing ${model} on ${game} (${mode}, ${lang})"
                { time clem run -g "${game}" -m "${model}" -i "instances_${mode}_${lang}" -r "results_${mode}_${lang}"; } 2>&1 \
                    | tee logs/runtime."${game}"."${model}".log
                { time clem transcribe -g "${game}" -r "results_${mode}_${lang}"; } 2>&1 \
                    | tee logs/runtime.transcribe."${game}".log
                { time clem score -g "${game}" -r "results_${mode}_${lang}"; } 2>&1 \
                    | tee logs/runtime.score."${game}".log
            done
        done
        echo "Evaluating results for ${mode} in ${lang}"
        { time clem eval -r "results_${mode}_${lang}"; }
    done
done

echo "==================================================="
echo "FINISHED: Benchmark Run"
echo "==================================================="
