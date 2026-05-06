#!/bin/bash
# 批量生成所有尺寸的 leukocyte 输入数据

set -e

cd "$(dirname "$0")"

echo "========================================="
echo "Generating Leukocyte Input Datasets"
echo "========================================="

PRESETS=("mini" "small" "medium" "large" "extra-large")

for preset in "${PRESETS[@]}"; do
    echo ""
    echo ">>> Generating $preset dataset..."
    python3 generate.py synthetic \
        --preset "$preset" \
        --out-dir "input_data/$preset" \
        --seed 42 \
        --overwrite
    
    if [ $? -eq 0 ]; then
        echo "✓ $preset dataset generated successfully"
    else
        echo "✗ Failed to generate $preset dataset"
        exit 1
    fi
done

echo ""
echo "========================================="
echo "All datasets generated successfully!"
echo "========================================="
echo ""
echo "Dataset sizes:"
du -sh input_data/*/testfile.avi 2>/dev/null || echo "Files generated in input_data/{mini,small,medium,large,extra-large}/"

