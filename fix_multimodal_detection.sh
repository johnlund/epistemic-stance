#!/bin/bash
# Workaround for Axolotl multimodal detection issue with Magistral-Small-2509
# This model is text-only but uses Mistral3 architecture which triggers multimodal detection

set -e

echo "Applying workaround for Magistral multimodal detection issue..."

# Clear any cached prepared data that might have old settings
rm -rf ./prepared_data

# Create a minimal preprocessor_config.json to satisfy the loader
# This is a dummy file that prevents the OSError
mkdir -p ~/.cache/huggingface/hub/models--mistralai--Magistral-Small-2509/snapshots/

# Find the actual snapshot directory
SNAPSHOT_DIR=$(find ~/.cache/huggingface/hub/models--mistralai--Magistral-Small-2509/snapshots/ -maxdepth 1 -type d | tail -1)

if [ -n "$SNAPSHOT_DIR" ] && [ -d "$SNAPSHOT_DIR" ]; then
    echo "Creating dummy preprocessor_config.json in $SNAPSHOT_DIR"
    cat > "$SNAPSHOT_DIR/preprocessor_config.json" << 'EOF'
{
  "do_convert_rgb": true,
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [0.48145466, 0.4578275, 0.40821073],
  "image_processor_type": "PixtralImageProcessor",
  "image_std": [0.26862954, 0.26130258, 0.27577711],
  "max_image_size": 1024,
  "patch_size": 16,
  "rescale_factor": 0.00392156862745098,
  "resample": 2,
  "size": {"longest_edge": 1024}
}
EOF
    echo "✓ Created dummy preprocessor_config.json"
else
    echo "⚠ Could not find model cache directory. Will create after model download."
fi

echo ""
echo "Workaround applied. Now run the training script."
