# HuggingFace Hub Setup Guide
=============================

This guide explains how to authenticate with HuggingFace Hub on a remote machine to upload trained models.

## Quick Start

1. **Install huggingface_hub** (if not already installed):
   ```bash
   pip install huggingface_hub
   ```

2. **Login to HuggingFace**:
   ```bash
   huggingface-cli login
   ```
   
   This will prompt you for your access token. You can create a token at:
   https://huggingface.co/settings/tokens

3. **Copy and paste your access token** when prompted.

## Alternative Methods

### Method 1: Interactive Login (Recommended)
```bash
huggingface-cli login
```
- Prompts for access token
- Automatically saves credentials to `~/.cache/huggingface/token`

### Method 2: Direct Token
```bash
huggingface-cli login --token <your-token>
```

### Method 3: Environment Variable
```bash
export HF_TOKEN=your-token-here
```
This works without running `huggingface-cli login`, but the token won't persist across sessions.

### Method 4: Token File
Create/edit `~/.cache/huggingface/token`:
```
your-token-here
```

### Method 5: Python API
```python
from huggingface_hub import login
login(token="your-token-here")
```

## Verify Login

Check if you're logged in:
```bash
huggingface-cli whoami
```

This will show your username if logged in.

## Getting Your Access Token

1. Go to https://huggingface.co/settings/tokens
2. Log in to your HuggingFace account
3. Click "New token"
4. Choose token type:
   - **Read**: For downloading models (default)
   - **Write**: For uploading models (required for model uploads)
5. Copy the token (you won't be able to see it again!)
6. Use it with `huggingface-cli login` or paste when prompted

## Using with Training Script

Once logged in, you can upload models after training:

```bash
# Basic usage (uploads to your username/model-name)
python train_longformer_classifier.py --data data.csv --push-to-hub

# With custom model ID
python train_longformer_classifier.py --data data.csv --push-to-hub --hub-model-id my-username/epistemic-stance-classifier

# With organization
python train_longformer_classifier.py --data data.csv --push-to-hub --hub-organization my-org
```

### Model ID Format

The model ID can be:
- `username/model-name` - Personal model under your account
- `organization/model-name` - Model under an organization

If you don't specify `--hub-model-id`, the script will:
1. Use `--hub-organization` if provided: `{org}/epistemic-stance-longformer`
2. Otherwise use your username: `{username}/epistemic-stance-longformer`

## What Gets Uploaded

When using `--push-to-hub`, the script uploads:
- ✅ Model weights (`pytorch_model.bin` or `model.safetensors`)
- ✅ Model configuration (`config.json`)
- ✅ Tokenizer files (`tokenizer.json`, `tokenizer_config.json`, etc.)
- ✅ Temperature scaler (if calibration was used)
- ✅ Model card (`README.md`)

## Troubleshooting

### "Not authenticated" error
- Run `huggingface-cli login` again
- Check that your token has **Write** permissions
- Verify token is saved in `~/.cache/huggingface/token`

### "Permission denied" errors
- Ensure your token has **Write** access (not just Read)
- Check that you have permission to create repos in the specified organization
- Verify you're logged in with `huggingface-cli whoami`

### "Repository already exists" error
- The model ID you specified already exists
- Either use a different model ID or delete the existing repo first
- You can delete repos at: https://huggingface.co/{username-or-org}/{model-name}/settings

### Network/Connection errors
- Check your internet connection
- Verify you can access https://huggingface.co
- Some networks may block uploads; try from a different network

## Manual Upload

If automatic upload fails, you can upload manually:

```bash
# From the model directory
cd output/best_model
huggingface-cli upload my-username/my-model-name .
```

Or using Python:
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./output/best_model",
    repo_id="my-username/my-model-name",
    repo_type="model"
)
```

## Creating Organizations

To upload models under an organization:

1. Go to https://huggingface.co/organizations/new
2. Create a new organization
3. Add members if needed
4. Use `--hub-organization org-name` when training

## Additional Resources

- HuggingFace Hub documentation: https://huggingface.co/docs/hub
- Python API reference: https://huggingface.co/docs/huggingface_hub
- Model cards guide: https://huggingface.co/docs/hub/model-cards
