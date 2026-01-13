# Weights & Biases (wandb) Setup Guide
=====================================

This guide explains how to authenticate with wandb on a remote machine for experiment tracking.

## Quick Start

1. **Install wandb** (if not already installed):
   ```bash
   pip install wandb
   ```

2. **Login to wandb**:
   ```bash
   wandb login
   ```
   
   This will prompt you for your API key. You can find your API key at:
   https://wandb.ai/authorize

3. **Copy and paste your API key** when prompted.

## Alternative Methods

### Method 1: Interactive Login (Recommended)
```bash
wandb login
```
- Opens a browser or prompts for API key
- Automatically saves credentials to `~/.netrc` or `~/.config/wandb/settings`

### Method 2: Direct API Key
```bash
wandb login <your-api-key>
```

### Method 3: Environment Variable
```bash
export WANDB_API_KEY=your-api-key-here
```
This works without running `wandb login`, but the key won't persist across sessions.

### Method 4: Settings File
Create/edit `~/.config/wandb/settings`:
```ini
[default]
api_key = your-api-key-here
```

## Verify Login

Check if you're logged in:
```bash
wandb status
```

This will show:
- Your username
- Your API key (masked)
- Default project settings

## Using with Training Script

Once logged in, you can use wandb with the training script:

```bash
# Basic usage
python train_longformer_classifier.py --data data.csv --wandb

# With custom project name
python train_longformer_classifier.py --data data.csv --wandb --wandb-project my-project

# With custom run name
python train_longformer_classifier.py --data data.csv --wandb --wandb-run-name experiment-1
```

## Troubleshooting

### "wandb: ERROR Not logged in"
- Run `wandb login` again
- Check that your API key is correct
- Verify network connectivity

### "Permission denied" errors
- Check file permissions on `~/.netrc` or `~/.config/wandb/`
- Ensure you have write access to these directories

### Offline Mode
If you can't connect to wandb servers, you can run in offline mode:
```bash
wandb offline
```
Sync later with:
```bash
wandb sync <run-directory>
```

## Getting Your API Key

1. Go to https://wandb.ai/authorize
2. Log in to your wandb account
3. Copy the API key shown on the page
4. Use it with `wandb login <api-key>` or paste when prompted

## Additional Resources

- Wandb documentation: https://docs.wandb.ai/
- Python API reference: https://docs.wandb.ai/ref/python
