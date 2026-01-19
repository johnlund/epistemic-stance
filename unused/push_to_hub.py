# Save this as push_to_hub.py and run it
from huggingface_hub import HfApi, upload_folder
import os

# Your model path and hub repo
model_path = "./epistemic_stance_model"
hub_model_id = "johnclund/epistemic-stance-analyzer"

# Upload the model
api = HfApi()
api.create_repo(repo_id=hub_model_id, exist_ok=True)

upload_folder(
    folder_path=model_path,
    repo_id=hub_model_id,
    ignore_patterns=["checkpoint-*", "*.pth", "optimizer.pt", "scheduler.pt", "global_step*"],  # Skip checkpoints and optimizer states
)

print(f"Model pushed to: https://huggingface.co/{hub_model_id}")