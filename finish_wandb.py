# Save this as finish_wandb.py and run it
import wandb

# Initialize a run to log final metrics (or connect to existing)
run = wandb.init(
    project="epistemic-stance",
    job_type="finalize",
    name="training-complete"
)

# Log final metrics from your last eval
run.log({
    "final/eval_loss": 0.025,  # From your logs
    "final/epoch": 3.0,
    "final/total_steps": 583,
})

run.finish()
print("W&B run finalized")