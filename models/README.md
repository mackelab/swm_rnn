# Training checkpoints (scratch)

Default output directory for [`train_scripts/`](../train_scripts/). Checkpoints written here during training are gitignored.

Copy finished runs to [`../final_models/`](../final_models/) (`macaque/` or `synthetic/`) for the figure pipeline.

Each saved model is a name prefix plus:

- `_state_dict_rnn.pkl`
- `_state_dict_enc.pkl` (if encoder enabled)
- `_vae_params.pkl`, `_task_params.pkl`, `_training_params.pkl`

Written by `vi_rnn.saving.save_model`.
