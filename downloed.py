from huggingface_hub import hf_hub_download
import os

os.makedirs('models_export', exist_ok=True)

print('Telechargement pretrain B5 (1.6GB)...')
hf_hub_download(
    repo_id='shawn24/Mammo-CLIP',
    filename='Pre-trained-checkpoints/b5-model-best-epoch-7.tar',
    local_dir='models_export'
)
print('Termine !')