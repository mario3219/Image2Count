import torch
import os
from typing import Optional, Tuple
from huggingface_hub import hf_hub_download
import shutil

def create_model_from_pretrained(
    checkpoint_path: Optional[str] = None,
    cfg_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    hf_auth_token: Optional[str] = None,
    cfg: Optional[dict] = None,
) -> Tuple[torch.nn.Module, torch.dtype, int]:

    checkpoint_filename = "kronos_vits16_model.pt"
        
    # Download checkpoint from Hugging Face Hub
    checkpoint_path = hf_hub_download(
        checkpoint_path[len("hf_hub:"):], 
        cache_dir=cache_dir,
        filename=checkpoint_filename,
        token=hf_auth_token
    )

    # Load the state dictionary, removing specific prefixes and entries
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    state_dict = state_dict['teacher']
    state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
    state_dict = {k: v for k, v in state_dict.items() if 'dino_head' not in k}

    print(f"\033[92mLoaded model weights from {checkpoint_path}\033[0m")

    return state_dict

def kr_download(model_path):
 
    state_dict = create_model_from_pretrained(
        checkpoint_path="hf_hub:MahmoodLab/kronos", # Make sure you have requested access on HuggingFace
        cache_dir="./model_assets",
    )
    torch.save(state_dict, os.path.join(model_path,"kronos_weights.pt"))
    if os.path.exists(os.path.join(model_path,"model_assets")):
        shutil.rmtree(os.path.join(model_path,"model_assets"))
    if os.path.exists(os.path.join(model_path,".locks")):
        shutil.rmtree(os.path.join(model_path,".locks"))
