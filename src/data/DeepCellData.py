import torch
from torch.utils.data import IterableDataset
import numpy as np
import yaml
import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from skimage import io
from src.utils.deepcell_kit.config import DCTConfig

class PatchDataset():
    """
    Dataset for single-image patchified data.
    """
    def __init__(self,
                 root_dir='data/raw',
                 raw_subset_dir='',
                 split='train',
                 crop_factor=0.5,
                 n_clusters=1,
                 save_embed_data=False,
                 **args):
        
        assert split in ['train', 'test'], f'split must be either train or test, but is {split}'
 
        self.work_dir = os.path.join(os.getcwd(), root_dir, 'raw', raw_subset_dir)
        self.img_dir = os.path.join(self.work_dir,split)
        self.cells_path = [os.path.join(self.img_dir, p) for p in os.listdir(self.img_dir) if p.lower().endswith('_cells.npy')]
        self.dct_config = DCTConfig()

    def create_attn_mask(self, sample, max_channels):
        # True = padding
        # https://pytorch.org/docs/stable/generated/torch.ao.nn.quantizable.MultiheadAttention.html#torch.ao.nn.quantizable.MultiheadAttention.forward
        mask = np.full((sample.shape[0], max_channels), True)
        mask[:, 0 : sample.shape[1]] = False       
        return mask

    def pad_images(self, sample, max_channels):
        paddings = -1.0 # retrieved as a constant from repo (?)
        return np.pad(
            sample,
            ((0, 0), (0, max_channels - sample.shape[1]), (0, 0), (0, 0), (0, 0)),
            mode="constant",
            constant_values=paddings,
        )

    def save_embed_data(self, model, device='cpu', batch_size=256):
        """
        Save model representations of all cells per ROI.

        model (torch.Module): deepcell model
        device (str): device to operate on
        batch_size (int): Number of cells to extract representations from at once
        """
        ch_idx_path = [os.path.join(self.work_dir, p) for p in os.listdir(self.work_dir) if p.split(os.sep)[-1] == 'channel_idx.npy'][0]
        ch_idx = torch.from_numpy(np.load(ch_idx_path))
        model.eval()
        with torch.no_grad():
            for path in tqdm(self.cells_path, desc='Save embeddings'):
                
                sample = torch.from_numpy(np.load(path)) # (B,C,3,H,W)
 
                attn_mask = self.create_attn_mask(sample, self.dct_config.MAX_NUM_CHANNELS)  # (C_max,)
                sample = self.pad_images(sample, self.dct_config.MAX_NUM_CHANNELS)  # (C_max, 3, H, W)
                sample, attn_mask = torch.as_tensor(sample, dtype=torch.float32), torch.as_tensor(attn_mask, dtype=torch.bool)

                embed = torch.empty((sample.shape[0], model.embed_size), dtype=torch.float32)
                num_batches = (sample.shape[0]+batch_size-1) // batch_size

                for batch_idx in range(num_batches):
                    if batch_idx < num_batches - 1:
                        idx_start = batch_idx*batch_size
                        idx_end = batch_idx*batch_size+batch_size
                    else:
                        idx_start = batch_idx*batch_size
                        idx_end = len(embed)
                    embed[idx_start:idx_end] = model(
                            sample[idx_start:idx_end].to(device),
                            ch_idx.to(device),
                            attn_mask[idx_start:idx_end].to(device)
                    ).to('cpu')
                torch.save(embed, os.path.join(path, path.split('.')[0]+'_embed.pt'))
                del sample
                del attn_mask
                del embed
