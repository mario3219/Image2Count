import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import torch

class KRONOSDataset():
    """
    Dataset for single-image patchified data.
    """
    def __init__(self,
                 root_dir='data/raw',
                 raw_subset_dir='',
                 split='train',
                 **args):
        
        assert split in ['train', 'test'], f'split must be either train or test, but is {split}'
 
        self.work_dir = os.path.join(os.getcwd(), root_dir, 'raw', raw_subset_dir)
        self.img_dir = os.path.join(self.work_dir,split)
        self.cells_path = [os.path.join(self.img_dir, p) for p in os.listdir(self.img_dir) if p.lower().endswith('_cells.npy')]
        
        try:
            marker_df = pd.read_csv(
                    os.path.join(os.getcwd(),root_dir,'raw',raw_subset_dir,'tsv')
                    )
        except Exception:
            print(f'Warning! tsv not found in {raw_subset_dir}, run image_preprocess for KRONOS first')
        self.marker_ids = marker_df['marker_id']
    
    def save_embed_data(self, model, device='cpu', batch_size=256):
        """
        Save model representations of all cells per ROI.

        model (torch.Module): KRONOS model
        device (str): device to operate on
        batch_size (int): Number of cells to extract representations from at once
        """
        
        model.eval()
        with torch.no_grad():
            for path in tqdm(self.cells_path, desc='Save embeddings'):
                sample = torch.from_numpy(np.load(path))
                marker_ids = torch.from_numpy(np.repeat(
                        np.expand_dims(self.marker_ids, axis=0),
                        repeats=sample.shape[0],axis=0))
                embed = torch.empty((sample.shape[0],model.embed_dim), dtype=torch.float32)
                num_batches = (sample.shape[0]+batch_size-1) // batch_size
                for batch_idx in range(num_batches):
                    if batch_idx < num_batches - 1:
                        idx_start = batch_idx*batch_size
                        idx_end = batch_idx*batch_size+batch_size
                    else:
                        idx_start = batch_idx*batch_size
                        idx_end = len(embed)

                    patch_features, patch_marker_features, patch_token_features = model(
                                sample[idx_start:idx_end].to(device,dtype=torch.float32),
                                marker_ids=marker_ids[idx_start:idx_end].to(device,dtype=torch.int64)
                    )
                    embed[idx_start:idx_end] = patch_features.to('cpu')
                    del patch_features
                    del patch_marker_features
                    del patch_token_features

                torch.save(embed, os.path.join(path, path.split('.')[0]+'_embed.pt'))
                del sample
                del marker_ids
                del embed
