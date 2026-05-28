import os
import numpy as np
import pandas as pd
import dask
import dask.array as da
from dask import delayed
from dask.diagnostics import ProgressBar
import h5py
import re
import torch
from tifffile import TiffFile
import torch.nn.functional as F

def process(raw_patch, mask_patch, marker_df):

    patch_markers = []
    marker_ids = []
    for _, r in marker_df.iterrows():
        #marker_name = r['marker_name']
        channel_index = r['channel_id']
        marker_id = r['marker_id']
        marker_mean = r['marker_mean']
        marker_std = r['marker_std']
        marker_patch = raw_patch[channel_index,:,:]

        marker_max_values = np.iinfo(raw_patch.dtype).max
        marker = marker_patch / marker_max_values
        marker = (marker - marker_mean) / marker_std

        patch_markers.append(torch.tensor(marker))
        marker_ids.append(np.uint16(marker_id))

    patch_markers = torch.stack(patch_markers, dim=0)
    marker_ids = torch.tensor(marker_ids)
    cell_mask = np.uint8(mask_patch)
    patch_markers = patch_markers * cell_mask

    return patch_markers, marker_ids

def patch_generator(raw, mask, cell_cutout, df):
    
    xs = df["Centroid.X.px"].round().astype(int).to_numpy()
    ys = df["Centroid.Y.px"].round().astype(int).to_numpy()
    if "cell_ID" in df.columns:
        cell_ids = df["cell_ID"].astype(int).to_numpy()
    else:
        cell_ids = np.empty(len(xs),dtype=np.uint16)
        for i in range(len(xs)):
            cell_id = int(mask[ys[i],xs[i]])
            if cell_id == 0:
                raise ValueError('Warning! A centroid points outside of cell')
            cell_ids[i] = cell_id

    for i in range(len(xs)):
        x, y = ys[i], xs[i]     # Switching x and y order because [row,column] = [y,x]
        cell_id = cell_ids[i]   # Original KRONOS interpret images with [H,W]

        x1 = (x - (cell_cutout // 2)) if (x - (cell_cutout // 2)) >= 0 else 0
        x2 = (x + (cell_cutout // 2)) if (x + (cell_cutout // 2)) < raw.shape[1] else raw.shape[1]
        y1 = (y - (cell_cutout // 2)) if (y - (cell_cutout // 2)) >= 0 else 0
        y2 = (y + (cell_cutout // 2)) if (y + (cell_cutout // 2)) < raw.shape[2] else raw.shape[2]

        raw_patch = raw[:, x1:x2, y1:y2]
        mask_patch = mask[x1:x2, y1:y2]

        if raw_patch.shape[1] != cell_cutout or raw_patch.shape[2] != cell_cutout:
            pre_pad_x = 0
            post_pad_x = 0
            pre_pad_y = 0
            post_pad_y = 0

            if (x - (cell_cutout // 2)) < 0:
                pre_pad_x = abs(x - (cell_cutout // 2))
            if (x + (cell_cutout // 2)) >= raw.shape[1]:
                post_pad_x = abs((x + (cell_cutout // 2)) - raw.shape[1])

            if (y - (cell_cutout // 2)) < 0:
                pre_pad_y = abs(y - (cell_cutout // 2))
            if (y + (cell_cutout // 2)) >= raw.shape[2]:
                post_pad_y = abs((y + (cell_cutout // 2)) - raw.shape[2])
            
            raw_patch = np.pad(raw_patch, ((0, 0), (pre_pad_x, post_pad_x), (pre_pad_y, post_pad_y)), mode='constant', constant_values=0)
            mask_patch = np.pad(mask_patch, ((pre_pad_x, post_pad_x), (pre_pad_y, post_pad_y)), mode='constant', constant_values=0)

            assert raw_patch.shape[1] == cell_cutout and raw_patch.shape[2] == cell_cutout, "Patch size mismatch after padding"
            assert mask_patch.shape[0] == cell_cutout and mask_patch.shape[1] == cell_cutout, "Mask size mismatch after padding"
        
        mask_patch = (mask_patch == cell_id).astype(np.uint8)

        yield raw_patch, mask_patch, i

@delayed
def cell_seg(img_path,
             mask_path,
             df, 
             cell_cutout,
             token_size,
             marker_df,
             channel_names,
             channel_mask):

    raw = TiffFile(img_path).asarray()
    mask = TiffFile(mask_path).asarray()

    cell_results = torch.empty((
        df.shape[0],
        marker_df.shape[0], 
        token_size*16,
        token_size*16
        ), dtype=torch.float32)

    for raw_patch, mask_patch, run in patch_generator(raw,
                                                      mask,
                                                      cell_cutout,
                                                      df):
        
        patch_markers,_ = process(raw_patch, mask_patch, marker_df)
        patch_markers = patch_markers.unsqueeze(0) # (C,H,W) -> (1,C,H,W)
        if cell_cutout != token_size*16:
            patch_markers = F.interpolate(
                    patch_markers,
                    size=(token_size*16,token_size*16),
                    mode="bilinear",
                    align_corners=False
                    )
        patch_markers = patch_markers.squeeze(0)
        cell_results[run] = patch_markers
    
    np.save(os.path.join(img_path.split('.ome.tif')[0]+'_cells.npy'), cell_results.numpy())
    del cell_results

def extract_idx(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem.rsplit('_', 1)[1])

def image_preprocess(path,
                     channel_names,
                     cell_cutout=64,
                     token_size=14,
                     batch_size=1,
                     ids_path=""):
 
    from src.utils.kronos_kit.marker_metadata import check_metadata
    channel_mask = check_metadata(path,channel_names)
    channel_names = [p for p,m in zip(channel_names,channel_mask) if m]

    mask_dir = os.path.join(os.getcwd(),path,'masks')
    train_dir = os.path.join(os.getcwd(),path,'train')
    test_dir = os.path.join(os.getcwd(),path,'test')

    marker_df = pd.read_csv(os.path.join(os.getcwd(),path,'tsv'))
 
    # Ideally centroids should be INSIDE the cell masks where it extracts cell_IDs from, but
    # in some cases this doesn't happen. If the raw data can provide cell_IDs alongside coordinates,
    # the information is instead extracted from a separate CSV instead of the {MEASUREMENTS}.csv
    if ids_path != "":
        df = pd.read_csv(os.path.join(os.getcwd(),"data","raw",ids_path))
    else:
        df = pd.read_csv([os.path.abspath(os.path.join(path, p)) for p in os.listdir(path) if p.lower().endswith(('csv'))][0])
    
    img_names = list(set(
        [re.sub(r'_\d+$', '', p.split(os.sep)[-1].split('.ome.tif')[0]) for p in df['Image'].tolist()]
        ))
    
    print(">>> Detected image names:")
    [print(p) for p in img_names]

    total_rois = len(os.listdir(mask_dir))
    current_total = 0
    for img_name in img_names:   
        img_paths = sorted(([os.path.join(train_dir,p)
                             for p in os.listdir(train_dir) 
                             if p.startswith(img_name) and not p.endswith(('.npy','.pt'))]
                             + ([os.path.join(test_dir,p)
                             for p in os.listdir(test_dir) 
                                 if p.startswith(img_name) and not p.endswith(('.npy','.pt'))])),
                            key=extract_idx)
        mask_paths = sorted([os.path.join(mask_dir,p)
                             for p in os.listdir(mask_dir)
                            if p.startswith(img_name)],
                            key=extract_idx)

        if len(img_paths) == 0:
            print(f">>> Warning! No matching images for {img_name}")
            continue
        if len(mask_paths) == 0:
            print(f">>> Warning! No matching masks for {img_name}")
            continue

        num_batches = (len(img_paths)+batch_size-1) // batch_size
        print(f'>> Processing {img_name}, {len(img_paths)} ROIs in batches of {batch_size}')
        for batch_idx in range(num_batches):
            if batch_idx < num_batches - 1:
                start = batch_idx*batch_size
                end = batch_idx*batch_size+batch_size
            else:
                start = batch_idx*batch_size
                end = len(img_paths)
            tasks = [cell_seg(img_paths[p],
                              mask_paths[p],
                              df[df["Image"]==img_paths[p].split(os.sep)[-1]],
                              cell_cutout,
                              token_size,
                              marker_df,
                              channel_names,
                              channel_mask)
                              for p in range(start,end)]
            with ProgressBar():
                dask.compute(*tasks)
        current_total += len(img_paths)
        print(f'{current_total}/{total_rois} ROIs processed')
