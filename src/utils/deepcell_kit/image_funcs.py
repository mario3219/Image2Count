import numpy as np
from skimage.transform import rescale
import os
import pandas as pd
import torch
import torch.nn.functional as F
import warnings
import dask
import dask.array as da
from dask import delayed
import re
from dask.diagnostics import ProgressBar
from src.utils.image_preprocess import load_img
from src.utils.deepcell_kit.config import DCTConfig

def pad_cell(X: np.ndarray, y: np.ndarray, crop_size: int):
    delta = crop_size // 2
    X = np.pad(X, ((delta, delta), (delta, delta), (0, 0)))
    y = np.pad(y, ((delta, delta), (delta, delta)))
    return X, y

def get_crop_box(centroid, delta):
    minr = int(centroid[0]) - delta
    maxr = int(centroid[0]) + delta
    minc = int(centroid[1]) - delta
    maxc = int(centroid[1]) + delta
    return np.array([minr, minc, maxr, maxc])

def get_neighbor_masks(mask_patch, cbox, cell_id):
    """Returns binary masks of a cell and its neighbors. This function expects padding around
    the edges, and will throw an error if you hit a wrap around."""
    minr, minc, maxr, maxc = cbox
    assert np.issubdtype(mask_patch.dtype, np.integer) and isinstance(cell_id, int)

    binmask_cell = (mask_patch == cell_id).astype(np.int32)

    binmask_neighbors = (mask_patch != cell_id).astype(np.int32) * (
        mask_patch != 0
    ).astype(np.int32)
    return binmask_cell, binmask_neighbors

def combine_masks(raw, mask):
    mask = np.swapaxes(mask, 0, 2)  # (2, H, W)
    mask = np.expand_dims(mask, axis=0)  # (1, 2, H, W)
    raw_aug_mask = np.concatenate(
        [
            np.expand_dims(raw, axis=1),  # (C, 1, H, W)
            np.tile(mask, (raw.shape[0], 1, 1, 1)),  # (C, 2, H, W)
        ],
        axis=1,
    )  # (C, 3, H, W)
    return raw_aug_mask

def get_channel_masking(channel_names, channel_mapping):
    if len(channel_names) == 0:
        print('Warning! channel_names is empty, all channels will be masked out!')
    channel_names_standard = []
    channel_masking = []
    for ch_name in channel_names:
        if ch_name not in channel_mapping:
            channel_masking.append(True)
            warnings.warn(
                f"Channel {ch_name} is not in the channel mapping. "
                "This channel will be masked out."
            )
        else:
            channel_masking.append(False)
            channel_names_standard.append(channel_mapping[ch_name])
    return channel_masking, channel_names_standard

def get_ch_idx(channel_names_standard, marker2idx, max_channels):
    return torch.as_tensor(
        [marker2idx[ch_name] for ch_name in channel_names_standard]
        + [-1] * (max_channels - len(channel_names_standard))
    )  # (C_max, )

def normalize_per_channel(image, min_vals, ptp_vals):
    return (image-min_vals)/ptp_vals

def percentile_threshold(image, img_max, percentile=99.9):
    """Copied and modified from: https://github.com/vanvalenlab/deepcell-toolbox/blob/e8c1277ee4243bc6a34916d554d0c2eab0cf7505/deepcell_toolbox/processing.py#L104
    Threshold an image to reduce bright spots

    Args:
        image: numpy array of image data
        percentile: cutoff used to threshold image

    Returns:
        np.array: thresholded version of input image
    """

    processed_image = np.zeros_like(image)
    for chan in range(image.shape[-1]):
        current_img = np.copy(image[..., chan])
        non_zero_vals = current_img[np.nonzero(current_img)]
        # only threshold if channel isn't blank
        if len(non_zero_vals) > 0:
            # threshold values down to max
            threshold_mask = current_img > img_max[chan]
            current_img[threshold_mask] = img_max[chan]

            # update image
            processed_image[..., chan] = current_img

    return processed_image

def get_cell_ids(mask, df):

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

    return cell_ids

def process(raw, mask, cell_cutout, mpp, properties, dct_config):
    
    raw = np.transpose(raw, (1, 2, 0))  # (H, W, C)
    raw = rescale(raw, mpp / dct_config.STANDARD_MPP_RESOLUTION, preserve_range=True, channel_axis=-1)

    mask = rescale(
        mask,
        mpp / dct_config.STANDARD_MPP_RESOLUTION,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.int32)

    # Because if images get rescaled, the cut-out has to be rescaled too
    # Rounded to closest %2 == 0
    cell_cutout = (2 * np.round(
        (cell_cutout * (mpp / dct_config.STANDARD_MPP_RESOLUTION)) / 2
    )).astype(int)

    min_vals, ptp_vals, img_max, _, _ = properties
    raw = percentile_threshold(raw, img_max, percentile=dct_config.PERCENTILE_THRESHOLD)
    raw = normalize_per_channel(raw, min_vals, ptp_vals)
    raw, mask = pad_cell(raw, mask, cell_cutout)

    return raw, mask, cell_cutout

@delayed
def per_image_hist(img_path, nbins, vmin, vmax, channel_masking, mpp):
    img = load_img(img_path, "").astype(np.float32)
    img = img[~np.array(channel_masking), :, :]
    img = np.transpose(img, (1, 2, 0))

    dct_config = DCTConfig()
    img = rescale(img, mpp / dct_config.STANDARD_MPP_RESOLUTION,
                  preserve_range=True, channel_axis=-1)

    C = img.shape[-1]
    H = np.zeros((C, nbins), dtype=np.int64)

    for c in range(C):
        vals = img[..., c]
        vals = vals[vals != 0]
        if vals.size:
            h, _ = np.histogram(vals, bins=nbins, range=(vmin[c], vmax[c]))
            H[c] = h

    return H  # (C, nbins)

def build_global_hist(img_paths, nbins, vmin, vmax, channel_masking, mpp):
    C = len(vmin)
    tasks = [per_image_hist(p, nbins, vmin, vmax, channel_masking, mpp) for p in img_paths]

    H = da.stack(
        [da.from_delayed(t, shape=(C, nbins), dtype=np.int64) for t in tasks],
        axis=0
    )  # (N, C, nbins)

    return H.sum(axis=0)  # (C, nbins)

def hist_to_percentile(global_hist, percentile, vmin, vmax):
    """
    https://stackoverflow.com/questions/10640759/how-to-get-the-cumulative-distribution-function-with-numpy
    """
    cdf = np.cumsum(global_hist, axis=1)
    totals = cdf[:, -1]

    out = np.full((global_hist.shape[0],), np.nan, dtype=np.float32)
    ok = totals > 0

    cdf_ok = (cdf[ok].T / totals[ok]).T
    idx = np.argmax(cdf_ok >= (percentile / 100.0), axis=1)

    bin_width = (vmax[ok] - vmin[ok]) / global_hist.shape[1]
    out[ok] = (vmin[ok] + idx * bin_width).astype(np.float32)
    return out

@delayed
def get_properties(img_path, channel_masking, mpp):
    img_data = load_img(img_path,'').astype(np.float32)[~np.array(channel_masking), :, :]
    img_data = np.transpose(img_data, (1, 2, 0))
    dct_config = DCTConfig()
    img_data = rescale(img_data, mpp / dct_config.STANDARD_MPP_RESOLUTION,
                  preserve_range=True, channel_axis=-1)
    min_vals = np.min(img_data, axis=(0, 1), keepdims=True)
    max_vals = np.max(img_data, axis=(0, 1), keepdims=True)
    return min_vals, max_vals

@delayed
def get_true_max(img_path, channel_masking, img_max, percentile, mpp):
    img_data = load_img(img_path,'').astype(np.float32)[~np.array(channel_masking), :, :]
    img_data = np.transpose(img_data, (1, 2, 0))
    dct_config = DCTConfig()
    img_data = rescale(img_data, mpp / dct_config.STANDARD_MPP_RESOLUTION,
                  preserve_range=True, channel_axis=-1)
    img_data = percentile_threshold(img_data, img_max, percentile)
    max_vals = np.max(img_data, axis=(0, 1), keepdims=True)
    return max_vals

def get_global_properties(img_paths,
                          channel_masking,
                          percentile,
                          standard_mpp,
                          cell_cutout,
                          mpp):

    tasks = [get_properties(p, channel_masking, mpp) for p in img_paths]
    results = dask.compute(*tasks)
    mins = [mn for (mn, mx) in results]
    maxs = [mx for (mn, mx) in results]

    min_vals = np.min(np.stack(mins, axis=0), axis=0)
    max_vals = np.max(np.stack(maxs, axis=0), axis=0)

    # This can be increased for higher accuracy, but
    # might run slower. 4096 is typically fine
    NBINS = 4096
    global_hist_dask = build_global_hist(img_paths, NBINS, np.squeeze(min_vals), np.squeeze(max_vals), channel_masking, mpp)
    global_hist = global_hist_dask.compute()
    img_max = hist_to_percentile(global_hist, percentile=percentile, vmin=np.squeeze(min_vals), vmax=np.squeeze(max_vals))
    
    tasks = [get_true_max(p, channel_masking, img_max, percentile, mpp) for p in img_paths]
    results = dask.compute(*tasks)
    maxs = [mx for mx in results]
    max_vals = np.max(np.stack(maxs, axis=0), axis=0)
    ptp_vals = max_vals - min_vals

    pad = cell_cutout // 2
    scale = mpp/standard_mpp

    return min_vals, ptp_vals, img_max, pad, scale

def patch_generator(raw,
                    mask,
                    df,
                    cell_ids,
                    cell_cutout,
                    properties,
                    dct_config):

    _, _, _, pad, scale = properties
    xs = np.round(df['Centroid.X.px'].values*scale+pad).astype(int)
    ys = np.round(df['Centroid.Y.px'].values*scale+pad).astype(int)
    
    for i in range(len(xs)):
        y, x = ys[i], xs[i]
        cell_id = int(cell_ids[i])

        delta = cell_cutout // 2
        cbox = get_crop_box((y, x), delta)
        minr, minc, maxr, maxc = cbox
        raw_patch = raw[minr:maxr, minc:maxc, :]  # (H, W, C)
        mask_patch = mask[minr:maxr, minc:maxc]

        self_mask, neighbor_mask = get_neighbor_masks(
            mask_patch, cbox, cell_id
        )  # (H, W), (H, W)

        raw_patch = np.transpose(raw_patch, (2, 0, 1))  # (C, H, W)
        mask_patch = np.stack([self_mask, neighbor_mask], axis=-1)  # (H, W, 2)

        yield raw_patch, mask_patch.astype(np.float32), i

@delayed
def cell_seg(img_path,
             mask_path,
             df,
             cell_cutout,
             properties,
             channel_masking,
             mpp,
             dct_config):

    raw = load_img(img_path,'').astype(np.float32)[~np.array(channel_masking), :, :]
    mask = np.squeeze(load_img(mask_path,'')).astype(np.uint32)
    
    # get_cell_ids was separated outside the patch_generator
    # compared to other model preprocess pipelines, because
    # the rescaling and rounding can lead to coordinates
    # shifting outside cells, even if coordinates are also
    # shifted
    cell_ids = get_cell_ids(mask, df)

    raw, mask, cell_cutout = process(raw,
                            mask,
                            cell_cutout,
                            mpp,
                            properties,
                            dct_config)

    cell_results = torch.empty((
        df.shape[0],
        raw.shape[2],
        3, 
        64,
        64
        ), dtype=torch.float32)

    for sample_patch, mask_patch, i in patch_generator(raw,
                                                     mask,
                                                     df,
                                                     cell_ids,
                                                     cell_cutout,
                                                     properties,
                                                     dct_config):
        cell = torch.as_tensor(
            combine_masks(sample_patch, mask_patch),  # (C, 3, H, W)
            dtype=torch.float32
            )
        if cell_cutout != 64:
            cell = F.interpolate(
                    cell,
                    size=(64,64),
                    mode="bilinear",
                    align_corners=False
                    )
        cell_results[i] = cell
    
    np.save(os.path.join(img_path.split('.ome.tif')[0]+'_cells.npy'), cell_results.numpy())
    del cell_results
    del raw 
    del mask
    del sample_patch
    del mask_patch
    del cell_ids

def extract_idx(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem.rsplit('_', 1)[1])

def image_preprocess(path,
                     channel_names='',
                     cell_cutout=64,
                     mpp=0.4,
                     batch_size=1,
                     ids_path=""):
    
    mask_dir = os.path.join(os.getcwd(),path,'masks')
    train_dir = os.path.join(os.getcwd(),path,'train')
    test_dir = os.path.join(os.getcwd(),path,'test')

    dct_config = DCTConfig()
    channel_masking, channel_names_standard = get_channel_masking(channel_names,
                                                                  channel_mapping=dct_config.channel_mapping)
    ch_idx = get_ch_idx(channel_names_standard,
                        marker2idx=dct_config.marker2idx,
                        max_channels=dct_config.MAX_NUM_CHANNELS)
    np.save(os.path.join(path, 'channel_idx.npy'), ch_idx.numpy())
    del ch_idx

    if ids_path != "":
        df = pd.read_csv(os.path.join(os.getcwd(),"data","raw",ids_path))
    else:
        df = pd.read_csv([os.path.abspath(os.path.join(path, p)) for p in os.listdir(path) if p.lower().endswith(('csv'))][0])

    img_names = list(set(
        [re.sub(r'_\d+$', '', p.split(os.sep)[-1].split('.ome.tif')[0]) for p in df['Image'].tolist()]
        ))
 
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

        properties = get_global_properties(img_paths,
                                           channel_masking,
                                           dct_config.PERCENTILE_THRESHOLD,
                                           dct_config.STANDARD_MPP_RESOLUTION,
                                           cell_cutout,
                                           mpp)
        
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
                              (properties),
                              channel_masking,
                              mpp,
                              dct_config)
                              for p in range(start,end)]
            with ProgressBar():
                dask.compute(*tasks)
        current_total += len(img_paths)
        print(f'{current_total}/{total_rois} ROIs processed')
