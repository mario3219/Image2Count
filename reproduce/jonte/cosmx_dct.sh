#!/bin/bash

experiment=cosmx_dct

python -m cellcontrast  --image_preprocess \
                        --preprocess_dir "data/raw/${experiment}/" \
                        --channel_names SKIP PanCK CD45 CD3 DAPI \
                        --cell_cutout 100 \
                        --foundation_model 'deepcell' \
                        --mpp 0.399 \
                        --batch_size 10 \
                        --cell_ids cosmx_ids_flipped_formatted.csv

python -m cellcontrast  --embed_image_data \
                        --raw_subset_dir "${experiment}" \
                        --foundation_model 'deepcell' \
                        --output_name 'out/models/deepcell_types_modified.pt' \
                        --channel_names SKIP PanCK CD45 CD3 DAPI
