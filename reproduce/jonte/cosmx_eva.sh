#!/bin/bash

experiment=cosmx_eva

python -m cellcontrast  --image_preprocess \
                        --preprocess_dir "data/raw/${experiment}/" \
                        --channel_names SKIP Keratin8/18 CD45 CD3e DAPI \
                        --cell_cutout 100 \
                        --foundation_model 'eva' \
                        --cell_ids cosmx_ids_flipped_formatted.csv \
                        --batch_size 10

python -m cellcontrast  --embed_image_data \
                        --raw_subset_dir "${experiment}" \
                        --foundation_model 'eva' \
                        --output_name 'out/models/eva_model.pt' \
                        --channel_names SKIP Keratin8/18 CD45 CD3e DAPI
