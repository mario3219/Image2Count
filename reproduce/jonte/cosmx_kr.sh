#!/bin/bash

experiment=cosmx_kr

python -m cellcontrast  --image_preprocess \
                        --preprocess_dir "data/raw/${experiment}/" \
                        --channel_names MEMBRANE CYTOKERATIN CD20 CD3 DAPI \
                        --cell_cutout 100 \
                        --foundation_model 'kronos' \
                        --batch_size 10 \
                        --cell_ids cosmx_ids_flipped_formatted.csv

python -m cellcontrast  --embed_image_data \
                        --raw_subset_dir "${experiment}" \
                        --foundation_model 'kronos' \
                        --output_name 'out/models/kronos.pt' \
                        --channel_names MEMBRANE CYTOKERATIN CD20 CD3 DAPI
