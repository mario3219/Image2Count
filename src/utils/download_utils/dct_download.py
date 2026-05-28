import os
import requests
from pathlib import Path
from hashlib import md5
from tqdm import tqdm
import logging
import tarfile
import zipfile
import torch

_latest = "2025-06-09"
_model_registry = {
    # Original model version uploaded with preprint
    "specific_ct_v0.1": "e499da92509821161be88a47237960a9",
    # Versions released June 9th 2025. The public-data-only version is trained
    # only on the subset of data that is publicly available (for reproducibility).
    # Users are recommended to use the *non* public-data-only option.
    "2025-06-09": "19b669675c06816414e8677f542ff542",
    "2025-06-09_public-data-only": "19b669675c06816414e8677f542ff542",
}

"""User interface to authentication layer for data/models."""

_api_endpoint = "https://users.deepcell.org/api/getData/"
_asset_location = Path.home() / ".deepcell"

def dct_download(download_location,*, version=None):
    """Download the deepcell-types model for local use.

    Parameters
    ----------
    version : str, optional
       Which version of the model to download. Default is `None`, which results
       in the latest (i.e. most-recently-released) version being downloaded.
    """

    version = version if version is not None else _latest
    asset_key = f"models/deepcell-types_{version}.pt"

    fetch_data(
        asset_key, cache_subdir="models", file_hash=_model_registry.get(version),
        download_location=download_location)

    dct_modify(download_location)

def dct_modify(model_path):

    WEIGHT_PATH = os.path.join(model_path,'deepcell-types_2025-06-09.pt')
    weights = torch.load(WEIGHT_PATH,
                   map_location=torch.device('cpu')
                   )

    SUB_STRINGS = ['image_encoder.']
    for SUB_STRING in SUB_STRINGS:
        weights = {
            k.replace(SUB_STRING,''): v
            for k,v in weights.items()
        }

    KEYS = ["logit_scale", "ct_embedding.weight", "image_adaptor.0.weight",
            "image_adaptor.0.bias", "text_adaptor.weight", "text_adaptor.bias",
            "classification_head."] 
    for key in list(weights.keys()):
        for KEY_NAME in KEYS:
            if KEY_NAME in key:
                weights.pop(key)

    modified_path = os.path.join(model_path, "deepcell_types_modified.pt")
    torch.save(weights, modified_path)

    if os.path.exists(os.path.join(model_path,"deepcell-types_2025-06-09.pt")):
        os.remove(os.path.join(model_path,"deepcell-types_2025-06-09.pt"))

def fetch_data(asset_key: str, cache_subdir=None, file_hash=None, download_location=None):
    """Fetch assets through users.deepcell.org authentication system.

    Download assets from the deepcell suite of datasets and models which
    require user-authentication.

    .. note::

       You must have a Deepcell Access Token set as an environment variable
       with the name ``DEEPCELL_ACCESS_TOKEN`` in order to access assets.

       Access tokens can be created at <https://users.deepcell.org>_

    Args:
        :param asset_key: Key of the file to download.
        The list of available assets can be found on the users.deepcell.org
        homepage.

        :param cache_subdir: `str` indicating directory relative to
        `~/.deepcell` where downloaded data will be cached. The default is
        `None`, which means cache the data in `~/.deepcell`.

        :param file_hash: `str` represented the md5 checksum of datafile. The
        checksum is used to perform data caching. If no checksum is provided or
        the checksum differs from that found in the data cache, the data will
        be (re)-downloaded.
    """
    logging.basicConfig(level=logging.INFO)

#    if cache_subdir is not None:
#        download_location /= cache_subdir
#    download_location.mkdir(exist_ok=True, parents=True)

    # Extract the filename from the asset_key, which can be a full path
    fname = os.path.split(asset_key)[-1]
#    fpath = download_location / fname
    fpath = os.path.join(download_location, fname)

    # Check for cached data
    if file_hash is not None:
        logging.info('Checking for cached data')
        try:
            with open(fpath, "rb") as fh:
                hasher = md5(fh.read())
            logging.info(f"Checking {fname} against provided file_hash...")
            md5sum = hasher.hexdigest()
            if md5sum == file_hash:
                logging.info(
                    f"{fname} with hash {file_hash} already available."
                )
                return fpath
            logging.info(
                f"{fname} with hash {file_hash} not found in {download_location}"
            )
        except FileNotFoundError:
            pass

    # Check for access token
    access_token = os.environ.get("DEEPCELL_ACCESS_TOKEN")
    if access_token is None:
        raise ValueError(
            "\nDEEPCELL_ACCESS_TOKEN not found.\n"
            "Please set your access token to the DEEPCELL_ACCESS_TOKEN\n"
            "environment variable.\n"
            "For example:\n\n"
            "\texport DEEPCELL_ACCESS_TOKEN=<your-token>.\n\n"
            "If you don't yet have a token, you can create one at\n"
            "https://users.deepcell.org"
        )

    # Request download URL
    headers = {"X-Api-Key": access_token}
    logging.info("Making request to server")
    resp = requests.post(
        _api_endpoint, headers=headers, data={"s3_key": asset_key}
    )
    # Raise informative exception for the specific case when the asset_key is
    # not found in the bucket
    if resp.status_code == 404 and resp.json().get("error") == "Key not found":
        raise ValueError(f"Object {asset_key} not found.")
    # Raise informative exception for the specific case when an invalid
    # API token is provided.
    if resp.status_code == 403 and (
       resp.json().get("detail") == "Authentication credentials were not provided."
    ):
        raise ValueError(
            f"\n\nAPI token {access_token} is not valid.\n"
            "The token may be expired - if so, create a new one at\n"
            "https://users.deepcell.org"
        )
    # Handle all other non-http-200 status
    resp.raise_for_status()

    # Parse response
    response_data = resp.json()
    download_url = response_data["url"]
    file_size = response_data["size"]
    # Parse file_size (TODO: would be more convenient if it were numerical, i.e. always bytes)
    val, suff = file_size.split(" ")
    # TODO: Case statement would be awesome here, but need to support all the
    # way back to Python 3.8
    suffix_mapping = {"KB": 2**10, "MB": 2**20, "B": 1, "GB": 2**30, "TB": 2**40}
    file_size_numerical = int(float(val) * suffix_mapping[suff])

    logging.info(
        f"Downloading {asset_key} with size {file_size} to {download_location}"
    )
    data_req = requests.get(
        download_url, headers={"user-agent": "Wget/1.20 (linux-gnu)"}, stream=True
    )
    data_req.raise_for_status()

    chunk_size = 4096
    with tqdm.wrapattr(
        open(fpath, "wb"), "write", miniters=1, total=file_size_numerical
    ) as fh:
        for chunk in data_req.iter_content(chunk_size=chunk_size):
            fh.write(chunk)

    logging.info(f"🎉 Successfully downloaded file to {fpath}")

    return fpath
