import hashlib
import json
import logging
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Literal, Optional, Tuple
from uuid import uuid4

import requests
import rich
import xmltodict
from dateutil.relativedelta import relativedelta
from hydra.conf import HydraConf
from joblib import Memory
from monty.json import MSONable, load
from omegaconf.dictconfig import DictConfig
from pathvalidate import sanitize_filename
from rich import print as print
from tqdm import tqdm
from typing_extensions import Annotated, TypedDict

from composer.global_state import get_memory_dir, get_verbosity
from composer.utils import Timer, get_file_hash

logger = logging.getLogger(__name__)
# logging.basicConfig(level="INFO")

memory = Memory(location=get_memory_dir(), verbose=int(get_verbosity()))


@memory.cache(ignore=["stream"])
def cached_get_request(url: str, stream: bool = True):
    """
    A GET request using a cached session.

    Args:
        url (str): The URL endpoint to send the GET request to.
        stream (bool, optional): Whether to stream the response. Defaults to True.

    Returns:
        Response: The response object from the GET request.

    Raises:
        HTTPError: If the request returned an unsuccessful status code.
    """
    logger.debug(f"Calling cached_get_request with url {url}")

    # Since requests is synchronous, run in a thread to avoid blocking event loop.
    response = requests.get(url, stream=stream)
    if response.status_code != 200:
        logger.warning(f"GET request returned status code {response.status_code} for {url}")
    response.raise_for_status()
    logger.debug(f"Received response (status code {response.status_code}) for GET {url}.")
    return response


@memory.cache
def cached_post_request(url: str, **params):
    """
    A POST request using a cached session.

    Args:
        url (str): The URL endpoint to send the POST request to.
        **params: Arbitrary keyword arguments for the POST request body or settings.

    Returns:
        Response: The response object from the POST request.

    Raises:
        HTTPError: If the request returned an unsuccessful status code.
    """
    logger.debug(f"Calling cached_post_request with url {url} and params {params}")

    # Since requests is synchronous, run in a thread to avoid blocking event loop.
    response = requests.post(url, **params)
    if response.status_code != 200:
        logger.warning(f"POST request returned status code {response.status_code} for {url}")
    response.raise_for_status()
    logger.debug(f"Received response (status code {response.status_code}) for POST {url}.")
    return response


@memory.cache(ignore=["data"])
def cached_xmltodict(date, data):
    return xmltodict.parse(data)


def pull_grants_gov_extract(hydra_conf: DictConfig):
    """
    Pull the grants.gov extract to a temporary directory, then extract
    it into the metadata directory. The requests calls are cached, so
    the XML file itself is not saved in the cache. The metadata directory
    will contain:

    - A JSON file for each opportunity (with metadata).
    - A 'hash_state.json' that contains file hashes to track changes.

    Args:
        hydra_conf (DictConfig): The Hydra configuration object that contains
            protocol configuration settings, such as root paths, URLs, filenames,
            date ranges, etc.
    """

    today = hydra_conf.today
    conf = hydra_conf.protocol.config
    root = Path(conf.root)
    metadata_path = root / "metadata"
    metadata_path.mkdir(exist_ok=True, parents=True)
    min_date = datetime.strptime(str(conf.min_date), "%Y%m%d")
    max_date = datetime.strptime(str(conf.max_date), "%Y%m%d")

    extracts_filename = conf.extracts_filename

    with TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        file_xml = tempdir / f"{extracts_filename}.xml"
        file_zip = tempdir / f"{extracts_filename}.zip"

        # Download and extract the zip file
        url = f"{conf.urls.extracts}/{extracts_filename}.zip"
        with Timer() as timer:
            logger.info(f"Fetching ZIP file from {url} or cache...")
            response = cached_get_request(url, stream=True)

            # Write ZIP to disk in chunks
            with open(file_zip, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        logger.debug("Received an empty chunk while writing ZIP file.")
                        continue
                    f.write(chunk)
            logger.debug(f"Finished writing ZIP to {file_zip}")

            # Extract ZIP
            try:
                with zipfile.ZipFile(file_zip) as zf:
                    logger.debug(f"Extracting {file_zip} to {tempdir}")
                    zf.extractall(tempdir)
            except zipfile.BadZipFile as e:
                logger.error(f"Failed to extract {file_zip}: {e}")
                return  # Exit early if ZIP is invalid

        logger.info(f"XML file pulled and extracted in {timer.elapsed:.02f} s")

        with Timer() as timer:
            if not file_xml.exists():
                logger.error(f"Expected XML file {file_xml} not found after extraction.")
                return
            with open(file_xml, "r", encoding="utf-8") as f:
                data = f.read()
        logger.info(f"XML file read in {timer.elapsed:.02f} s")

    # Convert XML to dictionary
    with Timer() as timer:
        data = cached_xmltodict(today, data)
    logger.info(f"XML file parsed to dictionary in {timer.elapsed:.02f} s")

    data = data["Grants"]["OpportunitySynopsisDetail_1_0"]
    logger.info("Saving JSON metadata files for each valid opportunity...")

    # Save all of the metadata inside the date range
    with Timer() as timer:
        count_saved = 0
        for opportunity in data:
            post_date = datetime.strptime(opportunity["PostDate"], "%m%d%Y")
            if min_date <= post_date <= max_date:
                uid = opportunity["OpportunityID"]
                path = metadata_path / f"{uid}.json"
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(opportunity, f, indent=4)
                count_saved += 1
        logger.debug(f"Number of opportunities saved: {count_saved}")

    logger.info(f"JSON files written in {timer.elapsed:.02f} s")


def pull_details(hydra_conf: DictConfig):
    """
    Pull additional details for each opportunity within the configured date range
    and update the local JSON metadata files.

    Args:
        hydra_conf (DictConfig): The Hydra configuration object that contains
            protocol configuration settings, such as root paths, URLs, filenames,
            date ranges, etc.
    """

    conf = hydra_conf.protocol.config
    root = Path(conf.root)
    metadata_path = root / "metadata"
    metadata_path.mkdir(exist_ok=True, parents=True)
    min_date = datetime.strptime(str(conf.min_date), "%Y%m%d")
    max_date = datetime.strptime(str(conf.max_date), "%Y%m%d")
    url = conf.urls.details

    count_updated = 0
    for path in metadata_path.glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            opportunity = json.load(f)

        post_date_str = opportunity.get("PostDate")
        if not post_date_str:
            logger.debug(f"Missing 'PostDate' in {path.name}, skipping.")
            continue

        post_date = datetime.strptime(post_date_str, "%m%d%Y")

        if min_date <= post_date <= max_date:
            logger.debug(f"Pulling information for {path.name}")
            params = {"data": {"oppId": opportunity["OpportunityID"]}}
            response = cached_post_request(url, **params)
            if response is None:
                logger.warning(f"Received None response for {path.name}, skipping update.")
                continue

            opportunity["@details"] = response.json()

            with open(path, "w", encoding="utf-8") as f:
                json.dump(opportunity, f, indent=4)
            count_updated += 1

    logger.info(f"Number of opportunities updated with details: {count_updated}")


def pull_pdfs(hydra_conf: DictConfig):
    conf = hydra_conf.protocol.config
    root = Path(conf.root)
    documents_path = root / "documents"
    documents_path.mkdir(exist_ok=True, parents=True)
    min_date = datetime.strptime(str(conf.min_date), "%Y%m%d")
    max_date = datetime.strptime(str(conf.max_date), "%Y%m%d")
    url = conf.urls.download
