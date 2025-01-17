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

import requests_cache
import rich
import xmltodict
from dateutil.relativedelta import relativedelta
from joblib import Memory
from monty.json import MSONable, load
from omegaconf.dictconfig import DictConfig
from pathvalidate import sanitize_filename
from rich import print as print
from tqdm import tqdm
from typing_extensions import Annotated, TypedDict

from composer.global_state import get_memory_dir
from composer.utils import Timer, get_file_hash

logger = logging.getLogger(__name__)
# logging.basicConfig(level="INFO")

memory = Memory(location=get_memory_dir(), verbose=0)


@dataclass
class Document(MSONable):
    metadata: Dict[str, Any]
    raw_bytes: Optional[bytes] = None

    @property
    def filename(self) -> str:
        return sanitize_filename(self.metadata["fileName"]).replace(" ", "_")

    @property
    def uid(self) -> str:
        """The unique identifier of the document itself. Used in the REST query."""

        return str(self.metadata["id"])

    def __str__(self):
        return self.filename

    def __repr__(self):
        return self.__str__()

    def __rich_repr__(self):
        yield "filename", self.filename
        yield "available", self.raw_bytes is not None

    def as_dict(self):
        return {"filename": self.filename}


@dataclass
class Opportunity(MSONable):
    metadata: Dict[str, Any]
    documents: Optional[Dict[str, bytes]] = None

    @property
    def pdf_attachments(self) -> Dict[str, Dict[str, str]]:
        return {
            attachment["id"]: attachment
            for folder in self.metadata["synopsisAttachmentFolders"]
            for attachment in folder["synopsisAttachments"]
            if "application/pdf" == attachment["mimeType"] and ".pdf" in attachment["fileName"]
        }

    def populate_documents(self, hydra_conf):
        conf = hydra_conf.protocol.config
        session = conf.requests_cache
        pdf_attachments = self.pdf_attachments
        documents = {}
        for uid, information in pdf_attachments.items():
            response = session.get(f"{conf.urls.download}/{uid}")
            response.raise_for_status()
            documents[str(uid)] = Document(metadata=information, raw_bytes=response.content)
        self.documents = documents

    @classmethod
    def from_opp_hit(cls, hydra_conf, opp_hit):
        conf = hydra_conf.protocol.config
        session = conf.requests_cache
        params = {"oppId": opp_hit["OpportunityID"]}
        response = session.post(conf.urls.details, data=params)
        response.raise_for_status()
        metadata = response.json()
        metadata["@oppHit"] = opp_hit
        klass = cls(metadata=metadata)
        klass.populate_documents(hydra_conf)
        return klass

    @property
    def uid(self) -> str:
        return str(self.metadata["@oppHit"]["id"])

    @property
    def date(self):
        date = self.metadata["@oppHit"]["openDate"]
        return datetime.strptime(date, "%m/%d/%Y").strftime("%Y%m%d")


@dataclass
class GrantGistCore(MSONable):
    metadata: Dict[str, Any]
    opportunities: List[Opportunity]

    @classmethod
    def from_hydra_config(cls, hydra_conf):
        """Instantiates a GrantGistCore class from a hydra config file. This
        clasmethod also pulls the XML file extract from grants.gov."""

        logger.info("Starting GrantGistCore.from_hydra_config method.")
        verbose = HydraConfig.get().verbose

        # Get the paths and construct the relevant filesystem
        today = hydra_conf.today
        conf = hydra_conf.protocol.config
        root = Path(conf.root)
        core_path = root / "core"

        logger.debug(f"Ensuring the core path '{core_path}' exists.")
        core_path.mkdir(exist_ok=True, parents=True)

        core_file = core_path / f"{today}.json"
        if core_file.exists():
            logger.info(f"Serialized core cache found. Loading from {core_file}")
            return load(core_file)
        else:
            logger.info(f"No serialized core cache found at {core_file}. Creating a new one.")

        extracts_filename = conf.extracts_filename
        session = conf.requests_cache
        # if verbose:
        #     logging.getLogger("requests_cache").setLevel("DEBUG")
        #     logger.debug(f"Set logging level for 'requests_cache' to DEBUG.")

        with TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir)
            file_xml = tempdir / f"{extracts_filename}.xml"
            file_zip = tempdir / f"{extracts_filename}.zip"

            # Download and extract the zip file
            url = f"{conf.urls.extracts}/{extracts_filename}.zip"
            logger.info(f"Downloading extracts from {url}")
            with Timer() as timer:
                response = session.get(url, stream=True)
                response.raise_for_status()
                with open(file_zip, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                logger.debug(f"Finished writing {file_zip} to disk.")
                with zipfile.ZipFile(file_zip) as zf:
                    zf.extractall(tempdir)
            logger.debug(f"Extraction of {file_zip} completed in {timer.elapsed:.02f} seconds.")

            # Read the XML data into a dictionary
            logger.info("Parsing XML to dictionary.")
            with Timer() as timer:
                with open(file_xml, "r") as f:
                    data = f.read()
            logger.debug(f"XML file {file_xml} read into memory in {timer.elapsed:.02f} seconds.")

        # Parse the XML data into a Python dictionary
        with Timer() as timer:
            extract_as_dict = xmltodict.parse(data)
        list_of_dict = extract_as_dict["Grants"]["OpportunitySynopsisDetail_1_0"]
        logger.debug(f"xmltodict.parse completed in {timer.elapsed:.2f} seconds.")

        # Parse documents
        opportunities = []
        for opp_hit in list_of_dict:
            opportunity = Opportunity.from_opp_hit(hydra_conf, opp_hit)
            opportunities.append(opportunity)

        klass = cls(metadata=extract_as_dict, opportunities=opportunities)
        logger.debug("Saving newly created GrantGistCore object.")
        klass.save(core_file, json_kwargs={"sort_keys": True, "indent": 4})
        logger.info(f"GrantGistCore object saved to {core_file}.")

        return klass


@memory.cache(ignore=["session", "stream"])
def cached_get_request(session, url, stream=True):
    logger.debug(f"Calling cached_get_request with url {url}")
    response = session.get(url, stream=stream)
    response.raise_for_status()
    return response


@memory.cache(ignore=["session"])
def cached_post_request(session, url, **params):
    logger.debug(f"Calling cached_put_request with url {url} and params {params}")
    response = session.post(url, **params)
    response.raise_for_status()
    return response


def pull_grants_gov_extract(hydra_conf: DictConfig):
    """Pulls the grants.gov extract to a temporary directory, then extracts
    it into the metadata directory. Note that the requests call is cached,
    so the XML file itself is not saved. The metadata directory will contain
    a json file for each opportunity, and will contain the metadata for
    that opportunity. In addition, `hash_state.json` will contain the
    hashes for each file, so that changes can be tracked."""

    logger.info("Starting: pull_grants_gov_extract...")

    conf = hydra_conf.protocol.config
    root = Path(conf.root)
    metadata_path = root / "metadata"
    metadata_path.mkdir(exist_ok=True, parents=True)
    min_date = datetime.strptime(str(conf.min_date), "%Y%m%d")
    max_date = datetime.strptime(str(conf.max_date), "%Y%m%d")

    extracts_filename = conf.extracts_filename
    session = conf.requests_cache

    with TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        file_xml = tempdir / f"{extracts_filename}.xml"
        file_zip = tempdir / f"{extracts_filename}.zip"

        # Download and extract the zip file
        url = f"{conf.urls.extracts}/{extracts_filename}.zip"
        with Timer() as timer:
            response = cached_get_request(session, url, stream=True)
            with open(file_zip, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.debug(f"Finished writing {file_zip} to disk.")
            with zipfile.ZipFile(file_zip) as zf:
                zf.extractall(tempdir)
        logger.info(f"XML file pulled and extracted in {timer.elapsed:.02f} s")

        with Timer() as timer:
            with open(file_xml, "r") as f:
                data = f.read()
        logger.info(f"XML file read in {timer.elapsed:.02f} s")

    with Timer() as timer:
        data = xmltodict.parse(data)
    logger.info(f"XML file parsed to dictionary in {timer.elapsed:.02f} s")
    data = data["Grants"]["OpportunitySynopsisDetail_1_0"]

    # Save all of the metadata inside the date range
    with Timer() as timer:
        for opportunity in data:
            post_date = datetime.strptime(opportunity["PostDate"], "%m%d%Y")
            if min_date <= post_date <= max_date:
                uid = opportunity["OpportunityID"]
                path = metadata_path / f"{uid}.json"
                with open(path, "w") as f:
                    json.dump(opportunity, f, indent=4)
    logger.info(f"JSON files written in {timer.elapsed:.02f} s")
    logger.info("Done: pull_grants_gov_extract")


def pull_details(hydra_conf: DictConfig):
    logger.info("Starting: pull_details")

    conf = hydra_conf.protocol.config
    root = Path(conf.root)
    metadata_path = root / "metadata"
    metadata_path.mkdir(exist_ok=True, parents=True)
    session = conf.requests_cache
    min_date = datetime.strptime(str(conf.min_date), "%Y%m%d")
    max_date = datetime.strptime(str(conf.max_date), "%Y%m%d")
    url = conf.urls.details

    with Timer() as timer:
        for path in Path(metadata_path).glob("*.json"):
            with open(path, "r") as f:
                opportunity = json.load(f)
            post_date = opportunity["PostDate"]
            post_date = datetime.strptime(post_date, "%m%d%Y")

            if min_date <= post_date <= max_date:
                logger.debug(f"Pulling information for {path.name}")
                params = {"data": {"oppId": opportunity["OpportunityID"]}}
                response = cached_post_request(session, url, **params)
                opportunity["@details"] = response.json()

                with open(path, "w") as f:
                    json.dump(opportunity, f, indent=4)

    logger.info(f"Done: pull_details in {timer.elapsed:.02f} s")
