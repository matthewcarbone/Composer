import json
import logging
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path

import requests
import xmltodict
from dateutil.relativedelta import relativedelta
from httpx import ConnectError

# from joblib import Memory
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from omegaconf.dictconfig import DictConfig
from omegaconf.errors import ConfigAttributeError

from composer.ai.tools import get_tool_agent
from composer.global_state import get_memory_dir
from composer.utils import Timer

logger = logging.getLogger(__name__)
# memory = Memory(location=get_memory_dir(), verbose=1)
# logger.debug(f"joblib.Memory: {memory}")


@dataclass
class GrantsGist:
    hydra_conf: DictConfig

    @cached_property
    def today(self) -> str:
        today_str = str(self.hydra_conf.today)
        logger.debug(f"today property accessed; returning {today_str}")
        return today_str

    @cached_property
    def max_date_dt(self) -> datetime:
        date = self.hydra_conf.protocol.config.max_date
        max_dt = datetime.strptime(date, "%Y%m%d")
        logger.debug(
            f"max_date_dt property accessed; raw config={date}, parsed={max_dt}"
        )
        return max_dt

    @cached_property
    def min_date_dt(self) -> datetime:
        date = self.hydra_conf.protocol.config.min_date
        if date is None:
            logger.info(
                "min_date not provided; defaulting to 3 months prior to max date"
            )
            default_min_dt = self.max_date_dt - relativedelta(months=3)
            logger.debug(f"Computed default min_date_dt={default_min_dt}")
            return default_min_dt
        parsed_dt = datetime.strptime(date, "%Y%m%d")
        logger.debug(
            f"min_date_dt property accessed; parsed min_date_dt={parsed_dt}"
        )
        return parsed_dt

    @cached_property
    def grants_dir(self):
        g = self.hydra_conf.protocol.config.grants_dir
        # Make sure directory exists
        Path(g).mkdir(exist_ok=True, parents=True)
        logger.debug(f"Ensured grants_dir exists at {g}")
        return g


class GrantGistSync(GrantsGist):
    """Handles interfacing to grants.gov and pulling the targeted xml extract files."""

    @cached_property
    def filename(self) -> str:
        filename = self.hydra_conf.protocol.config.filename
        logger.debug(f"filename property accessed; returning {filename}")
        return filename

    @cached_property
    def url_base(self) -> str:
        url_base = self.hydra_conf.protocol.config.url_base
        logger.debug(f"url_base property accessed; returning {url_base}")
        return url_base

    @cached_property
    def filename_full_path_as_xml(self) -> str:
        path_as_xml = str(Path(self.grants_dir) / self.filename) + ".xml"
        logger.debug(
            f"filename_full_path_as_xml property accessed; returning {path_as_xml}"
        )
        return path_as_xml

    @cached_property
    def filename_full_path_as_zip(self) -> str:
        path_as_zip = str(Path(self.grants_dir) / self.filename) + ".zip"
        logger.debug(
            f"filename_full_path_as_zip property accessed; returning {path_as_zip}"
        )
        return path_as_zip

    @cached_property
    def url_target(self) -> str:
        url = f"{self.url_base}/{self.filename}.zip"
        logger.debug(f"url_target property accessed; returning {url}")
        return url

    def pull_and_unzip(self) -> None:
        # If XML file already exists, skip the download step
        if Path(self.filename_full_path_as_xml).exists():
            logger.info(
                f"Done: pull_and_unzip: XML file ({self.filename_full_path_as_xml}) already exists"
            )
            return
        else:
            logger.info(f"Pulling data from {self.url_target}")
            response = requests.get(self.url_target, stream=True)
            response.raise_for_status()
            with open(self.filename_full_path_as_zip, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info(f"Zip file saved to {self.filename_full_path_as_zip}")

            # Extract the zip file
            logger.debug(
                f"Extracting zip file from {self.filename_full_path_as_zip}..."
            )
            with zipfile.ZipFile(self.filename_full_path_as_zip, "r") as zf:
                zf.extractall(self.grants_dir)
            logger.info(f"Zip file extracted into {self.grants_dir}")

        logger.info("Done: pull_and_unzip")

    def create_json_store(self):
        # Time reading and parsing of the XML file
        with Timer() as timer:
            with open(self.filename_full_path_as_xml) as file:
                data = file.read()
        logger.info(f"Read XML file into memory in {timer.dt:.02f} s")

        # Convert XML to dictionary
        with Timer() as timer:
            d = xmltodict.parse(data)
        logger.info(f"Parsed XML file to dictionary in {timer.dt:.02f} s")

        # Organize items by PostDate
        d_by_post_date = defaultdict(list)
        items = d["Grants"]["OpportunitySynopsisDetail_1_0"]
        logger.debug(f"Found {len(items)} items in the XML structure")

        for item in items:
            dt = datetime.strptime(item["PostDate"], "%m%d%Y")
            # Filter out-of-range items
            if dt < self.min_date_dt or dt > self.max_date_dt:
                logger.debug(
                    f"Skipping item with PostDate={dt} (outside range {self.min_date_dt} - {self.max_date_dt})"
                )
                continue
            sane_date = dt.strftime("%Y%m%d")  # a consistent format
            d_by_post_date[sane_date].append(item)

        # Write all JSON files
        counter = 0
        logger.debug("Writing JSON files to grants_dir...")
        for key, list_of_opportunities in d_by_post_date.items():
            for opportunity in list_of_opportunities:
                opportunity_id = opportunity["OpportunityID"]
                filename = f"{key}-{opportunity_id}.json"
                filename = Path(self.grants_dir) / filename
                with open(filename, "w") as f:
                    json.dump(opportunity, f, indent=4, sort_keys=True)
                counter += 1

        logger.info(f"Done: create_json_store (wrote {counter} JSON files)")

    def cleanup(self):
        # Remove any .zip files
        for path in Path(self.grants_dir).glob("*.zip"):
            logger.info(f"Cleaning up: unlinking {path}")
            path.unlink()

        # Remove any .xml files
        for path in Path(self.grants_dir).glob("*.xml"):
            logger.info(f"Cleaning up: unlinking {path}")
            path.unlink()

        logger.info("Done: cleanup")

    def run(self):
        logger.info("Starting GrantGistSync run workflow...")
        self.pull_and_unzip()
        self.create_json_store()
        self.cleanup()
        logger.info("GrantGistSync run workflow complete.")


def load_all_json(target_dir):
    """Loads all json files in the `target_directory` into a docs format.
    Note that the function is cached by date, so loading is quite fast after
    the first time. Joblib Memory is used to cache the results to disk as
    pickle files."""

    with Timer() as timer:
        loader = DirectoryLoader(
            path=target_dir,
            glob="*.json",
            loader_cls=JSONLoader,
            loader_kwargs={"jq_schema": ".", "text_content": False},
        )
        files = loader.load()
    dt = f"{timer.dt:.02f}"
    logger.info(f"Total {len(files)} json files loaded in {dt} s")
    return files


def load_vector_store(date, docs, embeddings, embeddings_name):
    collection_name = f"grantgist_vector_store_{embeddings_name}_{date}"
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=get_memory_dir(),
    )
    vs_sources = set()
    all_data = vector_store._collection.get()
    for metadata in all_data["metadatas"]:
        source = metadata["source"]
        source = str(Path(source).stem)
        vs_sources.add(source)
    logger.info(f"Passing {len(docs)} docs to load_vector_store")
    docs_to_add = [
        d
        for d in docs
        if str(Path(d.metadata["source"]).stem) not in vs_sources
    ]
    logger.info(f"{len(docs_to_add)} docs not already in vector store")
    if len(docs_to_add) > 0:
        vector_store.add_documents(documents=docs_to_add)
        logger.info("Added new docs to vector store")
    else:
        logger.info("No new docs to add")
    return vector_store


class GrantGistIndex(GrantsGist):
    @property
    def min_date(self):
        return self.hydra_conf.protocol.config.min_date

    @property
    def max_date(self):
        return self.hydra_conf.protocol.config.max_date

    def run(self):
        try:
            embeddings = self.hydra_conf.ai.embeddings
        except ConfigAttributeError as error:
            logger.critical(
                "ai key not found - you probably need to use +ai=ollama, "
                "or something like that when calling composer. Full error: "
                f"\n{error}"
            )
            exit(1)

        docs = load_all_json(self.grants_dir)


class GrantGistSummarize(GrantsGist):
    def run(self):
        try:
            llm = self.hydra_conf.ai.llm
            embeddings = self.hydra_conf.ai.embeddings
        except ConfigAttributeError as error:
            logger.critical(
                "ai key not found - you probably need to use +ai=ollama, "
                "or something like that when calling composer. Full error: "
                f"\n{error}"
            )
            exit(1)
        docs = load_all_json(self.grants_dir)
        for doc in docs:
            doc_metadata_source = doc.metadata["source"]
            logger.debug(f"Loading doc from {doc_metadata_source}")
        logger.info("Initialilzing vector store...")
        embeddings_name = self.hydra_conf.ai.embeddings.model
        try:
            vector_store = load_vector_store(
                self.today, docs, embeddings, embeddings_name
            )
        except ConnectError as error:
            logger.critical(
                "Connection via httpx refused using ai object "
                f"{self.hydra_conf.ai.embeddings}. You likely forgot to spin up an "
                "ollama server, or perhaps you forgot to port forward to an "
                f"external server. Full error: \n{error}"
            )
            exit(1)

        app = get_tool_agent(llm, vector_store)

        for chunk in app.stream(
            {
                "messages": [
                    (
                        "system",
                        "You are an expert grant-writer, specializing in Department of Energy (EN) grants. When asked for summaries or asked questions, you will focus on Department of Energy grants primarily, unless explicitly asked to give feedback for all Government departments. If you don't feel that there are any relevant grants related to the question, please say so.",
                    ),
                    (
                        "human",
                        "Are there any recently posted grants about workforce development? Pay particular attention to RENEW grants.",
                    ),
                ]
            },
            stream_mode="values",
        ):
            chunk["messages"][-1].pretty_print()
