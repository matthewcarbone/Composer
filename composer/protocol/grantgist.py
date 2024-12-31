import json
import logging
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests
import xmltodict
from httpx import ConnectError
from joblib import Memory
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from omegaconf.dictconfig import DictConfig
from omegaconf.errors import ConfigAttributeError

from composer.ai.tools import get_tool_agent
from composer.global_state import get_memory_dir
from composer.utils import Timer

logger = logging.getLogger(__name__)
memory = Memory(location=get_memory_dir(), verbose=1)
logger.debug(f"joblib.Memory: {memory}")


@dataclass
class GrantGistSync:
    """Handles grants.gov interface."""

    hydra_conf: DictConfig

    @property
    def filename(self):
        return self.hydra_conf.protocol.config.filename

    @property
    def url_base(self):
        return self.hydra_conf.protocol.config.url_base

    @property
    def grants_dir(self):
        g = self.hydra_conf.protocol.config.grants_dir
        Path(g).mkdir(exist_ok=True, parents=True)
        return g

    @property
    def filename_full_path_as_xml(self):
        return str(Path(self.grants_dir) / self.filename) + ".xml"

    @property
    def filename_full_path_as_zip(self):
        return str(Path(self.grants_dir) / self.filename) + ".zip"

    @property
    def url_target(self):
        return f"{self.url_base}/{self.filename}.zip"

    def pull_and_unzip(self):
        if Path(self.filename_full_path_as_xml).exists():
            f = self.filename_full_path_as_xml
            logger.info(f"Done: pull_and_unzip: xml file ({f}) already exists")
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
            with zipfile.ZipFile(self.filename_full_path_as_zip, "r") as zf:
                zf.extractall(self.grants_dir)
            logger.info(f"Zip file extracted in {self.grants_dir}")
        logger.info("Done: pull_and_unzip")

    def create_json_store(self):
        min_date = str(self.hydra_conf.protocol.config.index_from_post_date)
        min_date = datetime.strptime(min_date, "%Y%m%d")
        logger.info(f"Creating json store indexing from date: {min_date}")
        with Timer() as timer:
            with open(self.filename_full_path_as_xml) as file:
                data = file.read()
        logger.info(f"Read xml file into memory in {timer.dt:.02f} s")

        # Convert the xml file to dictionary
        # TODO: there has to be a more efficient way to do this, I'm just
        # not that comfortable with xml
        with Timer() as timer:
            d = xmltodict.parse(data)
        logger.info(f"Parsed xml file to dictionary in {timer.dt:.02f} s")

        d_by_post_date = defaultdict(list)
        for item in d["Grants"]["OpportunitySynopsisDetail_1_0"]:
            dt = datetime.strptime(item["PostDate"], "%m%d%Y")
            if dt < min_date:
                continue
            sane_date = dt.strftime("%Y%m%d")  # sane format
            d_by_post_date[sane_date].append(item)

        # Write all the json files
        counter = 0
        for key, list_of_opportunities in d_by_post_date.items():
            for opportunity in list_of_opportunities:
                opportunity_id = opportunity["OpportunityID"]
                filename = f"{key}-{opportunity_id}.json"
                filename = Path(self.grants_dir) / filename
                with open(filename, "w") as f:
                    json.dump(opportunity, f, indent=4, sort_keys=True)
                counter += 1
        logger.info(f"Done: create_json_store (wrote {counter} json files)")

    def cleanup(self):
        to_unlink = list(Path(self.grants_dir).glob("*.zip")) + list(
            Path(self.grants_dir).glob("*.xml")
        )
        for path in to_unlink:
            if str(path) == self.filename_full_path_as_xml:
                continue
            logger.info(f"Cleaning up: unlinking {path}")
            path.unlink()
        logger.info("Done: cleanup")

    def execute(self):
        self.pull_and_unzip()
        self.create_json_store()
        self.cleanup()


@memory.cache
def load_all_json(date, target_dir):
    """Loads all json files in the `target_directory` into a docs format.
    Note that the function is cached by date, so loading is quite fast after
    the first time. Joblib Memory is used to cache the results to disk as
    pickle files."""

    loader = DirectoryLoader(
        path=target_dir,
        glob="*.json",
        loader_cls=JSONLoader,
        loader_kwargs={"jq_schema": ".", "text_content": False},
    )
    files = loader.load()
    logger.info(f"Total {len(files)} json files loaded for {date}")
    return files


@memory.cache(ignore=["docs", "embeddings"])
def build_and_persist_vector_store(date, docs, embeddings, embeddings_name):
    collection_name = f"grantgist_vector_store_{embeddings_name}_{date}"
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=get_memory_dir(),
    )
    vector_store.add_documents(documents=docs)
    logger.info("Vector store created")
    return collection_name


def load_vector_store(date, docs, embeddings, embeddings_name):
    collection_name = build_and_persist_vector_store(
        date, docs, embeddings, embeddings_name
    )
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=get_memory_dir(),
    )
    return vector_store


@dataclass
class GrantGistSummarize:
    hydra_conf: DictConfig

    @property
    def grants_dir(self) -> str:
        g = self.hydra_conf.protocol.config.grants_dir
        if not Path(g).exists():
            logger.error(f"grantgist directory does not exist, run sync")
            exit(1)
        return g

    @property
    def date(self) -> str:
        return str(self.hydra_conf.date)

    def execute(self):
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
        docs = load_all_json(self.date, self.grants_dir)
        logger.info("Initialilzing vector store...")
        embeddings_name = self.hydra_conf.ai.embeddings.model
        try:
            vector_store = load_vector_store(
                self.date, docs, embeddings, embeddings_name
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
                        "Give me a summary of all grants related to rabbits.",
                    ),
                ]
            },
            stream_mode="values",
        ):
            chunk["messages"][-1].pretty_print()
