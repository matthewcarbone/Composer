import json
import logging
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import chromadb
import requests
import xmltodict
from dateutil.relativedelta import relativedelta
from httpx import ConnectError
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain_core.documents import Document

# from joblib import Memory
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from omegaconf.dictconfig import DictConfig
from omegaconf.errors import ConfigAttributeError
from typing_extensions import Annotated, TypedDict

from composer.utils import Timer, get_file_hash

logger = logging.getLogger(__name__)
# memory = Memory(location=get_memory_dir(), verbose=1)
# logger.debug(f"joblib.Memory: {memory}")


@dataclass
class GrantsGist:
    hydra_conf: DictConfig

    @cached_property
    def today(self) -> str:
        """Returns the current date in YYYYMMDD format."""

        today_str = str(self.hydra_conf.today)
        logger.debug(f"today property accessed; returning {today_str}")
        return today_str

    @cached_property
    def max_date_dt(self) -> datetime:
        """Returns the max date to consider grants."""

        date = str(self.hydra_conf.protocol.config.max_date)
        max_dt = datetime.strptime(date, "%Y%m%d")
        logger.debug(f"max_date_dt property accessed; raw config={date}, parsed={max_dt}")
        return max_dt

    @cached_property
    def min_date_dt(self) -> datetime:
        """Returns the min date to consider grants. If a min_date is not
        provided, this defaults to 3 months prior to the max date (which is
        usually given by today)."""

        date = str(self.hydra_conf.protocol.config.min_date)
        if date is None:
            logger.info("min_date not provided; defaulting to 3 months prior to max date")
            default_min_dt = self.max_date_dt - relativedelta(months=3)
            logger.debug(f"Computed default min_date_dt={default_min_dt}")
            return default_min_dt
        parsed_dt = datetime.strptime(date, "%Y%m%d")
        logger.debug(f"min_date_dt property accessed; parsed min_date_dt={parsed_dt}")
        return parsed_dt

    @cached_property
    def data_dir(self) -> str:
        """Provides the path to the data directory."""

        g = self.hydra_conf.protocol.config.data_dir
        Path(g).mkdir(exist_ok=True, parents=True)
        logger.debug(f"Ensured data dir exists at {g}")
        return g

    @cached_property
    def chroma_dir(self) -> str:
        """Provides the path to the Chroma datastore directory."""

        g = self.hydra_conf.protocol.config.chroma_dir
        Path(g).mkdir(exist_ok=True, parents=True)
        logger.debug(f"Ensured chroma dir exists at {g}")
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
        path_as_xml = str(Path(self.data_dir) / self.filename) + ".xml"
        logger.debug(f"filename_full_path_as_xml property accessed; returning {path_as_xml}")
        return path_as_xml

    @cached_property
    def filename_full_path_as_zip(self) -> str:
        path_as_zip = str(Path(self.data_dir) / self.filename) + ".zip"
        logger.debug(f"filename_full_path_as_zip property accessed; returning {path_as_zip}")
        return path_as_zip

    @cached_property
    def url_target(self) -> str:
        url = f"{self.url_base}/{self.filename}.zip"
        logger.debug(f"url_target property accessed; returning {url}")
        return url

    def pull_and_unzip(self) -> None:
        # If XML file already exists, skip the download step
        if Path(self.filename_full_path_as_xml).exists():
            logger.info(f"Done: pull_and_unzip: XML file ({self.filename_full_path_as_xml}) already exists")
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
            logger.debug(f"Extracting zip file from {self.filename_full_path_as_zip}...")
            with zipfile.ZipFile(self.filename_full_path_as_zip, "r") as zf:
                zf.extractall(self.data_dir)
            logger.info(f"Zip file extracted into {self.data_dir}")

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
            # Note that the dates in the extract are in the format MMDDYYYY
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
        logger.debug("Writing JSON files to data_dir...")
        for key, list_of_opportunities in d_by_post_date.items():
            for opportunity in list_of_opportunities:
                opportunity_id = opportunity["OpportunityID"]
                filename = f"{key}-{opportunity_id}.json"
                filename = Path(self.data_dir) / filename
                with open(filename, "w") as f:
                    json.dump(opportunity, f, indent=4, sort_keys=True)
                counter += 1

        logger.info(f"Done: create_json_store (wrote {counter} JSON files)")

    def cleanup(self):
        # Remove any .zip files
        for path in Path(self.data_dir).glob("*.zip"):
            logger.info(f"Cleaning up: unlinking {path}")
            path.unlink()

        # Remove any .xml files
        for path in Path(self.data_dir).glob("*.xml"):
            if str(path) == self.filename_full_path_as_xml:
                continue
            logger.info(f"Cleaning up: unlinking {path}")
            path.unlink()

        logger.info("Done: cleanup")

    def run(self):
        logger.info("Starting GrantGistSync run workflow...")
        self.pull_and_unzip()
        self.create_json_store()
        self.cleanup()
        logger.info("GrantGistSync run workflow complete.")


def _to_str(value):
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, list):
        return ",".join(value)
    raise ValueError


def _load_single_json(file_path):
    with open(file_path, "r") as f:
        d = json.load(f)
    title = d.pop("OpportunityTitle")
    description = d.pop("Description")
    try:
        additional_info = d.pop("AdditionalInformationOnEligibility")
    except KeyError:
        additional_info = None
    d["source"] = str(file_path)
    d["hash"] = get_file_hash(file_path)

    # Assign each metadata item a timestamp which can be much
    # more easily queried.
    new_metadata = {}
    for key, value in d.items():
        if "Date" in key:
            try:
                datetime_obj = datetime.strptime(value, "%m%d%Y")
                timestamp = int(datetime_obj.timestamp())
                new_key = f"{key}TimeStamp"
                new_metadata[new_key] = timestamp
            except ValueError:
                pass
    d = {key: _to_str(value) for key, value in d.items()}
    d = {**d, **new_metadata}
    page_content = f"Title: {title}\nDescription: {description}"
    if additional_info is not None:
        page_content += f"\nAdditional eligibility information: {additional_info}"
    return Document(page_content=page_content, metadata=d)


def load_all_json(target_dir):
    """Loads all json files in the `target_directory` into a docs format.
    Note that the function is cached by date, so loading is quite fast
    after the first time. Joblib Memory is used to cache the results to
    disk as pickle files."""

    logger.debug(f"Starting load_all_json for target_dir={target_dir}")
    if not Path(target_dir).exists():
        logger.error(f"Target directory does not exist: {target_dir}")
        exit(1)

    with Timer() as timer:
        files = [_load_single_json(f) for f in Path(target_dir).glob("*.json")]

    dt = f"{timer.dt:.02f}"
    logger.info(f"Total {len(files)} JSON files loaded from '{target_dir}' in {dt} s")
    logger.debug("Finished load_all_json")
    return files


def _get_vector_store(embeddings, embeddings_name, persist_directory):
    collection_name = f"grantgist_vector_store_{embeddings_name}"
    logger.info(f"Vector store collection {collection_name} in {persist_directory}")

    client_settings = chromadb.Settings(
        is_persistent=True,
        persist_directory=persist_directory,
        anonymized_telemetry=False,
    )

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        client_settings=client_settings,
    )


def _instantiate_vector_store(
    embeddings,
    embeddings_name,
    persist_directory,
    docs=None,
    max_docs_per_embedding_call=None,
):
    """Loads or updates a Chroma vector store for given embedding type
    - Checks which docs are already present in the store
    - Adds new docs as needed
    """

    vector_store = _get_vector_store(embeddings, embeddings_name, persist_directory)

    all_data = vector_store.get()
    L = len(all_data["ids"])
    logger.info(f"Total {L} documents in vector store")

    if docs is not None:
        # Print debug for each doc
        for doc in docs:
            doc_metadata_source = doc.metadata["source"]
            json_hash = get_file_hash(doc_metadata_source)
            doc.metadata["hash"] = json_hash
            logger.debug(f"Processing doc with source: {doc_metadata_source} " f"file hash {json_hash}")

        # Determine which doc sources are already in the store
        vs_sources = {}
        for _metadata, _id in zip(all_data["metadatas"], all_data["ids"]):
            source = _metadata["source"]
            file_hash = _metadata["hash"]
            source_stem = str(Path(source).stem)
            vs_sources[source_stem] = (file_hash, _id)
            logger.debug(f"Data with source {source_stem} and hash (id) {file_hash} " f"({_id}) in db")
        logger.debug(f"Found {len(vs_sources)} sources in existing vector store.")

        # Filter out docs that already exist
        new_docs_to_add = []
        docs_to_remove = []
        for d in docs:
            source = str(Path(d.metadata["source"]).stem)
            file_hash = d.metadata["hash"]
            if source not in vs_sources.keys():
                new_docs_to_add.append(d)
                logger.debug(f"Doc {source} not in db, adding")
                continue
            (vs_hash, _id) = vs_sources[source]
            if vs_hash != file_hash:
                new_docs_to_add.append(d)
                docs_to_remove.append(_id)
                logger.warning(f"Doc {source} has changed and will be replaced")

        logger.info(
            f"Parsing {len(docs)} docs to load_vector_store; " f"{len(new_docs_to_add)} are new and will be added."
        )

        if len(docs_to_remove) > 0:
            logger.info(f"Deleting {len(docs_to_remove)} documents from vector store")
            logger.debug(f"Deleting {docs_to_remove}")
            vector_store.delete(ids=docs_to_remove)

        if len(new_docs_to_add) > 0:
            if max_docs_per_embedding_call is None or max_docs_per_embedding_call == -1:
                vector_store.add_documents(documents=new_docs_to_add)
            else:
                for ii in range(0, len(new_docs_to_add), max_docs_per_embedding_call):
                    chunk = new_docs_to_add[ii : ii + max_docs_per_embedding_call]
                    vector_store.add_documents(documents=chunk)
            logger.info("Added new docs to vector store.")
        else:
            logger.info("No new docs to add.")

        logger.debug("Finished load_vector_store")
    return vector_store


def update_vector_store(docs, hydra_conf):
    # Attempt to retrieve embeddings from Hydra config
    try:
        embeddings = hydra_conf.ai.embeddings
    except ConfigAttributeError as error:
        logger.error(
            "ConfigAttributeError - issue with accessing a hydra key. "
            "ai key not found - you probably need to use +ai=ollama, "
            "or something like that when calling composer. Full error: "
            f"\n{error}"
        )
        exit(1)

    # Prepare vector store
    logger.info("Initializing vector store...")
    embeddings_name = hydra_conf.ai.embeddings.model
    try:
        vector_store = _instantiate_vector_store(
            embeddings,
            embeddings_name,
            hydra_conf.protocol.config.chroma_dir,
            docs=docs,
            max_docs_per_embedding_call=hydra_conf.ai.config.max_docs_per_embedding_call,
        )
    except ConnectError as error:
        logger.error(
            "Connection via httpx refused using ai object "
            f"{hydra_conf.ai.embeddings}. You likely forgot to spin up an "
            "ollama server, or perhaps you forgot to port forward to an "
            f"external server. Full error:\n{error}"
        )
        exit(1)

    return vector_store


def load_vector_store(hydra_conf):
    return update_vector_store(docs=None, hydra_conf=hydra_conf)


class GrantGistIndex(GrantsGist):
    def run(self):
        """Run the indexing process:
        - Attempt to get embeddings from Hydra config
        - Load all JSON files
        - Print debug for each doc
        - Initialize vector store
        - Handle any connection errors
        """
        logger.info("Starting GrantGistIndex.run()")

        # Load JSON documents
        docs = load_all_json(self.data_dir)
        logger.debug(f"Loaded {len(docs)} docs in GrantGistIndex.run()")

        # Update the vector store
        update_vector_store(docs, self.hydra_conf)
        logger.info("GrantGistIndex.run() complete.")


def get_tool_agent(llm, vectorstore):
    @tool(response_format="content_and_artifact")
    def get_grant_information(
        query: Annotated[str, ..., "Search query to run."],
        k: Annotated[
            int,
            ...,
            "The number of queries to make to the database. Make sure this value makes sense. I.e., it should probably be at least the number of dates queried.",
        ],
        min_date: Annotated[
            Optional[str],
            ...,
            "The minimum date relevant to the query (should be a string in format YYYYMMDD with no dashes). If no minimum date is relevant to the query, leave this as it's default.",
        ] = None,
        max_date: Annotated[
            Optional[str],
            ...,
            "The maximum date relevant to the query (should be a string in format YYYYMMDD with no dashes). If no maximum date is relevant to the query, leave this as it's default.",
        ] = None,
    ) -> Tuple[str, List[Document]]:
        """Tool for retrieving grant information from a database of grants."""

        # Deal with dates
        dates_conditions = []
        if min_date is not None:
            tmp = {"PostDateTimeStamp": {"$gte": int(datetime.strptime(min_date, "%Y%m%d").timestamp())}}
            dates_conditions.append(tmp)
        if max_date is not None:
            tmp = {"PostDateTimeStamp": {"$lte": int(datetime.strptime(max_date, "%Y%m%d").timestamp())}}
            dates_conditions.append(tmp)

        if len(dates_conditions) == 2:
            where = {"$and": dates_conditions}
        elif len(dates_conditions) == 1:
            where = dates_conditions
        else:
            where = None

        search_kwargs = {"k": k, "filter": where}

        retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

        docs = retriever.invoke(query)
        serialized = "\n\n".join((f"Source: {doc.metadata}\n" f"Content: {doc.page_content}") for doc in docs)
        return serialized, docs

    tools = [get_grant_information]
    tool_node = ToolNode(tools)

    model_with_tools = llm.bind_tools(tools)

    def should_continue(state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    def agent(state: MessagesState):
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")

    app = workflow.compile()
    return app


class GrantGistSummarize(GrantsGist):
    def run(self):
        vector_store = load_vector_store(self.hydra_conf)
        prompt = self.hydra_conf.prompt
        if prompt is None:
            logger.error("prompt must be set when using Summarize")
            exit(1)
        llm = self.hydra_conf.ai.llm
        app = get_tool_agent(llm, vector_store)

        for chunk in app.stream(
            {
                "messages": [
                    (
                        "system",
                        "You are an expert grant-writer. If you don't feel that there are any relevant grants related to the question, please say so. When possible, provide the specific grant # of the grant you refer to in your response.",
                    ),
                    (
                        "human",
                        prompt,
                    ),
                ]
            },
            stream_mode="values",
        ):
            chunk["messages"][-1].pretty_print()
