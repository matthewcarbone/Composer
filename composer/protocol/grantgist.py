import json
import logging
import zipfile
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Literal, Optional, Tuple
from uuid import uuid4

import requests
import rich
import xmltodict
from joblib import Memory, Parallel, delayed
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from omegaconf.dictconfig import DictConfig
from pathvalidate import sanitize_filename
from pydantic import BaseModel, Field
from rich import print as print
from tqdm import tqdm
from typing_extensions import Annotated, TypedDict

from composer.global_state import get_memory_dir, get_verbosity
from composer.utils import Timer, get_file_hash

logger = logging.getLogger(__name__)
memory = Memory(location=get_memory_dir(), verbose=int(get_verbosity()))

logging.getLogger("fontTools").setLevel(logging.WARNING)


@dataclass
class Params:
    """
    Provides convenient, cached access to commonly used parameters
    derived from a Hydra configuration object (`hydra_conf`).

    Each property is evaluated once and then cached. Debug-level logs
    are triggered whenever a property is first accessed, showing the
    computed values (except for the config itself, which really bloats the
    debug log).
    """

    hydra_conf: DictConfig

    @cached_property
    def today(self) -> str:
        """
        Returns:
            str: The 'today' value from `hydra_conf`, representing
            the current processing date.
        """
        value = self.hydra_conf.today
        logger.debug(f"Accessed 'today' property with value: {value}")
        return value

    @cached_property
    def conf(self) -> DictConfig:
        """
        Returns:
            DictConfig: The 'protocol.config' section from `hydra_conf`,
            containing configuration details (e.g., root paths, URLs,
            date ranges).
        """
        return self.hydra_conf.protocol.config

    @cached_property
    def extracts_filename(self) -> str:
        """
        Returns:
            str: The filename (without extension) for the extracts file,
            as specified in `conf`.
        """
        value = self.conf.extracts_filename
        logger.debug(f"Accessed 'extracts_filename' property with value: {value}")
        return value

    @cached_property
    def root(self) -> Path:
        """
        Returns:
            Path: The root directory for storing metadata, documents,
            and other downloaded artifacts. Created if it doesn't exist.
        """
        value = Path(self.conf.root)
        logger.debug(f"Accessed 'root' property with value: {value}")
        return value

    @cached_property
    def metadata_path(self) -> Path:
        """
        Returns:
            Path: The path to the 'metadata' directory. Created if
            it doesn't exist.
        """
        path_value = self.root / "metadata"
        path_value.mkdir(exist_ok=True, parents=True)
        logger.debug(f"Accessed 'metadata_path': created directory at {path_value}")
        return path_value

    @cached_property
    def documents_path(self) -> Path:
        """
        Returns:
            Path: The path to the 'documents' directory. Created if
            it doesn't exist.
        """
        path_value = self.root / "documents"
        path_value.mkdir(exist_ok=True, parents=True)
        logger.debug(f"Accessed 'documents_path': created directory at {path_value}")
        return path_value

    @cached_property
    def summaries_path(self) -> Path:
        """
        Returns:
            Path: The path to the 'summaries' directory. Created if
            it doesn't exist.
        """
        path_value = self.root / "summaries"
        path_value.mkdir(exist_ok=True, parents=True)
        logger.debug(f"Accessed 'documents_path': created directory at {path_value}")
        return path_value

    @cached_property
    def min_date(self) -> datetime:
        """
        Returns:
            datetime: The earliest (inclusive) date for processing
            opportunities, parsed from `conf.min_date`.
        """
        value = datetime.strptime(str(self.conf.min_date), "%Y%m%d")
        logger.debug(f"Accessed 'min_date' with value: {value}")
        return value

    @cached_property
    def max_date(self) -> datetime:
        """
        Returns:
            datetime: The latest (inclusive) date for processing
            opportunities, parsed from `conf.max_date`.
        """
        value = datetime.strptime(str(self.conf.max_date), "%Y%m%d")
        logger.debug(f"Accessed 'max_date' with value: {value}")
        return value

    @cached_property
    def metadata_filter(self) -> Dict[str, List[str]]:
        """
        Returns:
            dict: The list of allowed values keyed by metadata key.
        """
        value = self.conf.metadata_filter
        logger.debug(f"Accessed 'metadata_filter' with value {value}")
        return value

    @cached_property
    def urls(self) -> DictConfig:
        """
        Returns:
            DictConfig: The 'urls' sub-section from `conf`, containing
            endpoints for extracts, details, and file downloads.
        """
        url_data = self.conf.urls
        logger.debug(f"Accessed 'urls' property with value: {url_data}")
        return url_data

    @cached_property
    def extract_zip_target(self) -> str:
        """
        Returns:
            str: The complete URL for the ZIP file containing extracts,
            constructed from `urls.extracts` and `extracts_filename`.
        """
        target = f"{self.urls.extracts}/{self.extracts_filename}.zip"
        logger.debug(f"Accessed 'extract_zip_target' with value: {target}")
        return target

    @cached_property
    def details_url(self) -> str:
        """
        Returns:
            str: The URL endpoint for fetching opportunity details,
            derived from `urls.details`.
        """
        value = self.urls.details
        logger.debug(f"Accessed 'details_url' property with value: {value}")
        return value

    @cached_property
    def download_url(self) -> str:
        """
        Returns:
            str: The URL endpoint for downloading attachments,
            derived from `urls.download`.
        """
        value = self.urls.download
        logger.debug(f"Accessed 'download_url' property with value: {value}")
        return value

    @cached_property
    def n_jobs(self) -> int:
        """
        Returns:
            int: The number of parallel jobs to use in joblib calls. Defaults
            to the number of processors on the machine. Note that this
            specifies the number of 'threads' when making http calls, so be
            sure to respect rate limits, be decent and whatnot.
        """
        value = self.hydra_conf.n_jobs
        logger.debug(f"Accessed 'n_jobs' property with value: {value}")
        return value

    @cached_property
    def embedding_model(self) -> Any:
        return self.hydra_conf.ai.embeddings

    def initialize_vectorstore(self):
        """
        Initializes the vectorstore from the provided partial embedding_model.
        """

        name = f"grantgist-{self.embedding_model.model}"
        return self.hydra_conf.ai.vectorstore(
            collection_name=name, embedding_function=self.embedding_model
        )

    @cached_property
    def max_docs_per_embedding_call(self) -> int:
        value = self.conf.max_docs_per_embedding_call
        logger.debug(f"Accessed 'max_docs_per_embedding_call' property with value {value}")
        return value

    @cached_property
    def disqualifying_strings(self) -> List[str]:
        return self.conf.disqualifying_strings

    @cached_property
    def textsplitter(self):
        return self.hydra_conf.ai.textsplitter

    @cached_property
    def llm(self):
        return self.hydra_conf.ai.llm

    @cached_property
    def system_prompt(self) -> str:
        return self.conf.system_prompt

    @cached_property
    def human_prompts(self) -> Dict[str, str]:
        return self.conf.human_prompts


@memory.cache(ignore=["stream"])
def cached_get_request(url: str, stream: bool = True) -> requests.Response:
    """
    A GET request using a cached session via joblib Memory.

    This function is useful for large downloads that should not be
    re-downloaded unnecessarily. The `stream` parameter is ignored
    by the cache, meaning it will not affect cache hits.

    Args:
        url (str): The URL endpoint to send the GET request to.
        stream (bool, optional): Whether to stream the response.
            Defaults to True.

    Returns:
        requests.Response: The response object from the GET request.

    Raises:
        HTTPError: If the request returned an unsuccessful status code.
    """
    logger.debug(f"Entering cached_get_request with url={url}, stream={stream}")

    response = requests.get(url, stream=stream)
    logger.debug(f"Finished GET request to {url} with status code {response.status_code}")

    if response.status_code != 200:
        logger.warning(f"GET request returned status code {response.status_code} for {url}")
    response.raise_for_status()

    logger.debug("Leaving cached_get_request, returning response")
    return response


@memory.cache
def cached_post_request(url: str, **params) -> requests.Response:
    """
    A POST request using a cached session via joblib Memory.

    The request is cached based on the URL and keyword parameters.
    If the same request is made with identical parameters, the cached
    response will be returned rather than re-issuing the POST request.

    Args:
        url (str): The URL endpoint to send the POST request to.
        **params: Arbitrary keyword arguments for the POST request body
            or settings.

    Returns:
        requests.Response: The response object from the POST request.

    Raises:
        HTTPError: If the request returned an unsuccessful status code.
    """
    logger.debug(f"Entering cached_post_request with url={url} and params={params}")

    response = requests.post(url, **params)
    logger.debug(f"Finished POST request to {url} with status code {response.status_code}")

    if response.status_code != 200:
        logger.warning(f"POST request returned status code {response.status_code} for {url}")
    response.raise_for_status()

    logger.debug("Leaving cached_post_request, returning response")
    return response


@memory.cache(ignore=["data"])
def cached_xmltodict(date: str, data: str) -> Dict[str, Any]:
    """
    Parses XML data into a dictionary using `xmltodict` and caches the result.

    The 'data' parameter is excluded from caching to avoid creating
    enormous cache keys when dealing with large XML strings. Instead,
    only the 'date' parameter is factored into the cache key.

    Args:
        date (str): A date string (used to differentiate cache entries).
        data (str): The XML string to parse into a dictionary.

    Returns:
        Dict[str, Any]: The parsed XML data as a dictionary.
    """
    logger.debug(f"Entering cached_xmltodict for date={date}. Parsing XML data.")
    parsed_data = xmltodict.parse(data)
    logger.debug("Leaving cached_xmltodict, returning parsed data")
    return parsed_data


def pull_grants_gov_extract(hydra_conf: DictConfig):
    """
    Pull the grants.gov extract into a temporary directory and extract it
    into the metadata directory. The requests calls are cached to avoid
    re-downloading files unnecessarily.

    The resulting metadata directory will contain:
        - A JSON file for each opportunity (with metadata).
        - A 'hash_state.json' (if implemented) to track file changes.

    Args:
        hydra_conf (DictConfig): The Hydra configuration object that
            contains protocol configuration settings, such as root paths,
            URLs, filenames, date ranges, etc.
    """
    p = Params(hydra_conf)  # Instantiate the Params object

    with TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        file_xml = tempdir / f"{p.extracts_filename}.xml"
        file_zip = tempdir / f"{p.extracts_filename}.zip"

        # Download and extract the ZIP file
        with Timer() as timer:
            response = cached_get_request(p.extract_zip_target, stream=True)

            logger.debug(f"Writing ZIP file to {file_zip}")
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
                    logger.debug(f"Extracting {file_zip} to temporary directory {tempdir}")
                    zf.extractall(tempdir)
            except zipfile.BadZipFile as e:
                logger.error(f"Failed to extract {file_zip}: {e}")
                logger.info("Exiting pull_grants_gov_extract early due to bad ZIP file.")
                return

        logger.info(f"XML file pulled and extracted in {timer.elapsed:.02f} s")

        # Read the extracted XML
        with Timer() as timer:
            if not file_xml.exists():
                logger.error(f"Expected XML file {file_xml} not found after extraction.")
                logger.info("Exiting pull_grants_gov_extract early due to missing XML file.")
                return
            logger.debug(f"Reading XML file from {file_xml}")
            with open(file_xml, "r", encoding="utf-8") as f:
                data = f.read()
        logger.info(f"XML file read in {timer.elapsed:.02f} s")

    # Convert XML to dictionary
    logger.debug("Converting XML data to dictionary via cached_xmltodict")
    with Timer() as timer:
        data = cached_xmltodict(p.today, data)
    logger.info(f"XML file parsed to dictionary in {timer.elapsed:.02f} s")

    # The XML structure is assumed to be data["Grants"]["OpportunitySynopsisDetail_1_0"]
    data = data["Grants"]["OpportunitySynopsisDetail_1_0"]
    logger.info("Saving JSON metadata files for each valid opportunity...")

    # Save all metadata within the configured date range
    with Timer() as timer:
        count_saved = 0
        for opportunity in data:
            post_date = datetime.strptime(opportunity["PostDate"], "%m%d%Y")
            uid = opportunity["OpportunityID"]
            continue_forward = True
            for key, allowed_values in p.metadata_filter.items():
                try:
                    if opportunity[key] not in allowed_values:
                        logger.debug(
                            f"Opportunity {uid}: {key}: {opportunity[key]} not in allowed values {allowed_values} - skipping"
                        )
                        continue_forward = False
                        break
                except KeyError:
                    logger.error(
                        f"Opportunity {uid}: key {key} not found in opportunity - skipping"
                    )
                    continue_forward = False
                    break

            if p.min_date <= post_date <= p.max_date and continue_forward:
                path = p.metadata_path / f"{uid}.json"
                logger.debug(f"Saving JSON file for opportunity ID={uid} to {path}")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(opportunity, f, indent=4)
                count_saved += 1

        logger.debug(f"Number of opportunities saved: {count_saved}")

    logger.info(f"JSON files written in {timer.elapsed:.02f} s")


def _update_opportunity_details(
    path: Path, min_date: datetime, max_date: datetime, details_url: str
) -> int:
    """
    Read an opportunity JSON from `path`, fetch additional details if within
    the specified date range, and write updated JSON back to disk.

    Args:
        path (Path): The path to the JSON file that contains the basic
            opportunity data.
        min_date (datetime): The earliest date for which we update details.
        max_date (datetime): The latest date for which we update details.
        details_url (str): The endpoint for fetching opportunity details.

    Returns:
        int: Returns 1 if the JSON file was updated, otherwise 0.
    """
    logger.debug(f"Entering update_opportunity_details for file={path.name}")

    with open(path, "r", encoding="utf-8") as f:
        opportunity = json.load(f)

    post_date_str = opportunity.get("PostDate")
    if not post_date_str:
        logger.warning(f"Missing 'PostDate' in {path.name}, skipping.")
        return 0

    post_date = datetime.strptime(post_date_str, "%m%d%Y")
    if not (min_date <= post_date <= max_date):
        logger.debug(f"{path.name} post_date={post_date} not in range, skipping update.")
        return 0

    logger.debug(f"Requesting additional details for OpportunityID={opportunity['OpportunityID']}")
    params = {"data": {"oppId": opportunity["OpportunityID"]}}

    response = cached_post_request(details_url, **params)
    if response is None:
        logger.warning(
            f"Received None response for details request to {details_url}, skipping update."
        )
        return 0

    # Update the in-memory object
    logger.debug(f"Updating in-memory object for {path.name} with new details.")
    opportunity["@details"] = response.json()

    # Write JSON back to disk
    logger.debug(f"Writing updated opportunity data back to {path}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(opportunity, f, indent=4)

    logger.debug(f"Leaving update_opportunity_details for file={path.name} (updated).")
    return 1


def pull_details(hydra_conf: DictConfig):
    """
    Pulls additional details for each opportunity within the configured
    date range and updates the corresponding JSON files in the metadata
    directory.

    Args:
        hydra_conf (DictConfig): The Hydra configuration object that contains
            protocol configuration settings, such as root paths, URLs,
            filenames, date ranges, etc.
    """
    p = Params(hydra_conf)

    all_paths = list(p.metadata_path.glob("*.json"))
    logger.debug(f"Found {len(all_paths)} JSON files to process for detail updates.")

    # Parallel processing of all JSON files
    results = Parallel(n_jobs=p.n_jobs)(
        delayed(_update_opportunity_details)(path, p.min_date, p.max_date, p.details_url)
        for path in tqdm(all_paths, disable=not get_verbosity())
    )

    # Sum up how many were successfully updated
    count_updated = sum(results)  # type: ignore

    logger.info(f"Number of opportunities updated with details: {count_updated}")
    logger.info("Finished pull_details procedure")


def _pull_all_attachments(
    load_path: Path,
    target_directory: Path,
    min_date: datetime,
    max_date: datetime,
    download_url: str,
) -> int:
    """
    Read opportunity JSON from `load_path`, check if it is within the specified
    date range, and then download relevant attachments (in this case, PDFs)
    to `target_directory`.

    For each attachment, a folder is created based on the OpportunityID and
    attachment ID, and the file is saved there.

    Args:
        load_path (Path): The path to the JSON file containing the opportunity data.
        target_directory (Path): The path where attachments should be saved.
        min_date (datetime): The earliest date to process attachments.
        max_date (datetime): The latest date to process attachments.
        download_url (str): The base URL used to download attachments.

    Returns:
        int: The number of attachments successfully downloaded for this file.
    """
    logger.debug(f"Entering _pull_all_attachments for file={load_path.name}")
    with open(load_path, "r", encoding="utf-8") as f:
        opportunity = json.load(f)

    post_date_str = opportunity.get("PostDate")
    if not post_date_str:
        logger.warning(f"Missing 'PostDate' in {load_path.name}, skipping.")
        return 0

    post_date = datetime.strptime(post_date_str, "%m%d%Y")
    if not (min_date <= post_date <= max_date):
        logger.debug(
            f"{load_path.name} post_date={post_date} not in range, skipping attachment download."
        )
        return 0

    counter = 0
    opportunity_id = str(opportunity["OpportunityID"])
    logger.debug(f"Pulling attachments for OpportunityID={opportunity_id}")
    folders = opportunity["@details"]["synopsisAttachmentFolders"]

    for folder in folders:
        for attachment in folder["synopsisAttachments"]:
            attachment_id = str(attachment["id"])
            attachment_type = attachment["mimeType"]
            attachment_name = attachment["fileName"]

            # We're specifically pulling PDFs
            is_pdf_type = attachment_type == "application/pdf"
            has_pdf_ext = ".pdf" in attachment_name.lower()
            if is_pdf_type and has_pdf_ext:
                url = f"{download_url}/{attachment_id}"
                logger.debug(f"Downloading attachment {attachment_id} from {url}")
                response = cached_get_request(url, stream=True)
                if response is None:
                    logger.warning(f"Received None response for {url}, skipping this attachment.")
                    continue

                filename = sanitize_filename(attachment_name).replace(" ", "_")
                file_dir = target_directory / opportunity_id / attachment_id
                file_dir.mkdir(exist_ok=True, parents=True)
                file_target = file_dir / filename

                logger.debug(f"Writing attachment to {file_target}")
                with open(file_target, "wb") as f:
                    f.write(response.content)

                counter += 1

    logger.debug(
        f"Leaving _pull_all_attachments for file={load_path.name}, downloaded={counter} attachments."
    )
    return counter


def pull_pdfs(hydra_conf: DictConfig):
    """
    Pulls PDF attachments for each opportunity within the configured date range
    and saves them in a structured directory under 'documents/'.

    Args:
        hydra_conf (DictConfig): The Hydra configuration object that contains
            protocol configuration settings, such as root paths, URLs,
            filenames, date ranges, etc.
    """
    p = Params(hydra_conf)

    all_paths = list(p.metadata_path.glob("*.json"))
    logger.debug(f"Preparing to pull attachments from {len(all_paths)} opportunity files.")

    results = Parallel(n_jobs=p.n_jobs)(
        delayed(_pull_all_attachments)(
            path, p.documents_path, p.min_date, p.max_date, p.download_url
        )
        for path in tqdm(all_paths, disable=not get_verbosity())
    )

    # Sum up how many attachments were successfully downloaded
    count_updated = sum(results)  # type: ignore

    logger.info(f"Number of attachments pulled: {count_updated}")


def _get_unique_identifier(metadata: Dict[str, str]) -> str:
    source_api = metadata["source_api"]
    opportunity_id = metadata["OpportunityId"]
    document_id = metadata["DocumentId"]
    return f"{source_api}-{opportunity_id}-{document_id}"


def _write_vectorstore(vectorstore: VectorStore, p: Params):
    docs = []
    disqualifying_strings = p.disqualifying_strings
    metadatas = vectorstore.get()["metadatas"]  # type: ignore
    L = len(metadatas)
    logger.info(f"Processing data to vectorstore (found {L} in vectorstore already)")
    existing_unique_ids = [_get_unique_identifier(xx) for xx in metadatas]
    for d in p.documents_path.iterdir():
        opportunity_id = d.stem
        for document_path_directory in d.iterdir():
            document_path = list(document_path_directory.iterdir())
            if len(document_path) != 1:
                logger.error(f"Document path {document_path} should have only a single file")
            document_path = document_path[0]
            document_id = document_path_directory.stem
            loader = PyPDFLoader(str(document_path))

            # loaded_docs represents the pages of the pdf
            loaded_docs = loader.load()

            # For each page, we need to determine if the entire document
            # is valid
            tmp_docs = []
            doc_valid = True
            for ii, page in enumerate(loaded_docs):
                if any([s in page.page_content for s in disqualifying_strings]):
                    logger.warning(f"Document {document_path} skipped due to disqualifying string")
                    doc_valid = False
                    break
                page.metadata["source_path"] = str(document_path)
                page.metadata["source_api"] = "grants.gov"
                page.metadata["OpportunityId"] = str(opportunity_id)
                page.metadata["DocumentId"] = str(document_id)
                page.metadata["page"] = ii
                current_id = _get_unique_identifier(page.metadata)
                if current_id in existing_unique_ids:
                    logger.debug(f"id {current_id} skipped, already exists in the database")
                    doc_valid = False
                    break
                tmp_docs.append(page)

            if doc_valid:
                logger.debug(f"Document {document_path} added to docs list")
                docs.extend(tmp_docs)

    if p.textsplitter is not None:
        docs = p.textsplitter.split_documents(docs)

    logger.info(f"Processing {len(docs)} instances to the vectorstore")

    for ii in range(0, len(docs), p.max_docs_per_embedding_call):
        chunk = docs[ii : ii + p.max_docs_per_embedding_call]
        _ = vectorstore.add_documents(documents=chunk)


def construct_vectorstore(hydra_conf: DictConfig):
    """
    Creates the vectorstore.
    """

    p = Params(hydra_conf)
    vectorstore = p.initialize_vectorstore()
    _write_vectorstore(vectorstore, p)


def _is_safe(metadata, pre):
    _errors = []
    _warnings = []
    for key, value in metadata.items():
        try:
            severity = value["severity"]
            if severity == "safe":
                pass
            elif severity == "low":
                logger.warning(f"{pre} Safety check low severity: {key}: {value}")
                _warnings.append(f"{pre} {key}: {severity}")
            else:
                logger.error(f"{pre} Safety check medium or high severity: {key}: {value}")
                _errors.append(f"{pre} {key}: {severity}")
        except KeyError:
            pass
        try:
            detected = value["detected"]
            if detected:
                msg = f"Jailbreak detected for {pre} {key}: {value}"
                logger.critical(msg)
                raise RuntimeError
        except KeyError:
            pass
    return _warnings, _errors


def is_safe(metadata):
    _errors = []
    _warnings = []
    for d in metadata["prompt_filter_results"]:
        _w, _e = _is_safe(d["content_filter_results"], "Prompt")
        _errors.extend(_e)
        _warnings.extend(_w)
    _w, _e = _is_safe(metadata["content_filter_results"], "Content")
    _errors.extend(_e)
    _warnings.extend(_w)
    return len(_errors) == 0 and len(_warnings) == 0, _warnings, _errors


def construct_safety_message(responses):
    if responses is None:
        return "⚠️  Model metadata not available, safety check skipped"
    safe_message = "✅ This summary has passed Microsoft Azure guardrails and is designated as 'safe' or 'low severity'."
    _errors = []
    _warnings = []
    for name, prompt, ai_content, ai_metadata in responses:
        safe, _w, _e = is_safe(ai_metadata)
        if safe:
            continue
        _w = [f"{name}: {ww}" for ww in _w]
        _e = [f"{name}: {ee}" for ee in _e]
        _warnings.extend(_w)
        _errors.extend(_e)

    if len(_errors) == 0 and len(_warnings) == 0:
        return safe_message

    e_msg = ""
    if len(_errors) > 0:
        msg = [f"> ⛔️ {m}" for m in _errors]
        msg = "\n>\n".join(msg)
        e_msg = f"> ⛔️ Safety checks failed, assume responses are unsafe!\n>\n{msg}"
    w_msg = ""
    if len(_warnings) > 0:
        msg = [f"> ⚠️  {m}" for m in _warnings]
        msg = "\n>\n".join(msg)
        w_msg = f"> ⚠️  Be aware: some safety checks indicated 'low' severity\n>\n{msg}"

    return f"{e_msg}\n>\n{w_msg}"


def _summarize_grant(metadata_file: Path, p: Params):
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    title = metadata["OpportunityTitle"]
    foa_number = metadata["OpportunityNumber"]
    agency = metadata["AgencyName"]
    dt_postdate = datetime.strptime(metadata["PostDate"], "%m%d%Y")
    postdate = dt_postdate.strftime("%d %B %Y")

    # Execute summaries with the LLM to a variety of prompts
    vectorstore = p.initialize_vectorstore()
    opportunity_id = metadata["OpportunityID"]
    r_kwargs = p.conf.retriever_kwargs

    search_kwargs = {"k": r_kwargs["k"], "filter": {"OpportunityId": opportunity_id}}

    @tool(response_format="content_and_artifact")
    def retrieve(query: str) -> Tuple[str, Document]:
        """Retrieve information related to a general query."""

        _docs = vectorstore.similarity_search(query, **search_kwargs)
        reordering = LongContextReorder()
        _docs = reordering.transform_documents(_docs)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}") for doc in _docs
        )
        return serialized, _docs  # type: ignore

    tools = [retrieve]
    tool_node = ToolNode(tools)
    model_with_tools = p.llm.bind_tools(tools)

    def should_continue(state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:  # type: ignore
            return "tools"
        return END

    def call_model(state: MessagesState):
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")

    app = workflow.compile()

    responses = []
    for name, prompt in p.human_prompts.items():
        for chunk in app.stream(
            {"messages": [("system", p.system_prompt), ("human", prompt)]}, stream_mode="values"
        ):
            logger.debug(chunk["messages"][-1])
        chunk0 = chunk["messages"][-1]  # type: ignore
        ai_content = chunk0.content
        try:
            ai_metadata = chunk0.response_metadata
        except:
            ai_metadata = None
            # TODO: add option to override this
            logger.error("response_metadata not found, cannot verify safety")

        responses.append((name, prompt, ai_content, ai_metadata))

    # Parse through responses
    formatted_responses = []
    formatted_prompts = []
    for name, prompt, content, ai_metadata in responses:
        formatted_responses.append(f"#### ⚙️  {name}\n{content.strip()}")
        formatted_prompts.append(f"- *{name}*: {prompt.strip()}")

    formatted_responses = "\n\n".join(formatted_responses)
    formatted_prompts = "\n".join(formatted_prompts)

    safety_message = construct_safety_message(responses)

    summary = f"""
> `composer.grantgist` [grants.gov](https://grants.gov/search-grants) digest 
>
> 🚨 Attention! This summary is AI-generated. There can be errors. Always read the
full funding opportunity before responding to a call. This digest is only
intended as exactly that: a short summary.
>
{safety_message}
> ℹ️  Please direct any questions, comments or concerns to [mcarbone@bnl.gov](mailto:mcarbone@bnl.gov),
or [open an issue on GitHub](https://github.com/matthewcarbone/Composer/issues).
>
> 🚀 Free and open source. Contributions welcome!
[github.com/matthewcarbone/Composer](https://github.com/matthewcarbone/Composer)
</small>

## {title}

**NOFO/FOA#**: {foa_number}

**Issuing Agency**: {agency}

**Post Date**: {postdate}\n\n

{formatted_responses}


### Prompt Information\n

{formatted_prompts}
    """

    # Write the raw Markdown file
    with open(p.summaries_path / f"{opportunity_id}.md", "w") as f:
        f.write(summary)


def summarize_grants(hydra_conf: DictConfig):
    """For each opportunity, uses RAG to summarize."""

    p = Params(hydra_conf)

    for metadata_file in p.metadata_path.glob("*.json"):
        # # testing
        # # use only the early career award
        if "358302.json" not in str(metadata_file):
            continue

        _summarize_grant(metadata_file, p)
