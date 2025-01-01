import logging
from dataclasses import dataclass
from os import PathLike

from joblib import Memory
from langchain_chroma import Chroma
from omegaconf.dictconfig import DictConfig

from composer.global_state import get_memory_dir

logger = logging.getLogger(__name__)
memory = Memory(location=get_memory_dir(), verbose=1)


@memory.cache(ignore=["docs", "embeddings"])
def build_and_persist_vector_store(date, docs, hydra_conf):
    name = hydra_conf.name
    embeddings_name = hydra_conf.ai.embeddings.model
    collection_name = f"{name}_{embeddings_name}_{date}"
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=hydra_conf.ai.embeddings,
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
class VectorStore:
    hydra_conf: DictConfig
    collection_name: str
    collection_dir: PathLike
