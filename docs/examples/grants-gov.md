# Grants.gov XML extract summarizer

Currently, the [grants.gov](https://www.grants.gov) website offers numerous ways to query information about active and past grants. However, these are often difficult to parse through, and it's very easy to miss important grants in large email dumps, or especially if manually scrolling through them.

An alternative I've stumbled across: The [grants.gov XML extract](https://www.grants.gov/xml-extract) is posted daily around 4 am ET, and can be manually curled and unziped using a basic bash command. For instance, the below pulls the full index as of ~4 am ET on 29 December 2024.
```bash
curl https://prod-grants-gov-chatbot.s3.amazonaws.com/extracts/GrantsDBExtract20241229v2.zip -o grants.zip && unzip grants.zip
```

This resulting XML file can then be easily parsed using Python into individual JSON files, each corresponding to a single solicitation/FOA/NOFO. From there, these JSON files can be indexed using a [vector store](https://python.langchain.com/docs/integrations/retrievers/self_query/chroma_self_query/) for retrieval augmented generation (RAG) via a downstream large-language model.

In this example, I've written some simple code to pull the latest information from grants.gov, create a vector store, then use that vector store for simple RAG:

First, pull the latest XML file, and create all JSON files.
```bash
uv run composer protocol=grantgist-sync
```

Then, query.
> [!CAUTION]
> Using OpenAI as a backend can potentially be expensive. Creating the vector store for ~800 grants cost roughly $0.08. Obviously that alone is not very expensive, but it can add up quickly. Use at your own risk!

> [!NOTE]
> This is a constant work in progress, and I currently have hardcoded the query just for the sake of demonstration.

```bash
uv run composer protocol=grantgist-summarize +ai=openai
```
