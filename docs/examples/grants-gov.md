# Grants.gov XML extract summarizer

Currently, the [grants.gov](https://www.grants.gov) website offers numerous ways to query information about active and past grants. However, these are often difficult to parse through, and it's very easy to miss important grants in large email dumps, or especially if manually scrolling through them.

An alternative I've stumbled across: The [grants.gov XML extract](https://www.grants.gov/xml-extract) is posted daily around 4 am ET, and can be manually curled and unziped using a basic bash command. For instance, the below pulls the full index as of ~4 am ET on 29 December 2024.
```bash
curl https://prod-grants-gov-chatbot.s3.amazonaws.com/extracts/GrantsDBExtract20241229v2.zip -o grants.zip && unzip grants.zip
```

This resulting XML file can then be easily parsed using Python into individual JSON files, each corresponding to a single solicitation/FOA/NOFO. From there, these JSON files can be indexed using a [vector store](https://python.langchain.com/docs/integrations/retrievers/self_query/chroma_self_query/) for retrieval augmented generation (RAG) via a downstream large-language model.

In this example, I've written some simple code to pull the latest information from grants.gov, create a vector store, then use that vector store for simple RAG:

First, pull the latest XML file, and create all JSON files and the Chroma vector store. Then, query.
```bash
uv run composer protocol=grantgist-sync
uv run composer protocol=grantgist-index +ai=openai
uv run composer protocol=grantgist-summarize +ai=openai +ai.prompt="..."
```

> [!CAUTION]
> Using OpenAI as a backend can potentially be expensive. Creating the vector store for ~800 grants cost roughly $0.08. Obviously that alone is not very expensive, but it can add up quickly. Use at your own risk!


## Full example

Create JSON store for all grants from January 1, 2024 until the present.

```bash
uv run composer protocol=grantgist-sync protocol.config.min_date=20240101
```

Create the vector store using your model of choice (defaults to `text-embedding-3-small`, details [here](https://openai.com/api/pricing/), when using OpenAI).

```bash
uv run composer protocol=grantgist-index +ai=openai
```

Query the model.

```bash
uv run composer protocol=grantgist-summarize +ai=openai +ai.prompt="Tell me about RENEW grants in the database. RENEW stands for Reaching a New Energy Sciences Workforce."
```

Only including the model response here, ignoring all the logs and whatnot.

```
The RENEW grants, which stand for **Reaching a New Energy Sciences Workforce**, are aimed at building foundations for research through traineeships at academic institutions that have historically been underrepresented in the Department of Energy's (DOE) Office of Science (SC) portfolio. Below are key highlights regarding the RENEW grants:

### Key Details of the RENEW Grants
- **Opportunity Title**: FY 2024 Reaching a New Energy Sciences Workforce (RENEW)
- **Opportunity Number**: DE-FOA-0003280
- **Agency**: Office of Science, U.S. Department of Energy
- **Funding Instrument**: Grant
- **Estimated Total Program Funding**: $50,000,000
- **Expected Number of Awards**: Approximately 30
- **Award Ceiling**: $2,250,000
- **Award Floor**: $100,000
- **Close Date for Applications**: July 23, 2024
- **Cost Sharing or Matching Requirement**: No

### Purpose
The RENEW grant program aims to support traineeships for students and postdoctoral researchers from non-R1 Emerging Research Institutions (ERIs) and non-R1 Minority Serving Institutions (MSIs) in areas relevant to SC programs. The program leverages unique national laboratories, user facilities, and other research infrastructure to provide hands-on training opportunities.

### Traineeships
- Traineeships are designed to provide meaningful research experiences and professional development opportunities for participants.
- Undergraduates must engage in hands-on research to understand the research process, while graduate students and postdoctoral researchers should be involved in more advanced research activities.
- The program encourages involvement in various complementary activities to foster a sense of belonging and reinforce STEM identity.

### Eligibility
- Applications must be led by either a non-R1 ERI or a non-R1 MSI.
- Domestic entities can be proposed as team members, either as subrecipients or via a collaborative application process.

### Additional Information
- **Contact Email**: sc.renew@science.doe.gov
- For more details, you can visit the [RENEW Initiative website](https://science.osti.gov/Initiatives/RENEW).

This grant represents a significant opportunity for institutions looking to enhance their research capabilities and support underrepresented groups in energy sciences.
```
