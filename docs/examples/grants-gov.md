# Grants.gov XML extract summarizer

Currently, the [grants.gov](https://www.grants.gov) website offers numerous ways to query information about active and past grants. However, these are often difficult to parse through, and it's very easy to miss important grants in large email dumps, or especially if manually scrolling through them.

The [grants.gov XML extract](https://www.grants.gov/xml-extract) is posted daily around 4 am ET, and can be manually cURL'd and unziped using a basic scripting. For instance, the below pulls the full index as of 29 December 2024. Note this is a simple GET request.

```bash
curl https://prod-grants-gov-chatbot.s3.amazonaws.com/extracts/GrantsDBExtract20241229v2.zip -o grants.zip && unzip grants.zip
```

This resulting XML file can then be easily parsed using Python into individual JSON files, each corresponding to a single solicitation/FOA/NOFO. From there, these JSON files can be indexed using a [vector store](https://python.langchain.com/docs/integrations/retrievers/self_query/chroma_self_query/) for retrieval augmented generation (RAG) via a downstream large-language model.

The following is a workflow diagram of how `composer/grantgist` works.

![image](https://github.com/user-attachments/assets/eba3733d-e97c-4da9-96c8-fcd2db18671c)

## A few key points

Before diving into the details, I want to go over a few key pieces of the API and my motivations for choosing these tools.

- The code here can be used as an API, but it's designed as a CLI. The CLI is powered by [hydra](https://hydra.cc/), a powerful command line parser and configuration tool. It allows you to define specific Python instances using YAML configuration files or just the command line itself.
- As for running the code, I recommend using [uv](https://docs.astral.sh/uv/). If you don't know about uv, you should definitely go down that rabbit hole. No more Conda environments required!
- LLM use is abstracted. You will need an LLM back-end, accessible via Python library interfaces, to use this code. This can be e.g. a local Ollama instance, OpenAI, etc.
- [Langgraph](https://www.langchain.com/langgraph) is used to power the AI components. Although difficult to learn, Langgraph appears to be the current state of the art in terms of orchestrating chains of LLM agents, retrieval augmented generation, etc. The field is moving rapidly, so this may change, but I chose Langgraph because it provides a common abstraction for interfacing with just about _any_ LLM backend.
- Most of this code deals with pulling and wrangling data. To be respectful of external datasources _all html requests are cached_. This is generally best practice to avoid hitting external APIs too many times, which in turn helps us avoid hitting rate limits, getting IP banned, etc. It is also just generally respectful to do this. I use [joblib.memory](https://joblib.readthedocs.io/en/latest/generated/joblib.Memory.html) for on-disk caching.
- Various databases and patterns are used depending on the requirement at hand. Mostly, simple flat storage of PDF and JSON files suffices, but for fast retrieval augmented generation, I use a [ChromaDB vector store](https://www.trychroma.com/).


## Understanding the necessary APIs

Before using `uv run grantgist`, it's best to understand the structure of all the files involved and how to pull them from remote servers. The vast majority of the "data wrangling" for this project can actually be done on the command line. I use a Python API for simplicity, development speed, and convenience.

In this example, we'll work with the Early Career Research Program NOFO DE-FOA-0003450. Let's assume you have already pulled the latest extract via the `curl` command above.

### Step 1: Identifying the funding opportunity of interest

Each NOFO has a number of identifiers. The one that we're all familiar with is the "opportunity number". The opportunity number of the 2025 early career research award posting is "DE-FOA-0003450". Let's find that entry in the XML file.

```bash
rg "DE-FOA-0003450" GrantsDBExtract20250304v2.xml -B 3 > 0003450_extract.txt
```

Note that `rg` is for "Rip Grep", which is generally faster than `grep`, but the `grep` command should work just as well. In the above, we're searching the XML extract file for the opportunity number of the NOFO of interest. Then, we take the 3 lines immediately preceding that match. Finally, the output is piped to `0003450_extract.txt` for viewing later. Note that I simply chose the prior 3 lines empirically.

The file `0003450_extract.txt` should look like this:

```
<OpportunitySynopsisDetail_1_0>
	<OpportunityID>358302</OpportunityID>
	<OpportunityTitle>Early Career Research Program</OpportunityTitle>
	<OpportunityNumber>DE-FOA-0003450</OpportunityNumber>
```

We can see this is part of an XML file. The start of the NOFO is designated by `<OpportunitySynopsisDetail_1_0>`, and we see that the correct match has been found. Unfortunately, the opportunity number itself is not the "unique identifier" we need to programmatically find additional information. The number we actually need is the `OpportunityID`, which we can see as the first entry in the result: "358302". This is actually the number we'll need for the next steps. You can feel free to view the rest of the XML file if you wish, but I found it much easier to simply extract this number, then use other APIs in the next steps.

### Step 2: Retrieving the details for the opportunity of interest

The `OpportunityID` now gives us the correct unique identifier for pulling additional information. After some reverse engineering, and some help from Dakota Blair, we were able to identify the true origin of the information used to populate the XML synopsis. It turns out this endpoint is `https://apply07.grants.gov/grantsws/rest/opportunity/details`. Passing the `OpportunityID` as `oppID` (confusing, right?) allows us to pull the specific JSON data for this funding opportunity:

```bash
curl 'https://apply07.grants.gov/grantsws/rest/opportunity/details' --data-raw 'oppId=358302' | jq . > 358302.json
```

This straightforward POST request first pulls data from the "details" endpoint matching the provided opportunity ID, parses the results using `jq`, and dumps the output to file "358302.json". Check out the following "spoiler" to see exactly what the output looks like!

<details>
  <summary>
    358302.json
  </summary>
  <br>

```json
{
  "id": 358302,
  "revision": 1,
  "opportunityNumber": "DE-FOA-0003450",
  "opportunityTitle": "Early Career Research Program",
  "owningAgencyCode": "PAMS-SC",
  "listed": "L",
  "publisherUid": "laingki",
  "modifiedComments": "Please see cover page for amendment explanation",
  "flag2006": "N",
  "opportunityCategory": {
    "category": "D",
    "description": "Discretionary"
  },
  "synopsis": {
    "opportunityId": 358302,
    "version": 2,
    "agencyCode": "PAMS-SC",
    "agencyName": "Kimberlie J Laing\nGrant Analyst",
    "agencyPhone": "301-903-3026",
    "agencyAddressDesc": "SC.Early@science.doe.gov",
    "agencyDetails": {
      "code": "SC",
      "seed": "PAMS-SC",
      "agencyName": "Office of Science",
      "agencyCode": "PAMS-SC",
      "topAgencyCode": "PAMS"
    },
    "topAgencyDetails": {
      "code": "PAMS",
      "seed": "PAMS",
      "agencyName": "Department of Energy - Office of Science",
      "agencyCode": "PAMS",
      "topAgencyCode": "PAMS"
    },
    "agencyContactPhone": "301-903-3026",
    "agencyContactName": "Kimberlie J Laing\nGrant Analyst",
    "agencyContactDesc": "SC.Early@science.doe.gov",
    "agencyContactEmail": "SC.Early@science.doe.gov",
    "agencyContactEmailDesc": "SC.Early@science.doe.gov",
    "synopsisDesc": "<p>The Office of Science’s (SC) mission is to deliver scientific discoveries and major scientific tools to transform our understanding of nature and advance the energy, economic, and national security of the United States (U.S.). SC is the Nation’s largest Federal sponsor of basic research in the physical sciences and the lead Federal agency supporting fundamental scientific research for our Nation’s energy future.</p><p><br></p><p>·&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<em>Science for energy, economic and national security</em>―building a foundation of scientific and technical knowledge to spur discoveries and innovations for advancing the Department’s mission. SC supports a wide range of funding modalities from single principal investigators to large team-based activities to engage in fundamental research on energy production, conversion, storage, transmission, and use, and on our understanding of the earth systems.</p><p>·&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<em>The frontiers of science</em>—exploring nature’s mysteries from the study of fundamental subatomic particles, atoms, and molecules that are the building blocks of the materials of our universe and everything in it to the DNA, proteins, and cells that are the building blocks of life. Each of the programs in SC supports research probing the most fundamental disciplinary questions.</p><p><br></p><p><em>The 21st Century tools of science</em>—providing the nation’s researchers with 28 state-of-the-art national scientific user facilities, the most advanced tools of modern science, propelling the U.S. to the forefront of science, technology development, and deployment through innovation.</p><p>SC is an established leader of the U.S. scientific discovery and innovation enterprise. Over the decades, SC investments and accomplishments in basic research and enabling research capabilities have provided the foundations for new technologies, businesses, and industries, making significant contributions to our nation’s economy, national security, and quality of life.</p>",
    "responseDate": "Apr 22, 2025 12:00:00 AM EDT",
    "responseDateDesc": "",
    "postingDate": "Jan 17, 2025 12:00:00 AM EST",
    "archiveDate": "May 22, 2025 12:00:00 AM EDT",
    "costSharing": false,
    "estimatedFunding": "136000000",
    "estimatedFundingFormatted": "136,000,000",
    "awardCeiling": "2750000",
    "awardCeilingFormatted": "2,750,000",
    "awardFloor": "875000",
    "awardFloorFormatted": "875,000",
    "applicantEligibilityDesc": "In accordance with 2 CFR 910.126, Competition, eligibility for award is restricted to U.S. Institutions of Higher Education, DOE National Laboratories (listed at https://www.energy.gov/national-laboratories), and institutions operating SC Scientific User Facilities (listed at https://science.osti.gov/User-Facilities).This eligibility restriction is intended to create an opportunity for the most promising scientists who are (a) early in their careers, (b) in positions with sufficient permanence to support independent research efforts, and (c) for investigators not at DOE-affiliated institutions, in positions that require working with the students who will become the scientific workforce of the future.",
    "sendEmail": "Y",
    "createTimeStamp": "Feb 14, 2025 05:07:10 PM EST",
    "modComments": "Please see cover page for amendment explanation",
    "createdDate": "Jan 17, 2025 09:35:05 AM EST",
    "lastUpdatedDate": "Feb 03, 2025 09:44:28 AM EST",
    "applicantTypes": [
      {
        "id": "25",
        "description": "Others (see text field entitled \"Additional Information on Eligibility\" for clarification)"
      }
    ],
    "fundingInstruments": [
      {
        "id": "G",
        "description": "Grant"
      },
      {
        "id": "O",
        "description": "Other"
      },
      {
        "id": "PC",
        "description": "Procurement Contract"
      }
    ],
    "fundingActivityCategories": [
      {
        "id": "ST",
        "description": "Science and Technology and other Research and Development"
      }
    ],
    "responseDateStr": "2025-04-22-00-00-00",
    "postingDateStr": "2025-01-17-00-00-00",
    "archiveDateStr": "2025-05-22-00-00-00",
    "createTimeStampStr": "2025-02-14-17-07-10"
  },
  "agencyDetails": {
    "code": "SC",
    "seed": "PAMS-SC",
    "agencyName": "Office of Science",
    "agencyCode": "PAMS-SC",
    "topAgencyCode": "PAMS"
  },
  "topAgencyDetails": {
    "code": "PAMS",
    "seed": "PAMS",
    "agencyName": "Department of Energy - Office of Science",
    "agencyCode": "PAMS",
    "topAgencyCode": "PAMS"
  },
  "synopsisAttachmentFolders": [
    {
      "id": 77014,
      "opportunityId": 358302,
      "folderType": "Full Announcement",
      "folderName": "DE-FOA-0003450",
      "zipLobSize": 3717692,
      "createdDate": "Jan 17, 2025 09:35:24 AM EST",
      "lastUpdatedDate": "Feb 14, 2025 05:06:44 PM EST",
      "synopsisAttachments": [
        {
          "id": 346698,
          "opportunityId": 358302,
          "mimeType": "application/pdf",
          "fileName": "DE-FOA-0003450.000002.pdf",
          "fileDescription": "Amendment 000002",
          "fileLobSize": 1313214,
          "createdDate": "Feb 03, 2025 09:42:44 AM EST",
          "synopsisAttFolderId": 77014
        },
        {
          "id": 346434,
          "opportunityId": 358302,
          "mimeType": "application/pdf",
          "fileName": "DE-FOA-0003450.000001.pdf",
          "fileDescription": "Full Announcement",
          "fileLobSize": 1312987,
          "createdDate": "Jan 17, 2025 09:35:56 AM EST",
          "lastUpdatedDate": "Jan 24, 2025 10:46:33 AM EST",
          "synopsisAttFolderId": 77014
        },
        {
          "id": 346861,
          "opportunityId": 358302,
          "mimeType": "application/pdf",
          "fileName": "DE-FOA-0003450.000003.pdf",
          "fileDescription": "Full NOFO",
          "fileLobSize": 1313657,
          "createdDate": "Feb 14, 2025 05:06:44 PM EST",
          "synopsisAttFolderId": 77014
        }
      ]
    }
  ],
  "synopsisDocumentURLs": [],
  "synAttChangeComments": [
    {
      "id": {
        "opportunityId": 358302,
        "attType": "D",
        "createdDate": "Feb 14, 2025 05:07:10 PM EST",
        "attTypeDesc": "Related Documents",
        "commentsDate": "Feb 14, 2025"
      },
      "changeComments": "Amendment 3. See front page of NOFO."
    }
  ],
  "cfdas": [
    {
      "id": 427429,
      "opportunityId": 358302,
      "cfdaNumber": "81.049",
      "programTitle": "Office of Science Financial Assistance Program"
    }
  ],
  "opportunityHistoryDetails": [
    {
      "oppHistId": {
        "opportunityId": 358302,
        "revision": 0
      },
      "opportunityId": 358302,
      "revision": 0,
      "opportunityNumber": "DE-FOA-0003450",
      "opportunityTitle": "Early Career Research Program",
      "owningAgencyCode": "PAMS-SC",
      "publisherUid": "laingki",
      "listed": "L",
      "opportunityCategory": {
        "category": "D",
        "description": "Discretionary"
      },
      "synopsis": {
        "id": {
          "opportunityId": 358302,
          "revision": 0
        },
        "opportunityId": 358302,
        "revision": 0,
        "version": 1,
        "agencyCode": "PAMS-SC",
        "agencyAddressDesc": "SC.Early@science.doe.gov",
        "agencyDetails": {
          "code": "SC",
          "seed": "PAMS-SC",
          "agencyName": "Office of Science",
          "agencyCode": "PAMS-SC",
          "topAgencyCode": "PAMS"
        },
        "agencyContactPhone": "301-903-3026",
        "agencyContactName": "Kimberlie J Laing\nGrant Analyst",
        "agencyContactDesc": "SC.Early@science.doe.gov",
        "agencyContactEmail": "SC.Early@science.doe.gov",
        "agencyContactEmailDesc": "SC.Early@science.doe.gov",
        "synopsisDesc": "<p>The Office of Science’s (SC) mission is to deliver scientific discoveries and major scientific tools to transform our understanding of nature and advance the energy, economic, and national security of the United States (U.S.). SC is the Nation’s largest Federal sponsor of basic research in the physical sciences and the lead Federal agency supporting fundamental scientific research for our Nation’s energy future.</p><p><br></p><p>·&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<em>Science for energy, economic and national security</em>―building a foundation of scientific and technical knowledge to spur discoveries and innovations for advancing the Department’s mission. SC supports a wide range of funding modalities from single principal investigators to large team-based activities to engage in fundamental research on energy production, conversion, storage, transmission, and use, and on our understanding of the earth systems.</p><p>·&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<em>The frontiers of science</em>—exploring nature’s mysteries from the study of fundamental subatomic particles, atoms, and molecules that are the building blocks of the materials of our universe and everything in it to the DNA, proteins, and cells that are the building blocks of life. Each of the programs in SC supports research probing the most fundamental disciplinary questions.</p><p><br></p><p><em>The 21st Century tools of science</em>—providing the nation’s researchers with 28 state-of-the-art national scientific user facilities, the most advanced tools of modern science, propelling the U.S. to the forefront of science, technology development, and deployment through innovation.</p><p>SC is an established leader of the U.S. scientific discovery and innovation enterprise. Over the decades, SC investments and accomplishments in basic research and enabling research capabilities have provided the foundations for new technologies, businesses, and industries, making significant contributions to our nation’s economy, national security, and quality of life.</p>",
        "responseDate": "Apr 22, 2025 12:00:00 AM EDT",
        "responseDateDesc": "",
        "postingDate": "Jan 17, 2025 12:00:00 AM EST",
        "archiveDate": "May 22, 2025 12:00:00 AM EDT",
        "costSharing": false,
        "estimatedFunding": "136000000",
        "estimatedFundingFormatted": "136,000,000",
        "awardCeiling": "2750000",
        "awardCeilingFormatted": "2,750,000",
        "awardFloor": "875000",
        "awardFloorFormatted": "875,000",
        "applicantEligibilityDesc": "In accordance with 2 CFR 910.126, Competition, eligibility for award is restricted to U.S. \r\nInstitutions of Higher Education, DOE National Laboratories (listed at https://www.energy.gov/national-laboratories), and institutions operating SC Scientific User Facilities (listed at https://science.osti.gov/User-Facilities).\r\n\r\nThis eligibility restriction is intended to create an opportunity for the most promising scientists who are (a) early in their careers, (b) in positions with sufficient permanence to support independent research efforts, and (c) for investigators not at DOE-affiliated institutions, in positions that require working with the students who will become the scientific workforce of the future.",
        "createTimeStamp": "Feb 03, 2025 09:42:44 AM EST",
        "sendEmail": "Y",
        "actionType": "U",
        "actionDate": "Feb 03, 2025 09:44:28 AM EST",
        "createdDate": "Jan 17, 2025 09:35:05 AM EST",
        "lastUpdatedDate": "Jan 17, 2025 09:35:05 AM EST",
        "applicantTypes": [
          {
            "id": "25",
            "description": "Others (see text field entitled \"Additional Information on Eligibility\" for clarification)"
          }
        ],
        "fundingInstruments": [
          {
            "id": "G",
            "description": "Grant"
          },
          {
            "id": "O",
            "description": "Other"
          },
          {
            "id": "PC",
            "description": "Procurement Contract"
          }
        ],
        "fundingActivityCategories": [
          {
            "id": "ST",
            "description": "Science and Technology and other Research and Development"
          }
        ],
        "responseDateStr": "2025-04-22-00-00-00",
        "postingDateStr": "2025-01-17-00-00-00",
        "archiveDateStr": "2025-05-22-00-00-00",
        "createTimeStampStr": "2025-02-03-09-42-44"
      },
      "cfdas": [
        {
          "id": 427429,
          "opportunityId": 358302,
          "revision": 0,
          "cfdaNumber": "81.049",
          "programTitle": "Office of Science Financial Assistance Program"
        }
      ],
      "synopsisModifiedFields": [],
      "forecastModifiedFields": []
    }
  ],
  "opportunityPkgs": [
    {
      "id": 290195,
      "topportunityId": 358302,
      "familyId": 14,
      "dialect": "XFDL2.2",
      "opportunityNumber": "DE-FOA-0003450",
      "opportunityTitle": "Early Career Research Program",
      "cfdaNumber": "81.049",
      "openingDate": "Jan 17, 2025 12:00:00 AM EST",
      "closingDate": "Apr 22, 2025 12:00:00 AM EDT",
      "owningAgencyCode": "PAMS-SC",
      "agencyDetails": {
        "code": "SC",
        "seed": "PAMS-SC",
        "agencyName": "Office of Science",
        "agencyCode": "PAMS-SC",
        "topAgencyCode": "PAMS"
      },
      "topAgencyDetails": {
        "code": "PAMS",
        "seed": "PAMS",
        "agencyName": "Department of Energy - Office of Science",
        "agencyCode": "PAMS",
        "topAgencyCode": "PAMS"
      },
      "programTitle": "Office of Science Financial Assistance Program",
      "contactInfo": "SC.Early@science.doe.gov",
      "gracePeriod": 14,
      "competitionId": "DE-FOA-0003450",
      "competitionTitle": "Early Career Research Program",
      "electronicRequired": "Y",
      "expectedApplicationCount": 900,
      "openToApplicantType": 1,
      "listed": "L",
      "isMultiProject": "N",
      "extension": "pdf",
      "mimetype": "application/pdf",
      "lastUpdate": "Jan 24, 2025 10:47:00 AM EST",
      "workspaceCompatibleFlag": "Y",
      "packageId": "PKG00290195",
      "openingDateStr": "2025-01-17-00-00-00",
      "closingDateStr": "2025-04-22-00-00-00"
    }
  ],
  "closedOpportunityPkgs": [],
  "originalDueDate": "Apr 22, 2025 12:00:00 AM EDT",
  "originalDueDateDesc": "",
  "synopsisModifiedFields": [
    "revision",
    "version",
    "applicantEligibilityDesc",
    "createTimeStamp"
  ],
  "forecastModifiedFields": [],
  "errorMessages": [],
  "synPostDateInPast": true,
  "docType": "synopsis",
  "forecastHistCount": 0,
  "synopsisHistCount": 0,
  "assistCompatible": false,
  "assistURL": "",
  "relatedOpps": [],
  "draftMode": "N"
}
```

</details>

We now have what is, effectively, the most detailed description of the NOFO, and likely exactly what is used to populate the XML extract. Only one more step to go to find the actual NOFOs.

### Step 3: Retrieving the NOFOs programmatically

The "synopsisAttachmentFolders" key provides a list of "directories" containing various documents pertinent to the NOFO. The one we're interested in generally corresponds to a "folderType" of "Full Announcement". This is, usually, the NOFO that you download when you search for something on grants.gov!

In our case (as of the time of writing this tutorial) we are looking for filename "DE-FOA-0003450.000002.pdf", which is the second revision to the original NOFO. You'll note that this document itself has its _own_ unique id: "346698". So how do we access this?

It turns out that there is _yet another_ endpoint we needed to reverse engineer via the same process as in step 2. The right endpoint turns out to be `https://apply07.grants.gov/grantsws/rest/opportunity/download` and that we can pull the correct PDF document via

```bash
curl https://apply07.grants.gov/grantsws/rest/opportunity/att/download/346698 > 346698.pdf
```

And that's it! We now have the correct NOFO PDF for our own viewing, or in the case of using Composer, for systematically reading the data into a vector store for downstream retrieval augmented generation.

## Example in brief

```bash
uv run grantgist ai=azure-o3-mini protocol.config.min_date=20250117 protocol.config.max_date=20250117
```

That's it! The `grantgist` entrypoint will automatically run all steps.

> [!CAUTION]
> Using OpenAI as a backend can potentially be expensive. Use at your own risk!


## Full example

We can run each step sequentially as well. The options available are:

```
- "pull_grants_gov_extract"
- "pull_details"
- "pull_pdfs"
- "construct_vectorstore"
- "summarize_grants"
```

and running a single one of these commands (or multiple commands, just use list syntax, looks like this:

```bash
uv run grantgist "protocol.run=[pull_grants_gov_extract]" ai=azure
```

Composer will warn you if you forgot to provide required arguments. Check out [matthewcarbone.dev/grantgist](https://matthewcarbone.dev/grantgist) for some examples of the outputs of these models!
