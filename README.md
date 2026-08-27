# Medical Claim Denial Analyzer

An NLP + agent pipeline that reads an unstructured medical claim denial
letter and turns it into something actionable: structured claim data,
a predicted denial category, and a drafted appeal letter or internal
fix recommendation

Built to explore the same core problem healthcare revenue-cycle AI
products deal with: denial letters arrive as messy free text, and
someone (or something) has to read, categorize, and act on hundreds of
them

## How it works

```
denial letter (raw text)
        |
        v
  [extractor.py]   -- regex/pattern-based NLP entity extraction
        |              claim number, CPT codes, ICD-10 codes, dates, amounts
        v
  [classifier.py]  -- TF-IDF + Logistic Regression text classification
        |              predicts: medical_necessity / missing_authorization /
        |              coding_error / timely_filing / eligibility_issue /
        |              duplicate_claim
        v
  [agent.py]       -- LangGraph agent
        |              routes to one of two branches based on category:
        |                - appealable denials  -> drafts an appeal letter
        |                - fixable denials      -> recommends the correction
        v
     output
```

## Why this design

- **Extraction and classification run fully offline**, no API key
  needed this is the "classic NLP" half of the project: pattern
  matching for structured fields, and a trained text classifier for
  the denial category.
- **The agent step is a real LangGraph graph** (`StateGraph`, a
  conditional edge, two terminal branches) using Gemini via
  `langchain-google-genai`, requiring `GOOGLE_API_KEY`. This is the
  part that turns a static NLP pipeline into something that actually
  acts on its own output.

## Setup

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY=your_key_here  # only needed for the agent step
```
**If you want to add your own api key make sure to change model in `agent.py`** 

## Usage

```bash
python main.py                     # runs on a built-in sample letter
python main.py --no-agent          # extraction + classification only, no API key needed
python main.py --file letter.txt   # analyze your own denial letter
```

## Training the classifier

The classifier trains on `data/denial_letters.json`, a small synthetic
dataset (18 labeled examples across 6 categories) written to resemble
real payer denial language. On a held-out split it reaches 100%
accuracy — expected given the dataset's small size and clean category
separation, not a claim about real-world performance. On messier,
larger real-world denial text, the natural next step would be
swapping the TF-IDF vectorizer for sentence embeddings and training
on a much larger labeled set

Retrain manually:

```bash
python -m src.classifier
```
## Testing

Extraction and classification are pure, offline functions, so they're
fully covered by unit tests no API key or network access needed

```bash
python -m pytest tests/ -v
```

The agent step isn't unit tested here since it depends on a live LLM
call; the tests focus on the deterministic core of the pipeline

## Batch
Added the ability to process a whole folder of denial letters in one
run instead of one at a time. Each `.txt` file gets pushed through the
full extraction + classification pipeline, and results are saved to a
single CSV (`file`, `claim_number`, `category`, `confidence`,
`agent_output`) for easy review in Excel or Google Sheets
Tested with a batch of 4 sample letters from the training set (all
predicted correctly), and separately with 4 handwritten letters using
different phrasing the model had never seen 3 out of 4 predicted
correctly. See [Known limitations](#known-limitations) for why the
fourth one was misclassified

## Project structure

```
medical-claim-analyzer/
├── data/denial_letters.json   synthetic training data
├── model/                     saved trained classifier (generated)
├── src/
│   ├── batch.py                batch mode: processes a folder of letters into one CSV
│   ├── extractor.py           regex-based NLP entity extraction
│   ├── classifier.py          TF-IDF + Logistic Regression classifier
│   ├── agent.py                LangGraph agent
│   └── pipeline.py            ties the three stages together
├── main.py                    CLI entrypoint
└── requirements.txt
```

## Possible extensions

- Swap TF-IDF for transformer embeddings for better generalization
- Add a real appeal-success feedback loop (agent learns which drafted
  appeals actually got approved)
- Extract payer name and provider NPI as additional structured fields



## Known limitations
- **Classifier accuracy is bounded by dataset size.** With only 18
  labeled examples across 6 categories (~3 per category), the TF-IDF +
  Logistic Regression classifier has limited signal to distinguish
  categories that share vocabulary. For example, a `coding_error`
  denial that mentions "does not support medical necessity" as part
  of explaining a CPT/ICD-10 mismatch can get misclassified as
  `medical_necessity`, since the model has too few examples to learn
  that the phrase alone doesn't determine the category expanding the
  dataset prioritizing varied phrasing per category over raw
  volume is the direct fix