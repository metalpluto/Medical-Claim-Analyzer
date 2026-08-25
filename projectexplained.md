# Medical Claim Denial Analyzer — File by File

A thorough explaination of this code

---

## `data/denial_letters.json`

**What it is:** Your training data 18 fake but realistic-sounding denial letters, each labeled with which of 6 categories it belongs to (`medical_necessity`, `missing_authorization`, `coding_error`, `timely_filing`, `eligibility_issue`, `duplicate_claim`)

**Why it exists:** Machine learning models don't know anything until you show them examples. This file is the "textbook" the classifier studies to learn what each denial category sounds like in real text

**Format:** A JSON array of objects, each with a `text` field (the letter) and a `category` field (the correct answer). Nothing fancy just structured examples

---

## `src/extractor.py`

**What it is:** Pulls specific pieces of information out of messy text using **regex** (regular expressions patterns that describe what a piece of text should look like)

**How it works, piece by piece:**

- `CLAIM_NUMBER_RE` looks for the pattern "CLM-" followed by 4-6 digits, like `CLM-84421`.
- `CPT_CODE_RE` looks for the word "CPT" followed by exactly 5 digits, like `CPT 29881`. CPT codes are real medical billing codes for procedures.
- `ICD10_CODE_RE` looks for diagnosis codes like `M54.5` (a letter, two digits, optionally a dot and more digits/letters).
- `DOLLAR_RE` looks for a dollar sign followed by numbers, like `$412.00`
- `DATE_RE` looks for dates in `MM/DD/YYYY` style formatting

Each of these is a **compiled pattern** (`re.compile(...)`) — Python pre-processes the pattern once so it can search text quickly and repeatedly

The `extract()` function runs all five patterns against a piece of text and packages the results into an `ExtractedClaim` object (a small structured container, defined using `@dataclass` — a Python shortcut for "a class that's just a bundle of fields, nothing fancy")

**The `if __name__ == "__main__":` block at the bottom** is a common Python pattern — it only runs when you execute this file *directly* (`python extractor.py`), not when another file imports it. It's there so you can test the extractor on its own with one sample letter.

---

## `src/classifier.py`

**What it is:** A real, trained machine learning model that reads a denial letter's text and predicts which of the 6 categories it belongs to.

**The core idea — TF-IDF + Logistic Regression:**

1. **TF-IDF** (`TfidfVectorizer`) turns each letter into a list of numbers. It looks at every word (and pairs of words, thanks to `ngram_range=(1,2)`) and scores how *important* each one seems common words like "the" get low scores, distinctive words like "duplicate" or "unauthorized" get higher scores. This is how you turn text into math a model can actually work with
2. **Logistic Regression** (`LogisticRegression`) is the actual predictor it looks at those number patterns and learns which patterns tend to go with which category label

**`build_pipeline()`** chains these two steps together into one object (`Pipeline`), so you can feed it raw text and get a prediction out the other end without manually doing each step

**`train()`** does the real work:
- Loads the dataset from the JSON file
- Splits it into a *training set* (70%) and a *test set* (30%) using `train_test_split` this is standard ML practice: train on some examples, then check accuracy on examples the model has never seen, to get an honest sense of how well it actually learned
- Trains the pipeline, then prints how accurate it was on that held-out test set
- Re-trains on *all* the data (no split) to make the final saved model as strong as possible
- Saves the trained model to a file (`model/denial_classifier.joblib`) using `joblib`, so you don't have to retrain it every single time you run the program — training happens once, then the saved file gets reused instantly

**`load_model()`** checks if a saved model file already exists — loads it if so, trains a fresh one if not

**`predict()`** is what actually gets used elsewhere in the project — feed it a string of text, it returns the predicted category plus a confidence score (how sure the model is, from 0 to 1)

---

## `src/agent.py`

**What it is:** The "smart" part of the project — a LangGraph agent that takes the structured data from the extractor and classifier and decides what to actually *do* about the denial, draft an appeal letter, or recommend an internal fix

**Key concepts:**

- **`ClaimState`** `TypedDict` defining exactly what pieces of data flow through the agent (the text, extracted fields, category, confidence, chosen action, and final output). Think of it as a shared clipboard that gets passed from step to step and updated along the way.
- **`APPEALABLE` / `NEEDS_FIX`** — two sets of category names. This is the business logic: some denial types are worth appealing (like medical necessity disputes), others just need to be fixed and resubmitted (like a coding error).
- **`llm = ChatGoogleGenerativeAI(...)`** — sets up the connection to Google's Gemini model, so the agent can actually generate written text (an appeal letter or a recommendation), not just move data around

**The graph itself this is the LangGraph part:**

- **Nodes** are individual steps. Each one is a plain Python function that takes the state, does something, and returns the updated state:
  - `analyze_node` — currently a placeholder/hook for future analysis steps
  - `decide_action_node` — checks which category the claim falls into and decides "appeal" or "fix"
  - `draft_appeal_node` — builds a prompt describing the claim, sends it to Gemini, and saves the drafted appeal letter into the state
  - `recommend_fix_node` — same idea, but asks Gemini for a fix recommendation instead
- **`route_action()`** — a function that looks at the decided action and tells the graph which node to go to next. This is what makes it a *branching* agent instead of a straight line
- **`build_graph()`** — this is where the graph structure actually gets wired together:
  - `add_node(...)` registers each function as a step
  - `set_entry_point("analyze")` says where the graph starts
  - `add_edge(...)` connects steps that always follow each other
  - `add_conditional_edges(...)` is the branching logic after `decide_action`, go to `draft_appeal` OR `recommend_fix` depending on what `route_action()` returns
  - `.compile()` turns this definition into something you can actually run

**`run_agent()`** is the function other files call it checks you have an API key set, builds the graph, runs it with your claim's data, and returns just the final drafted text

---

## `src/pipeline.py`

**What it is:** The glue that connects everything — extraction, classification, and the agent into one simple function call

**`analyze_letter()`** does, in order:
1. Runs the extractor on the text
2. Runs the classifier on the text
3. If `use_agent=True`, also runs the agent (importing it only at this point — called a "deferred import" so the rest of the pipeline still works even if LangGraph/Gemini packages aren't installed, since you only need them for this last optional step)
4. Packages everything into one dictionary and returns it

This file exists so `main.py` doesn't have to know or care about the details of how extraction/classification/the agent each work internally — it just calls one function and gets a complete answer back. This is a common software design idea called **separation of concerns**

---

## `main.py`

**What it is:** The command line entry point the file you actually run

- **`argparse`** sets up command-line options: `--file` (analyze your own letter instead of the built-in sample) and `--no-agent` (skip the Gemini step, useful for quick offline testing)
- **`load_dotenv()`** reads your `.env` file and loads `GOOGLE_API_KEY` into the environment automatically, so you don't have to manually set it every terminal session
- **`SAMPLE_LETTER`** is a built-in example so the program works out of the box with zero setup, even before you write your own test letter
- **`main()`** ties it together: reads the input (file or sample), calls `analyze_letter()` from the pipeline, then neatly prints the extracted fields, predicted category, and (if the agent ran) the drafted output

---

## `requirements.txt`

**What it is:** A list of every external package this project needs, so anyone (including you, on a fresh computer) can install everything in one command: `pip install -r requirements.txt`. Without this file, you'd have to remember and manually install each package one by one

---

## `src/__init__.py`

**What it is:** An empty file whose only job is to tell Python "treat this folder as a package." That's what makes lines like `from src.pipeline import analyze_letter` in `main.py` work without this file, Python wouldn't recognize `src/` as something you can import from
although this is not really mandatory to add `__init.py__` it's just a stylized way to structure your project and i like it anyway

---

## `model/denial_classifier.joblib`

**What it is:** Not code this is the actual *saved, trained* classifier from `classifier.py`, stored as a binary file It gets created automatically the first time you run the classifier, and reused after that so you're not retraining the model every single time you run the program (training only takes a fraction of a second here since the dataset is tiny, but this is exactly how it'd work with a much bigger, slower-to-train real-world model too)

---

## How it all connects, start to finish

```
You run: python main.py
        |
main.py reads your input text
        |
        v
pipeline.py's analyze_letter() runs:
        |
        +--> extractor.py   pulls out claim #, CPT codes, dates, etc.
        |
        +--> classifier.py  predicts the denial category
        |
        +--> agent.py       (if not skipped) LangGraph decides:
                                appeal-worthy? -> draft an appeal via Gemini
                                needs a fix?    -> recommend the fix via Gemini
        |
        v
main.py prints everything nicely to your terminal
```
