Prompt 0 — Establish the baseline first

You are modifying an existing Semantic Search Engine repository. Before changing any code, perform a strict baseline inspection.

Repository components include:

Sentence-Transformers using all-MiniLM-L6-v2
FAISS IndexFlatIP with normalized embeddings
BM25 retrieval
Hybrid lexical-semantic retrieval with Min-Max score fusion
MS MARCO data
Approximately 10,000 indexed passages
Precision@K, Recall@K, and MRR@K evaluation
Existing latency benchmark scripts
Streamlit application

The goal is to make targeted fixes before putting this project on a competitive ML/SWE resume.

Your task

Do NOT modify any files yet.

First:

Recursively inspect the entire repository.
Identify the exact commands required to:
download/preprocess data;
build the FAISS index;
run BM25 retrieval;
run semantic retrieval;
run hybrid retrieval;
run evaluation;
run benchmarks;
launch the Streamlit application.
Run or inspect every existing validation, evaluation, and benchmark path where possible.
Record the current state of:
Precision@10;
Recall@10;
MRR@10;
latency;
corpus size;
embedding model;
FAISS index type.
Identify existing failures, warnings, missing dependencies, hardcoded paths, and non-reproducible commands.
Inspect the Git working tree before making changes.
Required output

Create a concise BASELINE.md containing:

repository commit/working-tree state;
environment and dependency information;
exact commands used;
corpus size;
model name and embedding dimension;
FAISS index type;
baseline retrieval metrics;
baseline latency measurements;
known failures or limitations.

Do not invent any numbers. If a measurement cannot be reproduced, explicitly write:

NOT VERIFIED

Do not refactor the project in this task. This task exists only to establish a trustworthy baseline for later changes.

At the end, provide a concise summary of what is currently reproducible and what is not.

Prompt 1 — Add proper automated tests

You are working on an existing Semantic Search Engine repository.

The repository already implements:

Sentence-Transformers dense embeddings;
FAISS IndexFlatIP;
BM25 retrieval;
hybrid lexical-semantic score fusion;
Precision@K;
Recall@K;
MRR@K.

The repository currently lacks meaningful automated tests.

Your task is to add a focused pytest test suite without performing unnecessary refactoring.

First inspect the current implementation

Before changing code:

Locate the actual implementations of:
precision_at_k;
recall_at_k;
mrr_at_k;
min_max_normalize;
semantic retrieval;
hybrid retrieval;
FAISS ID/document mapping.
Understand the exact function signatures before writing tests.
Do not invent APIs that do not exist.
Required tests
1. Precision@K tests

Test:

all top-K results relevant;
no results relevant;
some results relevant;
K = 1;
fewer retrieved results than K, according to the current intended semantics;
empty retrieved results;
empty relevant set.

Verify expected values exactly.

2. Recall@K tests

Test:

all relevant documents retrieved;
no relevant documents retrieved;
partial retrieval;
empty relevant set;
empty retrieved results.

Ensure the zero-relevant-documents behavior is explicit and documented.

3. MRR@K tests

Test:

first result relevant → MRR = 1;
second result relevant → MRR = 0.5;
relevant result at rank K;
no relevant result;
multiple relevant documents;
empty results.
4. Min-Max normalization tests

Test:

normal score ranges;
negative values;
identical scores;
a single score;
empty input.

The implementation must not silently produce NaN or divide by zero.

5. Semantic retrieval integration test

Using a tiny deterministic corpus and test fixture:

build a FAISS index;
add stable document IDs;
execute a query;
verify that exactly top_k or the maximum available valid results are returned;
verify that returned FAISS IDs map to the correct documents.

Do not download MS MARCO or a model during every unit test if mocking or a tiny fixture can avoid it.

6. Hybrid retrieval integration test

Verify:

BM25 and semantic candidate results are combined;
score normalization does not produce NaN;
fusion scores are calculated according to the actual formula;
final results are correctly ordered;
duplicate documents do not appear unexpectedly.
Testing principles
Use deterministic fixtures.
Keep tests fast.
Do not make network calls.
Do not require the full 10K corpus.
Do not weaken assertions just to make tests pass.
If a production bug is discovered, fix it and add a regression test.
Deliverables

Add:

a tests/ directory;
appropriate test files;
any minimal test configuration required;
test dependencies in the correct dependency file.

Add a clear command for running tests.

Run the complete suite and report:

total tests;
passed tests;
failed tests;
skipped tests;
warnings.

Do not claim all tests pass unless you actually ran them.

Prompt 2 — Fix retrieval correctness and edge cases

You are auditing and improving an existing Semantic Search Engine repository.

The goal is not to add flashy features. The goal is to make the existing retrieval pipeline technically correct and robust enough to defend in an ML/SWE interview.

Inspect the actual code before making modifications.

Audit the complete retrieval path

Trace:

Raw query
  ↓
Input validation
  ↓
Query embedding
  ↓
Optional normalization
  ↓
FAISS/BM25 retrieval
  ↓
Candidate IDs
  ↓
Document mapping
  ↓
Hybrid fusion where applicable
  ↓
Deduplication
  ↓
Final top-K results
Required edge cases

Fix or explicitly handle:

Queries
None;
empty string;
whitespace-only string;
extremely long query;
non-string query.

Use clear errors rather than obscure downstream exceptions.

top_k / K

Handle:

zero;
negative values;
non-integer values;
top_k > corpus size.

Do not return invalid FAISS placeholder IDs or invalid documents.

Index and data consistency

Verify:

every FAISS result ID maps to a real document;
document IDs remain stable;
missing IDs are handled safely;
duplicate IDs cannot silently corrupt retrieval;
index/document count mismatches are detected.
FAISS correctness

Verify that:

normalized vectors are used consistently with IndexFlatIP;
query vectors are normalized consistently;
similarity semantics are correctly described;
invalid -1 IDs returned when requesting more neighbors than exist are filtered.
Hybrid fusion

Verify:

no divide-by-zero or NaN from Min-Max normalization;
empty candidate sets work;
score ranges are handled safely;
duplicate documents are merged consistently;
sorting is deterministic;
the fusion formula matches the implementation.
Constraints
Do not redesign the architecture.
Do not add ANN indexing.
Do not change the retrieval model.
Do not alter benchmark/evaluation results unless fixing a real correctness bug.
Preserve backward compatibility where practical.
Testing requirement

For every real bug fixed:

Add or update an automated regression test.
Run the affected tests.
Run the complete test suite.
Required final report

Provide:

Issue	Severity	Root Cause	Fix	Regression Test Added

Then list any remaining edge cases that were intentionally left unchanged and explain why.

Prompt 3 — Make configuration and execution reproducible

You are improving an existing Semantic Search Engine repository before it is presented on a technical resume.

The repository currently has hardcoded paths and parameters.

Your task is to improve reproducibility and configuration without overengineering.

Inspect before modifying

Identify all hardcoded values for:

dataset paths;
processed-data paths;
index paths;
metadata/document paths;
embedding model name;
batch size;
top_k;
hybrid fusion alpha;
benchmark query count;
benchmark warmup runs;
benchmark repetitions;
device selection.

Do not blindly convert every constant into configuration. Only externalize values that a user or experimenter should reasonably change.

Required changes

Create a consistent configuration approach using one of:

argparse plus centralized defaults; or
a lightweight configuration module.

Prefer the simplest approach that fits the existing repository.

The following workflows must be runnable from documented commands:

1. Download/sample data
2. Preprocess data
3. Build embeddings and index
4. Run semantic search
5. Run BM25 search
6. Run hybrid search
7. Run evaluation
8. Run benchmark
9. Run tests
10. Launch Streamlit

Commands must not require users to edit source code.

Path handling
Replace fragile relative paths where necessary.
Use pathlib.
Create output directories when appropriate.
Fail with actionable messages when required files are missing.
Do not silently create empty indexes or use fallback data.
Reproducibility

Document:

Python version tested;
dependency installation command;
model name;
dataset version/source;
corpus size;
index type;
embedding normalization behavior.

If randomness exists, add seed handling where meaningful.

Deliverables

Update:

CLI/configuration implementation;
dependency files if necessary;
README with exact commands.

Add a short section:

Reproducing Results

A fresh user should be able to clone the repository and understand the intended execution order without reading source code.

Do not claim reproducibility unless you actually execute the documented commands in a clean or equivalent environment.

Prompt 4 — Strengthen evaluation rigor and evidence

You are improving the evaluation rigor of an existing Semantic Search Engine repository.

The project currently uses approximately:

10,000 MS MARCO passages;
50 evaluation queries;
ground-truth relevant document IDs;
Precision@10;
Recall@10;
MRR@10;
BM25, semantic, and hybrid retrieval.

Your goal is to ensure that every reported metric is reproducible, correctly calculated, and fairly compared.

First inspect the existing evaluation

Determine:

How the 50 queries were selected.
Where the relevance labels come from.
Whether query IDs and document IDs are stable.
Whether the evaluation queries overlap incorrectly with indexing data.
Whether the same candidate corpus is used for every retriever.
Whether all systems use the same K.
Whether BM25, semantic, and hybrid methods are evaluated on identical queries and qrels.

Do not assume the current evaluation design is valid.

Required improvements
Evaluation validation

Add checks for:

duplicate query IDs;
duplicate document IDs;
relevance labels referencing missing documents;
queries with empty relevance sets;
missing or malformed evaluation fields.
Method comparison

Ensure one command can evaluate:

BM25;
semantic retrieval;
hybrid retrieval.

The report should clearly show each method side by side.

At minimum report:

Precision@10;
Recall@10;
MRR@10.

Do not add metrics merely for appearance.

Reproducible artifacts

Make evaluation produce a machine-readable result such as JSON or CSV containing:

timestamp;
model;
corpus size;
query count;
index type;
retrieval method;
relevant configuration;
metrics.

A human-readable Markdown report may also be generated from this data.

Do not overwrite prior results silently.

Provenance

Every result must be traceable to:

dataset;
corpus size;
query set;
model;
retrieval method;
configuration.
Required comparison table

Produce:

Method	P@10	R@10	MRR@10	Query Count

Only populate values that were actually measured during the task.

Critical rule

Do not manipulate the evaluation dataset or cherry-pick queries to make semantic or hybrid retrieval look better.

If the current 50-query set is too small to support strong generalization claims, explicitly document that limitation rather than hiding it.

Final report

State exactly which resume claims are now supported by the evaluation evidence and which remain qualified.

Prompt 5 — Make latency benchmarking credible

You are improving the benchmark methodology of an existing Semantic Search Engine repository.

The repository currently contains latency measurements for semantic and BM25 retrieval over approximately 10,000 documents.

The project may claim low-latency retrieval at this corpus size, but must not claim large-scale retrieval.

Your task is to make the existing benchmark credible and reproducible.

Do not introduce artificial performance optimizations just to improve numbers.

First inspect current benchmarking

Determine exactly:

what operation is timed;
whether query embedding time is included;
whether model initialization is included;
whether disk loading is included;
number of warmup runs;
number of measured runs;
number of queries;
whether timing uses time.perf_counter;
whether each retriever receives identical queries;
hardware and device used.
Required benchmark design

Benchmark the end-to-end retrieval path separately from individual components where useful.

At minimum define clearly:

End-to-end query latency
Query
→ query embedding
→ retrieval
→ result mapping

Also, if practical, separately measure:

embedding latency;
FAISS search latency;
BM25 search latency;
hybrid retrieval latency.
Statistical reporting

Do not report only one average.

For each method, report:

mean latency;
median/P50 latency;
P95 latency;
minimum;
maximum;
number of measured queries/runs.

Use milliseconds.

Benchmark controls
Include warmup runs.
Exclude one-time model/index initialization from steady-state query latency.
Use identical query sets across methods.
Handle outliers without silently deleting them.
Record corpus size.
Record CPU/GPU device.
Record relevant hardware information where available.
Reproducible output

Save raw measurements to CSV or JSON.

Generate a Markdown summary from the raw data.

Include metadata:

timestamp;
Python version;
model;
FAISS index type;
corpus size;
benchmark configuration.
Critical wording

The benchmark may support wording such as:

"measured low-latency retrieval on a 10K-passage corpus"

only if the measurements support it.

It must NOT support:

"large-scale search"

or

"scalable ANN retrieval"

unless the repository actually demonstrates those capabilities.

Final output

Provide:

the exact benchmark command;
methodology;
raw result artifact;
summary table;
clear statement of what performance claims are supported.

Do not invent or preserve stale benchmark numbers if they cannot be reproduced.

Prompt 6 — Improve error handling and production readiness

You are making targeted engineering improvements to an existing Semantic Search Engine repository.

The repository currently has minimal error handling and may expose raw exceptions for missing files, invalid indexes, malformed data, or invalid user input.

Your goal is to make failures understandable without turning this into an overengineered production platform.

Inspect actual failure points

Find operations involving:

loading models;
loading the FAISS index;
loading document metadata;
reading datasets;
parsing evaluation data;
generating embeddings;
running FAISS search;
running BM25 search;
hybrid score fusion;
Streamlit user input.
Required improvements
Actionable failures

Replace obscure downstream failures with clear messages where appropriate.

Examples:

missing index file;
missing metadata;
index cannot be deserialized;
index dimension differs from model embedding dimension;
dataset file missing;
malformed JSON/CSV;
query is invalid;
requested top_k is invalid.
Validation

Validate important assumptions at startup or before expensive operations:

index dimension matches embedding dimension;
metadata count/IDs are compatible with the index;
required columns exist;
evaluation data has required fields.
Logging

Add lightweight structured or standard Python logging.

Log useful lifecycle events:

model loaded;
corpus loaded;
index loaded;
index size;
evaluation started/completed;
benchmark started/completed.

Do not log every embedding or flood the console.

Streamlit UX

Ensure the application displays understandable user-facing errors instead of raw stack traces for expected failures.

Constraints
Do not catch broad Exception and silently continue.
Do not suppress errors.
Preserve stack traces/logging for unexpected failures.
Do not add a heavy observability framework.
Tests

Add tests for at least the most important validation paths.

Run the full test suite afterward.

Final report

List:

Failure Scenario	Previous Behavior	New Behavior	Tested?
Prompt 7 — Clean up project structure without a massive rewrite

You are improving the maintainability of an existing Semantic Search Engine repository before it is included in a competitive technical portfolio.

The current repository is functional but has a relatively flat structure, including scripts such as:

app.py
benchmark.py
bm25_search.py
build_index.py
dataset_download.py
evaluation.py
hybrid_search.py
preprocess.py
vector_engine/utils.py

Your task is to improve organization only where it materially improves readability and maintainability.

First inspect imports and execution paths

Do not move files blindly.

Map:

public entry points;
imports;
circular dependency risks;
reusable library logic;
script-only orchestration.
Target structure

Aim for a lightweight structure conceptually similar to:

project/
├── src/ or semantic_search/
│   ├── retrieval/
│   ├── indexing/
│   ├── evaluation/
│   └── utils/
├── scripts/
├── tests/
├── data/
├── artifacts/
├── docs/
└── app.py

You may use a different structure if it fits the existing code better.

Rules
Separate reusable implementation from command-line scripts.
Preserve working public commands where practical.
Avoid unnecessary abstractions.
Do not introduce classes just to satisfy OOP conventions.
Avoid large-scale renaming.
Update imports and documentation.
Run tests after refactoring.
Verify the end-to-end workflows still work.
Deliverables

Provide:

old → new file mapping;
rationale for every moved module;
updated project tree;
exact commands for key workflows;
test results.

Do not perform cosmetic refactoring unrelated to maintainability.

Prompt 8 — Improve README and architecture documentation

You are documenting an existing Semantic Search Engine repository for a technical recruiter and ML engineer.

The documentation must be strictly evidence-based.

Before editing, inspect the actual implementation, tests, evaluation artifacts, and benchmark artifacts.

Update the README

The README should include:

1. One-paragraph project overview

Explain what the system actually does.

2. Architecture

Document the real pipeline:

Documents
  ↓
Preprocessing
  ↓
Transformer Embeddings
  ↓
FAISS Dense Index
  ↓
Query Embedding
  ↓
Dense Retrieval ──────┐
                      ├→ Hybrid Score Fusion → Ranked Results
BM25 Retrieval ───────┘

Only include components that actually exist.

3. Technical stack

Clearly distinguish:

directly used technologies;
transitive dependencies.

Do not claim direct PyTorch implementation if PyTorch is only used underneath Sentence Transformers.

4. Retrieval methods

Explain:

BM25;
dense semantic retrieval;
hybrid retrieval;
the actual fusion method.
5. Evaluation

Document:

dataset/corpus;
corpus size;
query count;
metrics;
comparison methodology;
limitations.
6. Benchmarking

Document:

exactly what latency means;
hardware/environment;
corpus size;
benchmark methodology.

Do not use "large-scale", "highly scalable", or similar language unless proven.

7. Reproduction

Provide exact commands for:

installation;
data preparation;
index building;
evaluation;
benchmarking;
tests;
application.
8. Limitations

Explicitly mention:

approximately 10K-passage corpus;
exact IndexFlatIP retrieval;
current evaluation query count;
what the project does not claim.
Documentation quality

Make the project understandable within 60 seconds of reading the README.

Use real measured numbers only when they are verified and reproducible.

Do not invent benchmark results, model capabilities, scale, or improvements.

Prompt 9 — Verify resume claims against the final repository

You are the final technical auditor for an ML/NLP repository that is about to be included on a resume.

Inspect the entire current repository after all fixes.

Do not modify code in this task unless a tiny documentation correction is required.

Audit the following candidate claims.

Claim A

Built a semantic and hybrid search engine using transformer embeddings, FAISS dense retrieval, BM25 lexical retrieval, and score fusion.

Claim B

Implemented FAISS vector retrieval over approximately 10,000 MS MARCO passages using normalized transformer embeddings and inner-product similarity.

Claim C

Evaluated BM25, semantic, and hybrid retrieval using Precision@10, Recall@10, and MRR@10.

Claim D

Achieved low-latency retrieval.

For each claim, assign exactly one verdict:

SUPPORTED
QUALIFIED
UNSUPPORTED
Evidence standard

For every verdict, cite:

exact files;
functions/classes;
tests;
evaluation artifacts;
benchmark artifacts.

Do not rely on README wording as evidence.

Numeric claims

For every proposed metric or latency number:

verify it exists in a reproducible artifact;
verify the artifact records relevant context;
verify it corresponds to the current implementation.

If any number cannot be verified, mark it:

DO NOT USE ON RESUME

Required output

Create:

Resume Claim	Verdict	Evidence	Safe Wording

Then provide:

1. Best three resume bullets today

Use only supported evidence and real measured numbers.

2. Claims that must not be made

List every unsupported or misleading claim.

3. Interview defense questions

Generate the 10 most likely technical interview questions about this project, including:

Why FAISS?
Why IndexFlatIP?
Why normalize embeddings?
Why use BM25?
Why Min-Max normalization?
Why hybrid retrieval?
How was MRR calculated?
What are the limitations of the 10K corpus?
What does benchmark latency include?
What would you change at 1M documents?

For each, provide a concise answer strictly grounded in the repository.

4. Final decision

Choose exactly one:

RESUME READY
RESUME READY WITH QUALIFIED WORDING
NOT YET RESUME READY

Justify the decision using evidence only.

Prompt 10 — Final adversarial repository audit

This is the last prompt. Use it after all fixes.

Act as an adversarial senior ML engineer, retrieval researcher, and technical interviewer.

This Semantic Search Engine repository is about to be added to a competitive resume.

Your goal is to find every remaining weakness, misleading claim, correctness bug, reproducibility problem, or interview red flag.

Assume that any unsupported claim may be challenged during a technical interview.

Audit the repository recursively

Inspect:

all source code;
all tests;
dependency files;
README and documentation;
evaluation datasets;
evaluation code;
evaluation outputs;
benchmark code;
benchmark outputs;
configuration;
CLI commands;
Streamlit application;
Git status where available.
Adversarial checks
Implementation

Ask:

Does every advertised feature actually execute?
Is any code dead or disconnected?
Are there hidden stubs?
Are IDs mapped correctly?
Is normalization mathematically consistent?
Does hybrid fusion behave correctly on edge cases?
Testing

Ask:

Are the important algorithms tested?
Are tests meaningful or superficial?
Would tests catch a regression in metric calculations?
Would tests catch invalid FAISS IDs?
Would tests catch constant-score normalization failures?
Evaluation

Ask:

Are metrics mathematically correct?
Are results reproducible?
Are all retrievers evaluated fairly?
Is the query set large enough for the claims made?
Are any results cherry-picked?
Benchmarking

Ask:

What exactly is timed?
Are initialization costs excluded appropriately?
Are warmups used?
Are P50/P95 available?
Are performance claims limited to the measured corpus size?
Resume honesty

Challenge every use of:

scalable;
large-scale;
low-latency;
high-performance;
improved;
state-of-the-art;
production-ready.

Require evidence for every adjective.

Required severity table

Classify every issue:

Severity	Issue	Evidence	Why an Interviewer Would Care	Required Action

Severity must be one of:

CRITICAL
HIGH
MEDIUM
LOW
Final scoring

Score:

Category	Score /10
Retrieval Correctness	

NLP Technical Depth	

Evaluation Rigor	

Benchmark Rigor	

Testing	

Code Quality	

Reproducibility	

Documentation	

Resume Credibility	

Overall	

Final verdict

Choose exactly one:

A. RESUME READY

No material unsupported claims remain.

B. RESUME READY WITH QUALIFIED WORDING

The implementation is strong enough, but wording must remain conservative.

C. ONE MORE FIX ROUND REQUIRED

A material issue would likely be exposed in a technical interview.

Then provide:

the exact three strongest resume bullets supported by evidence;
the exact technologies that may safely be listed;
technologies that should NOT be claimed;
the five most likely interview weaknesses;
the smallest remaining actions required before submitting the project on a resume.

Do not recommend new features unless they are necessary to correct a material weakness.
