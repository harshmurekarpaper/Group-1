# PhishGuard AI

## Explainable Machine Learning System for Phishing Detection and Digital Forensic Reporting

---

## 1. Project Overview

**PhishGuard AI** is an explainable machine-learning system for phishing detection and digital forensic reporting.

The system is designed around the following research pipeline:

> **Detection → Explanation → Evidence → Forensic Finding → Structured Report**

The core implementation is intentionally **hardcoded and deterministic** for the ML, XAI, evidence interpretation, and forensic finding layers.

The system does **not** use LLMs, RAG, AutoML, dynamic model selection, or AI agents to make the phishing prediction or generate the model explanation.

---

# 2. Tech Flow

```text
                    ┌───────────────────────────┐
                    │     PHISHING ARTIFACT     │
                    │                           │
                    │ • URL                     │
                    │ • Email                   │
                    │ • Web Evidence            │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │    ARTIFACT INGESTION      │
                    │                           │
                    │ • Input Validation        │
                    │ • Artifact Identification  │
                    │ • Timestamp / Metadata    │
                    │ • Evidence ID             │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ HARD-CODED FEATURE        │
                    │ EXTRACTION                │
                    │                           │
                    │ URL Features              │
                    │ • URL Length               │
                    │ • Token Structure         │
                    │ • Subdomain Depth         │
                    │ • Special Characters      │
                    │ • Suspicious Keywords     │
                    │ • Domain Characteristics  │
                    │                           │
                    │ Email/Web Features        │
                    │ • Headers / Metadata      │
                    │ • Links / Redirects       │
                    │ • Page Indicators         │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                         ┌───────────────────┐
                         │ FIXED FEATURE     │
                         │ VECTOR            │
                         │                   │
                         │ Fixed feature     │
                         │ order             │
                         └─────────┬─────────┘
                                   │
                                   ▼
                    ┌───────────────────────────┐
                    │   HARDCODED ML ENGINE     │
                    │                           │
                    │   XGBoost Classifier      │
                    │   Fixed Parameters        │
                    │   Fixed Features         │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴──────────────┐
                    ▼                            ▼
             Class Prediction              Probability
                    │                            │
                    └─────────────┬──────────────┘
                                  ▼
                    ┌───────────────────────────┐
                    │ FIXED DECISION RULE       │
                    │                           │
                    │ Probability ≥ 0.50        │
                    │       → PHISHING          │
                    │ Probability < 0.50        │
                    │       → LEGITIMATE        │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │    HARDCODED XAI LAYER   │
                    │                           │
                    │ SHAP TreeExplainer        │
                    │ Feature Contributions     │
                    │ Hardcoded Interpretation  │
                    │ Rules                     │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ EVIDENCE & CORRELATION    │
                    │                           │
                    │ • Prediction              │
                    │ • Probability             │
                    │ • SHAP Values              │
                    │ • Important Features      │
                    │ • Artifact Evidence       │
                    │ • Metadata                │
                    │ • Model Version           │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ HARDCODED FORENSIC        │
                    │ FINDING ENGINE            │
                    │                           │
                    │ • Evidence Interpretation │
                    │ • Indicator Correlation   │
                    │ • Analytical Findings     │
                    │ • Provenance Tracking     │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ REPORT GENERATION         │
                    │                           │
                    │ • JSON                    │
                    │ • HTML                    │
                    │ • PDF                     │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ INVESTIGATOR INTERFACE    │
                    │                           │
                    │ Classification            │
                    │ Confidence                │
                    │ Indicators                │
                    │ XAI Explanation           │
                    │ Evidence                  │
                    │ Report                    │
                    └───────────────────────────┘
```

---

# 3. Technology Flow

```text
INPUT
  │
  ├── URL
  ├── Email
  └── Web Evidence
        │
        ▼
Python Backend
        │
        ▼
Hardcoded Feature Extraction
        │
        ▼
Pandas / NumPy
        │
        ▼
Fixed XGBoost Model
        │
        ▼
Fixed Prediction Threshold
        │
        ▼
SHAP TreeExplainer
        │
        ▼
Hardcoded Interpretation Rules
        │
        ▼
Evidence Correlation
        │
        ▼
Hardcoded Forensic Findings
        │
        ▼
Report Generator
        │
        ├── JSON
        ├── HTML
        └── PDF
```

---

# 4. Technology, Tools & Stack

## 4.1 Core Development Stack

| Layer | Technology | Purpose |
|---|---|---|
| Primary Language | **Python 3.11+** | Backend, ML, XAI and forensic pipeline |
| Backend Framework | **FastAPI** | REST API and service orchestration |
| API Server | **Uvicorn** | Running FastAPI |
| Package Management | **pip / venv** | Dependency management |
| Development IDE | **VS Code** | Development and debugging |
| Version Control | **Git + GitHub** | Source-code and experiment management |

---

## 4.2 Data Processing & Feature Engineering

| Technology / Tool | Purpose |
|---|---|
| **Pandas** | Dataset manipulation and preprocessing |
| **NumPy** | Numerical computation |
| **Scikit-learn** | Preprocessing, training utilities and evaluation |
| **Matplotlib** | Evaluation plots and visualizations |
| **Joblib** | Saving/loading trained model artifacts |

### Phishing Features

```text
URL
├── URL length
├── Number of dots
├── Number of subdirectories
├── Number of parameters
├── Special characters
├── IP address usage
├── Suspicious keywords
├── Subdomain depth
├── HTTPS usage
└── URL token characteristics

Domain
├── Domain characteristics
├── Domain age indicators
├── Registration-related information
└── Suspicious domain patterns

Web / Email
├── Redirects
├── Links
├── Metadata
├── Header information
└── Page indicators
```

---

# 5. Hardcoded Machine Learning Architecture

## 5.1 Fixed ML Model

The primary ML model is **XGBoost**.

The implementation must not dynamically select models at runtime.

```python
MODEL_TYPE = "XGBoost"

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)
```

The following are fixed in source code:

- Model architecture
- Hyperparameters
- Feature order
- Preprocessing logic
- Decision threshold
- Inference logic
- XAI method
- Interpretation rules

There is no runtime model selection such as:

```python
model = select_best_model()
```

and no conditional model switching based on dataset or input.

---

## 5.2 Hardcoded Feature Extraction

Every feature is explicitly defined.

```python
FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "dot_count",
    "hyphen_count",
    "slash_count",
    "question_mark_count",
    "equal_count",
    "at_count",
    "digit_count",
    "subdomain_count",
    "has_ip_address",
    "has_https",
    "has_suspicious_keyword"
]
```

Example implementation:

```python
def extract_features(url):

    parsed = urlparse(url)

    features = [
        len(url),
        len(parsed.hostname or ""),
        len(parsed.path),
        len(parsed.query),
        url.count("."),
        url.count("-"),
        url.count("/"),
        url.count("?"),
        url.count("="),
        url.count("@"),
        sum(c.isdigit() for c in url),
        count_subdomains(parsed.hostname),
        contains_ip_address(parsed.hostname),
        int(parsed.scheme == "https"),
        contains_suspicious_keyword(url)
    ]

    return np.array(features).reshape(1, -1)
```

The model always receives the same feature ordering.

---

# 6. Hardcoded Prediction Logic

The inference threshold is fixed.

```python
PHISHING_THRESHOLD = 0.50

probability = model.predict_proba(features)[0][1]

if probability >= PHISHING_THRESHOLD:
    prediction = "PHISHING"
else:
    prediction = "LEGITIMATE"
```

There is no dynamic threshold optimization during inference.

---

# 7. Hardcoded Explainable AI

## 7.1 Fixed XAI Method

The fixed XAI method is:

> **SHAP TreeExplainer**

```python
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(features)
```

The architecture is:

```text
XGBoost
   ↓
SHAP TreeExplainer
   ↓
Feature Contributions
   ↓
Hardcoded Interpretation Rules
   ↓
Human-readable Explanation
```

There is no runtime selection between SHAP, LIME, or another explainer.

---

## 7.2 Hardcoded Explanation Rules

SHAP provides the mathematical feature contribution. Fixed Python rules convert the relevant contributions into investigator-readable descriptions.

```python
EXPLANATION_RULES = {
    "url_length": "Unusually long URL structure",
    "hostname_length": "Unusually long hostname",
    "subdomain_count": "Excessive subdomain depth",
    "has_ip_address": "IP address used instead of conventional domain",
    "has_suspicious_keyword": "Suspicious phishing-related keyword detected",
    "at_count": "Potentially deceptive @ character detected",
    "digit_count": "Unusual digit density in URL",
}
```

Example:

```python
if shap_value > 0:
    explanation = EXPLANATION_RULES[feature]
```

The system does not use an LLM to explain the model.

---

# 8. Explicitly Excluded From ML/XAI

The research implementation excludes:

```text
❌ ChatGPT
❌ Gemini
❌ Claude
❌ Llama
❌ RAG
❌ Generative AI explanation
❌ Prompt-based explanation
❌ AI agent deciding the prediction
❌ AutoML
❌ Dynamic model selection
❌ Runtime selection between ML models
❌ Runtime selection between XAI methods
```

The ML/XAI chain is fixed:

```text
XGBoost
   ↓
SHAP
   ↓
Hardcoded Interpretation Rules
```

---

# 9. Digital Forensics Stack

| Technology / Tool | Purpose |
|---|---|
| **Python** | Forensic processing pipeline |
| **JSON** | Structured evidence representation |
| **SQLite** | Initial evidence and investigation storage |
| **PostgreSQL** | Optional production-scale database |
| **SHA-256** | Artifact integrity/hash identification |
| **UUID** | Unique evidence/artifact identifiers |
| **UTC timestamps** | Evidence timeline |
| **Pydantic** | Evidence/report schema validation |

## Evidence Model

```text
Artifact
   │
   ├── Artifact ID
   ├── SHA-256 Hash
   ├── Acquisition Timestamp
   └── Source
   │
   ▼
Extracted Evidence
   │
   ├── URL features
   ├── Domain information
   ├── Email metadata
   ├── Redirect information
   └── Web indicators
   │
   ▼
ML Prediction
   │
   ▼
XAI Explanation
   │
   ▼
Evidence Correlation
   │
   ▼
Analytical Finding
   │
   ▼
Forensic Report
```

---

# 10. Evidence & Provenance Layer

PhishGuard AI maintains traceability from the original artifact to the final report.

```text
Evidence ID
    │
    ▼
Original Artifact
    │
    ▼
SHA-256 Hash
    │
    ▼
Feature Extraction
    │
    ▼
Model Version
    │
    ▼
Prediction
    │
    ▼
SHAP Explanation
    │
    ▼
Evidence Correlation
    │
    ▼
Forensic Finding
    │
    ▼
Generated Report
```

This provides a traceable chain showing how a finding was produced from the original artifact.

---

# 11. Hardcoded Evidence Correlation

Evidence correlation uses deterministic rules.

```python
if has_ip_address:
    evidence.append(
        "The URL uses an IP address instead of a conventional hostname."
    )

if subdomain_count >= 3:
    evidence.append(
        "The URL contains multiple subdomain levels."
    )

if has_suspicious_keyword:
    evidence.append(
        "The URL contains a suspicious keyword."
    )
```

No additional AI model is required for this layer.

---

# 12. Hardcoded Forensic Finding Generation

Forensic findings are generated from known evidence and the fixed model output.

```python
if prediction == "PHISHING":

    finding = (
        "The artifact was classified as phishing based on "
        "the combination of extracted URL characteristics "
        "and model-supported feature contributions."
    )

else:

    finding = (
        "The artifact was classified as legitimate based on "
        "the extracted characteristics and model prediction."
    )
```

The finding-generation logic is deterministic.

> **Important:** A machine-learning prediction is an analytical output and should not be represented as unquestionable forensic fact.

---

# 13. Backend / API Architecture

| Technology | Purpose |
|---|---|
| **FastAPI** | Backend REST API |
| **Uvicorn** | ASGI server |
| **Pydantic** | Request/response validation |
| **Python logging** | Application and investigation logs |
| **SQLite** | Lightweight development database |
| **PostgreSQL** | Optional production database |

## API Structure

```text
POST /analyze
        ↓
POST /predict
        ↓
POST /explain
        ↓
GET /evidence/{id}
        ↓
GET /report/{id}
```

A unified analysis endpoint can orchestrate:

```text
POST /analyze
     │
     ├── Feature Extraction
     ├── ML Prediction
     ├── SHAP Explanation
     ├── Evidence Correlation
     └── Report Generation
```

---

# 14. Report Generation

| Tool | Purpose |
|---|---|
| **Jinja2** | HTML report templates |
| **JSON** | Machine-readable forensic report |
| **HTML** | Human-readable report |
| **PDF renderer** | PDF report generation |
| **Pydantic** | Report schema validation |
| **Python** | Report orchestration |

## Report Structure

```text
FORENSIC REPORT
│
├── Case Information
├── Artifact Information
├── Evidence Identifier
├── Artifact Hash
├── Timestamp
├── Detection Result
├── Confidence
├── Key Indicators
├── Extracted Features
├── XAI Explanation
├── Evidence Correlation
├── Analytical Findings
├── Model Information
└── Investigation Summary
```

---

# 15. Frontend / Investigator Interface

If a graphical interface is implemented:

| Technology | Purpose |
|---|---|
| **React** | Frontend framework |
| **Vite** | Frontend build tool |
| **JavaScript / TypeScript** | Frontend development |
| **HTML/CSS** | User interface |
| **Axios / Fetch** | API communication |
| **Recharts / Chart.js** | XAI and evidence visualization |

## Investigator Dashboard

```text
┌─────────────────────────────────────────────┐
│             PHISHGUARD AI                   │
├─────────────────────────────────────────────┤
│ Artifact Input                              │
│ [ URL / Email / Evidence ]    [ ANALYZE ]  │
├─────────────────────────────────────────────┤
│ Classification:  PHISHING                  │
│ Confidence:      96.2%                     │
├─────────────────────────────────────────────┤
│ Key Indicators                              │
│ • Suspicious URL structure                  │
│ • Abnormal domain characteristics           │
│ • Suspicious keywords                       │
├─────────────────────────────────────────────┤
│ XAI Explanation                             │
│ Feature contributions → visualization       │
├─────────────────────────────────────────────┤
│ Evidence Timeline / Provenance              │
├─────────────────────────────────────────────┤
│ [ Generate JSON ] [ Generate PDF ]          │
└─────────────────────────────────────────────┘
```

---

# 16. Testing & Evaluation Stack

| Tool | Purpose |
|---|---|
| **PyTest** | Unit and integration testing |
| **Scikit-learn metrics** | ML evaluation |
| **Matplotlib** | Confusion matrix, ROC and PR visualization |
| **Jupyter Notebook** | Research experiments and analysis |
| **MLflow** | Optional experiment/model tracking |

## Evaluation

```text
                    MODEL EVALUATION
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    Classification     Generalization     XAI
      Metrics            Testing        Evaluation
          │                │                │
    Accuracy          Cross-Dataset    Explanation
    Precision         Temporal         Consistency
    Recall             Testing         Stability
    F1                 Unseen Data
    ROC-AUC
```

The evaluation should cover:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Cross-dataset performance
- Temporal/generalization performance
- Unseen-sample performance
- Explanation consistency
- False-positive behavior
- False-negative behavior
- Evidence traceability
- Report completeness

---

# 17. Security Tools

| Tool / Practice | Purpose |
|---|---|
| **Git** | Version control |
| **Bandit** | Python security linting |
| **pip-audit** | Dependency vulnerability checking |
| **OWASP practices** | API/web security |
| **SHA-256** | Evidence integrity |
| **Environment variables** | Secret/configuration management |

For deployment involving active web retrieval, suspicious content should be isolated/sandboxed rather than directly executed or visited from the production analysis server.

---

# 18. Deployment Stack

## Development

```text
Windows / Linux
      │
      ├── Python
      ├── FastAPI
      ├── XGBoost
      ├── SHAP
      ├── SQLite
      └── React/Vite
```

## Production Option

```text
                    Internet
                       │
                       ▼
                  Reverse Proxy
                  Nginx / Caddy
                       │
                       ▼
                  FastAPI API
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       ML Model      XAI        Evidence DB
          │            │            │
          └────────────┼────────────┘
                       ▼
                 Report Engine
                       │
                 ┌─────┴─────┐
                 ▼           ▼
               JSON         PDF
```

Docker can package the backend, frontend, and database for reproducible deployment.

---

# 19. Complete System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHISHGUARD AI                                     │
│     Explainable ML-Based Phishing Detection & Forensic Reporting System     │
└─────────────────────────────────────────────────────────────────────────────┘

                              INPUT LAYER
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
        ┌───────────┐        ┌───────────┐        ┌──────────────┐
        │    URL    │        │   Email   │        │ Web Evidence │
        │  Artifact │        │  Artifact │        │ / Metadata   │
        └─────┬─────┘        └─────┬─────┘        └──────┬───────┘
              │                    │                     │
              └────────────────────┼─────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ARTIFACT INGESTION LAYER                             │
│                                                                             │
│  • Input Validation                                                        │
│  • Artifact ID Generation                                                  │
│  • SHA-256 Hash                                                            │
│  • Timestamp                                                                │
│  • Source / Metadata                                                       │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HARDCODED FEATURE EXTRACTION LAYER                       │
│                                                                             │
│  URL Features                Domain Features          Web/Email Features    │
│  ───────────                 ───────────────          ───────────────────   │
│  • URL length                • Domain structure      • Redirects            │
│  • Dot count                 • Subdomain depth      • Links                 │
│  • Slash count               • IP address           • Metadata              │
│  • Special chars             • Domain indicators    • Headers               │
│  • Keywords                  • HTTPS                • Page indicators        │
│  • Digit count                                                              │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ FIXED FEATURE     │
                         │ VECTOR            │
                         │                   │
                         │ Hardcoded order   │
                         │ Hardcoded logic   │
                         └─────────┬─────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       HARDCODED ML ENGINE                                   │
│                                                                             │
│                         ┌─────────────────┐                                 │
│                         │    XGBOOST      │                                 │
│                         │   CLASSIFIER    │                                 │
│                         │                 │                                 │
│                         │ Fixed model     │                                 │
│                         │ Fixed params    │                                 │
│                         │ Fixed features  │                                 │
│                         └────────┬────────┘                                 │
│                                  │                                          │
│                    ┌─────────────┴──────────────┐                           │
│                    ▼                            ▼                           │
│             Class Prediction              Probability                      │
│                    │                            │                           │
│                    └─────────────┬──────────────┘                           │
│                                  ▼                                          │
│                     ┌────────────────────────┐                              │
│                     │ FIXED DECISION RULE    │                              │
│                     │                        │                              │
│                     │ Probability ≥ 0.50     │                              │
│                     │       → PHISHING       │                              │
│                     │ Probability < 0.50     │                              │
│                     │       → LEGITIMATE     │                              │
│                     └───────────┬────────────┘                              │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HARDCODED XAI LAYER                                │
│                                                                             │
│                         ┌─────────────────┐                                 │
│                         │ SHAP            │                                 │
│                         │ TreeExplainer   │                                 │
│                         └────────┬────────┘                                 │
│                                  │                                          │
│                                  ▼                                          │
│                     Feature Contributions                                  │
│                                  │                                          │
│                                  ▼                                          │
│                  ┌───────────────────────────┐                              │
│                  │ HARDCODED INTERPRETATION  │                              │
│                  │ RULES                     │                              │
│                  │                           │                              │
│                  │ Feature → Meaning         │                              │
│                  └─────────────┬─────────────┘                              │
└────────────────────────────────┼────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     EVIDENCE & CORRELATION LAYER                            │
│                                                                             │
│  ML Evidence                 XAI Evidence              Artifact Evidence   │
│  ───────────                 ───────────               ────────────────     │
│  • Prediction                • SHAP values             • URL                │
│  • Probability               • Important features      • Domain             │
│  • Model version             • Contribution direction  • Metadata           │
│                                                                             │
│                         ↓                                                   │
│                 Evidence Correlation                                        │
│                         ↓                                                   │
│                 Correlated Indicators                                      │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HARDCODED FORENSIC FINDING ENGINE                        │
│                                                                             │
│  • Evidence Interpretation                                                  │
│  • Indicator Correlation                                                    │
│  • Analytical Findings                                                      │
│  • Evidence Provenance                                                      │
│  • Timestamp / Model Version                                                │
│                                                                             │
│  IMPORTANT: Model prediction ≠ forensic fact                               │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        REPORT GENERATION LAYER                              │
│                                                                             │
│                    ┌────────────────────────┐                               │
│                    │ Structured Report      │                               │
│                    └───────────┬────────────┘                               │
│                                │                                            │
│                ┌───────────────┼────────────────┐                           │
│                ▼               ▼                ▼                           │
│             JSON            HTML              PDF                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INVESTIGATOR LAYER                                  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                       PhishGuard Dashboard                             │  │
│  │                                                                       │  │
│  │ Classification │ Confidence │ Indicators │ XAI │ Evidence │ Report   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘


                         PERSISTENCE / AUDIT
                                 │
             ┌───────────────────┼───────────────────┐
             ▼                   ▼                   ▼
       ┌───────────┐       ┌────────────┐      ┌──────────────┐
       │  SQLite   │       │ Evidence   │      │ Model / Code │
       │ Database  │       │ Metadata   │      │ Versioning   │
       └───────────┘       └────────────┘      └──────────────┘
```

---

# 20. Technology Mapping

```text
┌────────────────────────────────────────────────────────────┐
│                    PHISHGUARD AI                           │
├────────────────────────────────────────────────────────────┤
│ Input                 → Python                            │
│ API                   → FastAPI + Uvicorn                 │
│ Feature Extraction    → Python + Pandas + NumPy           │
│ ML                    → XGBoost                           │
│ ML Preprocessing      → Scikit-learn                      │
│ XAI                   → SHAP TreeExplainer                │
│ XAI Interpretation    → Hardcoded Python Rules            │
│ Evidence              → Pydantic + JSON                   │
│ Integrity             → SHA-256                           │
│ Storage               → SQLite                            │
│ Report                → Jinja2 + HTML + PDF               │
│ Frontend              → React + Vite                      │
│ Testing               → PyTest                            │
│ Version Control       → Git                               │
│ Deployment            → Docker                            │
└────────────────────────────────────────────────────────────┘
```

---

# 21. Final Fixed ML/XAI Stack

| Component | Fixed Technology |
|---|---|
| Language | **Python** |
| Feature extraction | **Custom hardcoded Python code** |
| Data processing | **Pandas + NumPy** |
| ML model | **XGBoost** |
| ML framework | **XGBoost + Scikit-learn** |
| Model parameters | **Hardcoded** |
| Feature order | **Hardcoded** |
| Decision threshold | **Hardcoded** |
| XAI | **SHAP TreeExplainer** |
| Explanation rules | **Hardcoded Python rules** |
| Evidence correlation | **Hardcoded Python rules** |
| Forensic findings | **Hardcoded deterministic logic** |
| Report schema | **Pydantic** |
| Report formats | **JSON + HTML + PDF** |
| Database | **SQLite initially** |
| Backend | **FastAPI** |
| Testing | **PyTest** |
| Version control | **Git** |

---

# 22. Complete Hardcoded ML/XAI Pipeline

```text
                 PHISHING ARTIFACT
                       │
                       ▼
               URL / Email Input
                       │
                       ▼
         ┌──────────────────────────┐
         │ HARD-CODED FEATURE       │
         │ EXTRACTION               │
         │                          │
         │ Predefined phishing      │
         │ features                 │
         └────────────┬─────────────┘
                      │
                      ▼
              Fixed Feature Vector
                      │
                      ▼
         ┌──────────────────────────┐
         │ FIXED XGBOOST MODEL      │
         │                          │
         │ Fixed hyperparameters    │
         │ Fixed feature order      │
         │ Fixed threshold          │
         └────────────┬─────────────┘
                      │
                      ▼
                 Probability
                      │
                      ▼
         ┌──────────────────────────┐
         │ FIXED DECISION RULE      │
         │                          │
         │ probability >= 0.50      │
         │       ↓                  │
         │    PHISHING              │
         └────────────┬─────────────┘
                      │
                      ▼
         ┌──────────────────────────┐
         │ FIXED SHAP EXPLAINER     │
         │                          │
         │ TreeExplainer            │
         └────────────┬─────────────┘
                      │
                      ▼
            Feature Contributions
                      │
                      ▼
         ┌──────────────────────────┐
         │ HARDCODED INTERPRETATION │
         │ RULES                    │
         └────────────┬─────────────┘
                      │
                      ▼
                 Evidence List
                      │
                      ▼
         ┌──────────────────────────┐
         │ FORENSIC FINDING ENGINE  │
         │                          │
         │ Fixed deterministic      │
         │ rules                    │
         └────────────┬─────────────┘
                      │
                      ▼
         ┌──────────────────────────┐
         │ STRUCTURED REPORT        │
         │                          │
         │ JSON / HTML / PDF        │
         └──────────────────────────┘
```

---

# 23. Core Research Architecture

The definitive research architecture is:

```text
PHISHING ARTIFACT
       ↓
HARDCODED FEATURE EXTRACTION
       ↓
HARDCODED XGBOOST MODEL
       ↓
PHISHING / LEGITIMATE + CONFIDENCE
       ↓
HARDCODED SHAP XAI
       ↓
FEATURE CONTRIBUTIONS
       ↓
HARDCODED INTERPRETATION RULES
       ↓
EVIDENCE CORRELATION
       ↓
HARDCODED FORENSIC FINDINGS
       ↓
STRUCTURED FORENSIC REPORT
```

This keeps the **ML, XAI, evidence interpretation, and forensic finding logic deterministic and explicitly implemented in code**, rather than delegating those decisions to an LLM or dynamic agent.

---

# 24. Hardcoding Requirements

The following requirements are mandatory for the implementation:

1. **One fixed primary ML model:** XGBoost.
2. **One fixed XAI method:** SHAP TreeExplainer.
3. Feature names and feature order must be hardcoded.
4. Feature extraction logic must be explicitly implemented in Python.
5. ML hyperparameters must be explicitly defined.
6. The inference threshold must be explicitly defined.
7. XAI interpretation mappings must be hardcoded.
8. Evidence-correlation rules must be hardcoded.
9. Forensic-finding generation must be deterministic.
10. No runtime model selection.
11. No runtime XAI selection.
12. No AutoML.
13. No LLM-based prediction.
14. No LLM-based explanation.
15. No RAG-based prediction or explanation.
16. No AI agent deciding the classification.
17. The trained model artifact must be versioned.
18. The model version must be recorded with each analysis.
19. The original artifact must be identifiable through an evidence ID and hash.
20. The final report must retain the prediction, model information, evidence, XAI output, and provenance.

---

# 25. Important Model Implementation Distinction

**Hardcoded does not mean manually writing learned model weights into Python.**

Training may still use a phishing dataset to learn the XGBoost parameters.

The hardcoded implementation means that the following are explicitly fixed in source code:

```text
Model architecture
        ↓
Feature engineering
        ↓
Feature order
        ↓
Hyperparameters
        ↓
Inference logic
        ↓
Decision threshold
        ↓
XAI method
        ↓
Interpretation rules
        ↓
Evidence rules
        ↓
Forensic finding rules
```

The trained XGBoost model can then be saved as a fixed, versioned model artifact and loaded by the application.

---

# 26. Recommended Core Prototype

For the academic prototype, the mandatory core stack is:

> **Python + FastAPI + Pandas + NumPy + Scikit-learn + XGBoost + SHAP + Pydantic + SQLite + Jinja2 + JSON/PDF + React/Vite + Git + PyTest**

Optional components should not be allowed to change the core deterministic ML/XAI architecture.

The core research pipeline remains:

> **Phishing Detection → Explainability → Evidence Correlation → Forensic Findings → Structured Reporting**
