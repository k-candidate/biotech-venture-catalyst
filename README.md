# Biotech Venture Catalyst: Predicting Phase 1 Trial Outcomes for Venture Capital Due Diligence


[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://biotech-venture-catalyst.streamlit.app/)

**Live Interactive Dashboard:** https://biotech-venture-catalyst.streamlit.app/

**Optional Client-Side Browser Demo (ONNX and WASM):** https://k-candidate.github.io/biotech-venture-catalyst/

**End-to-End Video Walkthrough:**

[![Biotech Venture Catalyst Demo](https://youtu.be/qqN5eXiKUrU)](https://youtu.be/qqN5eXiKUrU)

***

## Problem Description and Context

Bringing a therapeutic drug to market is a costly, uncertain process. Clinical trials evaluate investigational drugs in phases, beginning with studies of safety, tolerability, and pharmacokinetics.

Statistically, 50% of drugs fail during Phase 1 ([source](https://go.bio.org/rs/490-EHZ-999/images/ClinicalDevelopmentSuccessRates2011_2020.pdf)), frequently due to structural design flaws, unrealistic timelines, unmanageable protocols, or unexpected side-effects. For venture capital funds and pharmaceutical companies, a Phase 1 failure can mean substantial sunk costs and years of lost development time.

Evaluating the operational feasibility and regulatory risk of early-stage clinical trial designs is traditionally a slow, manual due-diligence process handled by medical boards and investment analysts.

***

## Objective

The goal of this project is to build a data-driven decision-support tool that predicts whether a proposed Phase 1 clinical trial is statistically likely to succeed (complete its objectives) or fail (be terminated, suspended, or withdrawn) based purely on its operational design metrics.

The notebook compares **Logistic Regression**, **Random Forest**, and **XGBoost** using historical trial metadata from the Therapeutics Data Commons ([TDC](https://tdcommons.ai/)). It selects Random Forest as the best-performing model, and `train.py` trains that model for production.

***

## How the Solution is Used

This solution bridges engineering, clinical strategy, and venture capital operations. It can be utilized in two distinct scenarios:
1. **Venture Capital (VC) Due Diligence**: Investment associates can plug a target biotech startup's proposed clinical trial design into the model. The model outputs a **Probability of Success (PoS)** score. Teams can use this estimate as one input to an independent rNPV (Risk-Adjusted Net Present Value) analysis, and compare estimates across hypothetical trial inputs.
2. **Enterprise Pharma Leadership**: Clinical operations executives can simulate variations of a trial setup before submitting it to the FDA/EMA. For example, they can test how modifying patient exclusion metrics or accelerating timelines impacts the probability of a successful completion, reducing institutional R&D waste. NB: these comparisons are exploratory; they do not show that changing the trial design will cause its outcome to improve.

***

## Data Sourcing & Local Replication

The dataset used in this project is sourced directly from the **Therapeutics Data Commons (TDC)** under the `TrialOutcome` (Phase 1 Success) benchmark collection. 

You can fetch the original tabular data using:

```python
from tdc.multi_pred import TrialOutcome

data = TrialOutcome(name='phase1').get_data()
```

The project includes the source data in `data/phase1.tab` and a CSV version in `data/clinical_trials_raw.csv`.

***

## Environment & Dependency Management

This project uses **`uv`** to manage the project environment and locked dependencies.  
Follow the steps below to initialize the exact local runtime workspace.

### 1. Install `uv` (If not already installed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Initialize Virtual Environment & Install Locked Dependencies

Navigate to the root directory of the project and execute:

```bash
# Synchronizes and creates a project virtual environment automatically
uv sync --frozen
```

### 3. Manual Activation (Optional)

If your terminal shell layout requires manual virtual environment sourcing:

```bash
# macOS/Linux
source .venv/bin/activate
```

***

## Training the Production Pipeline (`train.py`)

To train the model, run the script below. It loads the CSV data, splits it into training, validation, and test sets (60/20/20), trains a Random Forest with `max_depth=10` and `n_estimators=50`, evaluates it on the hold-out test set, and saves the scikit-learn pipeline to `model.bin`.

```bash
uv run python train.py
```

*Expected Output:*

```text
Starting production model training script pipeline...
Loading raw data matrix...
Running text and temporal feature transformations...
Assembling scikit-learn preprocessing and model pipeline...
Verified Final Test Hold-out ROC AUC: 0.63286
Exporting pipeline model artifact to: model.bin
train.py successfully executed. Training completed successfully.
```

***

## Containerization


### Option 1: Pull Pre-Built Image (Recommended)

You can pull and run the pre-built production image directly from Docker Hub without rebuilding it locally:

```bash
docker pull kcandidate/biotech-venture-catalyst:v1
docker run -it --rm -p 9696:9696 kcandidate/biotech-venture-catalyst:v1
```

### Option 2: Build Image Locally

Build the image:

```bash
docker build -t biotech-venture-catalyst:v1 .
```

Run the container:

```bash
docker run -it --rm -p 9696:9696 biotech-venture-catalyst:v1
```

### Verify Container Health

Once running, the server will boot up inside the container and begin listening for incoming inference payloads on `http://localhost:9696`.

You can visit `http://localhost:9696/docs` to test it via FastAPI’s interactive API documentation, or open a secondary terminal window and send an HTTP GET request to verify endpoint connectivity:

```bash
curl http://localhost:9696/health
```

*Expected JSON Response:*
```json
{"status": "healthy", "pipeline_loaded": true}
```

***

## Testing Local API and Sample Payloads

Once the Docker container is running, the FastAPI microservice is ready to accept raw clinical registry inputs, transform them on the fly, and calculate the Probability of Success (PoS).

You can send sample requests with `./test.sh`. Make sure to have `curl` and `jq`.

```bash
./test.sh
```

Expected output:

```
======== Test 01 ========

{
  "probability_of_success": 0.43057,
  "risk_status": "High Risk",
  "internals": {
    "start_year": 1996,
    "start_month": 3,
    "duration_days": 1583,
    "num_conditions": 2,
    "primary_disease_chapter": "D",
    "criteria_word_count": 6,
    "exclusion_keyword_count": 1
  }
}

======== Test 02 ========

{
  "probability_of_success": 0.67215,
  "risk_status": "Low Risk",
  "internals": {
    "start_year": 2003,
    "start_month": 6,
    "duration_days": 1614,
    "num_conditions": 12,
    "primary_disease_chapter": "P",
    "criteria_word_count": 99,
    "exclusion_keyword_count": 1
  }
}
```

***

### Running the Streamlit Dashboard Locally

To launch the interactive Streamlit web interface locally:

```bash
uv run streamlit run app.py
```

***

## Downstream Venture Capital Application (rNPV)

In institutional biotech due diligence, the **Probability of Success (PoS)** generated by this service serves as a risk-discounting variable in **Risk-Adjusted Net Present Value (rNPV)** models:

$$\text{rNPV} = \left( \frac{\text{Expected Market Value after Approval}}{(1 + \text{Discount Rate})^{\text{Years to Launch}}} \times \text{PoS} \right) - \text{Trial Operational Cost}$$

By automating the PoS estimation, investment teams can test asset valuations before committing capital to Phase 1 clinical operations.

***

## Client-Side Browser Inference (ONNX + WebAssembly)

In addition to the core FastAPI microservice and Streamlit dashboard, this repository includes a zero-backend static web page in `docs/` that runs model inference directly inside the user's browser:

* **Model Export (`export_onnx.py`):** Converts the trained `RandomForestClassifier` into `docs/model.onnx` and exports the `DictVectorizer` feature vocabulary to `docs/metadata.json`.
* **In-Browser Execution (`docs/index.html`):** Uses **ONNX Runtime Web (WASM)** to perform feature extraction and inference locally on the client device without sending data to a server.
* **Live GitHub Pages Deployment:** https://k-candidate.github.io/biotech-venture-catalyst/
