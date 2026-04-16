📘 README.md — Python Package In Conda

A fully self‑healing, auto-fixing, auto‑versioned, auto‑releasing MLOps system.

 

🚀 Overview

OASIS is a fully autonomous Machine Learning + DevOps hybrid pipeline featuring:

Real‑dataset LightGBM training

Versioned model saving

Semantic versioning

Full CLI toolkit ( oasis train ,  oasis version ,  oasis auto‑fix , etc.)

Automatic changelog generation

Automatic GitHub Releases

CI Retry + Auto‑Merge system

PR‑based self‑healing

Auto‑close failing PRs

Nightly auto‑fix pipelines

Auto‑formatting, linting, diagnostics, and repository cleanup

OASIS maintains itself — heals its own repo, fixes CI failures, formats code, retries CI, publishes releases, updates changelogs, and more.

 

📁 Project Structure

 
OASIS/
│
├── data/
│   └── dataset.csv
│
├── models/
│   ├── trained_model.pkl
│   ├── version.txt
│   └── history.log
│
├── src/
│   ├── train_pipeline.py
│   ├── model_loader.py
│   └── oasis/
│       └── cli.py
│
├── tests/
│   └── test_lgb_model.py
│
└── .github/workflows/
    ├── ci.yml
    ├── oasis-auto-fix.yml
    ├── oasis-auto-fix-pr.yml
    ├── oasis-auto-fix-nightly.yml
    ├── oasis-auto-merge.yml
    ├── oasis-auto-close.yml
    └── oasis-ci-retry.yml
 

 

🧠 Training Pipeline

Training uses:

 
src/train_pipeline.py
 

Pipeline includes:

Loading real dataset

Splitting training/test

Training LightGBM

Saving model + metadata

Recording semantic version

Appending version history

Train manually:

 
oasis train
 

 

🧪 Testing

Tests validate:

Model load

Feature alignment

Prediction behavior

Deterministic output

Run manually:

 
pytest -v
 

 

⚙️ GitHub Actions Overview

OASIS includes 7 fully autonomous workflows:

✔  ci.yml 

Standard train + test workflow.

✔  oasis-auto-fix.yml 

Self-heals repository on command.

✔  oasis-auto-fix-pr.yml 

Creates auto-fix PRs instead of pushing changes.

✔  oasis-auto-fix-nightly.yml 

Runs nightly repository healing at 2AM UTC.

✔  oasis-auto-merge.yml 

Auto-merges approved auto-fix PRs only when CI is green.

✔  oasis-auto-close.yml 

Auto-closes persistent failing PRs after 3 CI failures.

✔  oasis-ci-retry.yml 

Retries CI up to 3 times before merging or closing.

Combined, these workflows create a self-maintaining MLOps ecosystem.

 

🧵 OASIS CLI Commands

Your CLI includes:

🔧 Training & Model Management

 
oasis train
oasis evaluate <dataset.csv>
oasis predict <input.csv>
 

🔍 Model Metadata

 
oasis version
oasis version --json
 

Metadata includes:

Semantic version

Timestamp

Feature list

Model size

File path

🧾 Version History & Releases

 
oasis bump-version --level patch|minor|major
oasis history
oasis changelog
oasis release
 

Release automatically:

Tags Git

Generates changelog

Uploads model to GitHub Releases

🛠 Auto‑Fix & Formatting

 
oasis auto-fix
oasis auto-fix-strict
oasis format
oasis clean
 

🩺 Diagnostics

 
oasis doctor
oasis doctor --json
oasis doctor --fix
oasis doctor --fix --commit --push
 

Doctor checks:

Python syntax

YAML health

GPU availability

Missing dependencies

Model file integrity

Git status

Auto-healing

 

🤖 Self‑Healing DevOps Explained

OASIS includes autonomous maintenance loops:

1️⃣ Failure → Auto-Fix PR

A CI failure triggers a repair branch & PR.

2️⃣ Auto‑Retry CI

OASIS retries CI up to 3 times.

3️⃣ Auto‑Comment Failure Reasons

Explains why CI failed directly on PR.

4️⃣ Auto‑Merge

If CI passes + PR is approved → merge.

5️⃣ Auto‑Close

If CI fails 3 times → PR closed with explanation.

6️⃣ Nightly Repair

Nightly self-healing runs regardless of CI.

 

🚀 Release Automation

Release with:

 
oasis release
 

This:

Reads semantic version

Creates Git tag

Generates changelog

Uploads model

Publishes GitHub Release

Optional:

 
oasis release --no-confirm
oasis release --notes "Custom message"
 

 

🧹 Cleanup & Formatting

Run:

 
oasis clean
oasis format
 

Removes:

Caches

Build files

Logs

Model artifacts (optional)

And formats code using:

Black

isort

docformatter

 

📦 Installation

Editable mode installation:

 
pip install -e .
 

 Project files (all ready to save)

Below are the complete files you can save into your project exactly as shown.

---

phish.py
`python

phish.py
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

def load_data(path):
    """Load CSV into a DataFrame."""
    return pd.read_csv(path)

def detect_target(df, candidates=None):
    """Return the name of a likely target column or None."""
    if candidates is None:
        candidates = ["label", "target", "is_phish", "phishing", "class"]
    for c in candidates:
        if c in df.columns:
            return c
    for col in df.columns:
        if df[col].nunique() == 2:
            return col
    return None

def simpleurlfeatures(series):
    """Extract simple URL features from a text series."""
    out = pd.DataFrame()
    s = series.fillna("").astype(str)
    out["url_len"] = s.apply(len)
    out["num_dots"] = s.apply(lambda x: x.count("."))
    out["has_ip"] = s.apply(lambda x: bool(re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", x))).astype(int)
    return out

def preprocess(df, targetcol, textmaxfeatures=500, svdcomponents=10):
    """
    Preprocess DataFrame and return (Xarray, yarray).
    - Drops rows with missing target.
    - Encodes target if object.
    - Adds simple URL features if a URL-like column exists.
    - Vectorizes text columns (TF-IDF + SVD) and scales numeric columns.
    """
    df = df.dropna(subset=[targetcol]).resetindex(drop=True)
    if df[target_col].dtype == "object":
        le = LabelEncoder()
        df[targetcol] = le.fittransform(df[target_col])
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Add URL features if a likely URL column exists
    url_cols = [c for c in X.columns if "url" in c.lower() or "link" in c.lower() or "domain" in c.lower()]
    if url_cols:
        X = pd.concat([X.resetindex(drop=True), simpleurlfeatures(X[urlcols[0]]).reset_index(drop=True)], axis=1)

    textcols = X.selectdtypes(include=["object"]).columns.tolist()
    numcols = X.selectdtypes(include=[np.number]).columns.tolist()

    transformers = []
    if num_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", numpipeline, numcols))

    for col in text_cols:
        text_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(maxfeatures=textmax_features)),
            ("svd", TruncatedSVD(ncomponents=min(svdcomponents, max(1, textmaxfeatures//10))))
        ])
        transformers.append((f"text{col}", textpipeline, col))

    if not transformers:
        # No transformers: return numeric matrix (filled) and labels
        return X.fillna(0).values, y.values

    pre = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
    Xtrans = pre.fittransform(X)
    return X_trans, y.values
`

---

tests/test_phish.py
`python

tests/test_phish.py
import pandas as pd
import numpy as np
from phish import loaddata, detecttarget, preprocess

def makesampledf():
    return pd.DataFrame({
        "url": ["http://good.com", "http://bad.com/login", "http://192.168.0.1/mal"],
        "feature1": [1.0, 2.5, np.nan],
        "label": [0, 1, 1]
    })

def testdetecttarget_found():
    df = makesampledf()
    t = detect_target(df)
    assert t == "label"

def testloaddataandpreprocess(tmp_path):
    df = makesampledf()
    p = tmp_path / "sample.csv"
    df.to_csv(p, index=False)
    df2 = load_data(str(p))
    assert "label" in df2.columns
    X, y = preprocess(df2, "label", textmaxfeatures=10, svd_components=2)
    assert X.shape[0] == len(y)
    assert set(y) == {0, 1}

def testpreprocesshandlesmissingtarget():
    df = makesampledf()
    df.loc[0, "label"] = None
    X, y = preprocess(df, "label", textmaxfeatures=10, svd_components=2)
    assert len(y) == 2
`

---

tests/testintegrationmodel.py
`python

tests/testintegrationmodel.py
import numpy as np
from sklearn.modelselection import traintest_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pytest

@pytest.mark.slow
def testtinyxgboost_accuracy():
    # Create tiny synthetic dataset with a learnable pattern
    rng = np.random.RandomState(42)
    n = 200
    X = rng.randn(n, 10)
    # label depends on sum of first three features
    y = (X[:, :3].sum(axis=1) + 0.5 * rng.randn(n) > 0).astype(int)
    Xtrain, Xtest, ytrain, ytest = traintestsplit(X, y, testsize=0.25, randomstate=42, stratify=y)
    clf = xgb.XGBClassifier(uselabelencoder=False, evalmetric="logloss", randomstate=42, nestimators=50, maxdepth=3, n_jobs=1)
    clf.fit(Xtrain, ytrain)
    ypred = clf.predict(Xtest)
    acc = accuracyscore(ytest, y_pred)
    assert acc > 0.5, f"Expected accuracy > 0.5, got {acc:.3f}"
`

---

phishinganalysisnotebook.ipynb
Save the following JSON exactly into phishinganalysisnotebook.ipynb. It is a complete Jupyter notebook with the full analysis cells.

`json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Phishing Dataset Analysis\n",
    "\n",
    "This notebook loads malicious_phish.csv, runs preprocessing, trains XGBoost and LightGBM baselines, and shows quick EDA."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Requirements\n",
    "\n",
    "Install the packages before running:\n",
    "`\n",
    "pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn wordcloud tldextract imbalanced-learn\n",
    "`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import re\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import tldextract\n",
    "\n",
    "from sklearn.modelselection import traintest_split\n",
    "from sklearn.metrics import accuracyscore, classificationreport\n",
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.decomposition import TruncatedSVD\n",
    "\n",
    "from imblearn.over_sampling import SMOTE\n",
    "\n",
    "import xgboost as xgb\n",
    "import lightgbm as lgb\n",
    "\n",
    "from wordcloud import WordCloud\n",
    "\n",
    "RANDOM_STATE = 42\n",
    "TEST_SIZE = 0.2\n",
    "DATAPATH = \"maliciousphish.csv\"\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Helper functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def detect_target(df, candidates=None):\n",
    "    if candidates is None:\n",
    "        candidates = [\"label\", \"target\", \"is_phish\", \"phishing\", \"class\"]\n",
    "    for c in candidates:\n",
    "        if c in df.columns:\n",
    "            return c\n",
    "    for col in df.columns:\n",
    "        if df[col].nunique() == 2:\n",
    "            return col\n",
    "    return None\n",
    "\n",
    "def simpleurlfeatures(series):\n",
    "    s = series.fillna(\"\").astype(str)\n",
    "    out = pd.DataFrame()\n",
    "    out[\"url_len\"] = s.apply(len)\n",
    "    out[\"num_dots\"] = s.apply(lambda x: x.count(\".\"))\n",
    "    out[\"has_ip\"] = s.apply(lambda x: bool(re.search(r\"\\b\\d{1,3}(?:\\.\\d{1,3}){3}\\b\", x))).astype(int)\n",
    "    return out\n",
    "\n",
    "def preprocessdataframe(df, targetcol, textmaxfeatures=2000, svd_components=50):\n",
    "    df = df.dropna(subset=[targetcol]).resetindex(drop=True)\n",
    "    if df[target_col].dtype == \"object\":\n",
    "        le = LabelEncoder()\n",
    "        df[targetcol] = le.fittransform(df[target_col])\n",
    "    y = df[target_col]\n",
    "    X = df.drop(columns=[target_col])\n",
    "\n",
    "    url_cols = [c for c in X.columns if \"url\" in c.lower() or \"link\" in c.lower() or \"domain\" in c.lower()]\n",
    "    if url_cols:\n",
    "        X = pd.concat([X.resetindex(drop=True), simpleurlfeatures(X[urlcols[0]]).reset_index(drop=True)], axis=1)\n",
    "\n",
    "    textcols = X.selectdtypes(include=[\"object\"]).columns.tolist()\n",
    "    numcols = X.selectdtypes(include=[np.number]).columns.tolist()\n",
    "\n",
    "    transformers = []\n",
    "    if num_cols:\n",
    "        num_pipeline = Pipeline([(\"imputer\", SimpleImputer(strategy=\"median\")), (\"scaler\", StandardScaler())])\n",
    "        transformers.append((\"num\", numpipeline, numcols))\n",
    "\n",
    "    for col in text_cols:\n",
    "        text_pipeline = Pipeline([\n",
    "            (\"tfidf\", TfidfVectorizer(maxfeatures=textmaxfeatures, ngramrange=(1,2))),\n",
    "            (\"svd\", TruncatedSVD(ncomponents=min(svdcomponents, max(1, textmaxfeatures//20))))\n",
    "        ])\n",
    "        transformers.append((f\"text{col}\", textpipeline, col))\n",
    "\n",
    "    if not transformers:\n",
    "        return X.fillna(0).values, y.values\n",
    "\n",
    "    pre = ColumnTransformer(transformers, remainder=\"drop\", sparse_threshold=0)\n",
    "    Xtrans = pre.fittransform(X)\n",
    "    return X_trans, y.values\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Load dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "if not os.path.exists(DATA_PATH):\n",
    "    raise FileNotFoundError(f\"Dataset not found at {DATA_PATH}\")\n",
    "df = pd.readcsv(DATAPATH)\n",
    "print(\"Shape:\", df.shape)\n",
    "print(\"Columns:\", df.columns.tolist())\n",
    "df.head()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Detect target and preprocess"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "targetcol = detecttarget(df)\n",
    "if target_col is None:\n",
    "    raise ValueError(\"Could not detect target column. Set target_col manually.\")\n",
    "print(\"Using target:\", target_col)\n",
    "\n",
    "X, y = preprocessdataframe(df, targetcol, textmaxfeatures=2000, svd_components=50)\n",
    "print(\"Feature matrix shape:\", X.shape)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Train test split and imbalance handling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "Xtrain, Xtest, ytrain, ytest = traintestsplit(X, y, testsize=TESTSIZE, randomstate=RANDOMSTATE, stratify=y)\n",
    "if np.bincount(ytrain).min() / len(ytrain) < 0.2:\n",
    "    sm = SMOTE(randomstate=RANDOMSTATE)\n",
    "    Xtrain, ytrain = sm.fitresample(Xtrain, y_train)\n",
    "    print(\"Applied SMOTE. New class counts:\", np.bincount(y_train))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Train XGBoost baseline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "xgbclf = xgb.XGBClassifier(uselabelencoder=False, evalmetric=\"logloss\", randomstate=RANDOMSTATE, n_jobs=-1)\n",
    "xgbclf.fit(Xtrain, y_train)\n",
    "ypredxgb = xgbclf.predict(Xtest)\n",
    "print(\"XGBoost Accuracy:\", accuracyscore(ytest, ypredxgb))\n",
    "print(classificationreport(ytest, ypredxgb))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Train LightGBM baseline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "lgbclf = lgb.LGBMClassifier(randomstate=RANDOMSTATE, njobs=-1)\n",
    "lgbclf.fit(Xtrain, y_train)\n",
    "ypredlgb = lgbclf.predict(Xtest)\n",
    "print(\"LightGBM Accuracy:\", accuracyscore(ytest, ypredlgb))\n",
    "print(classificationreport(ytest, ypredlgb))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Quick EDA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(6,4))\n",
    "sns.countplot(x=y)\n",
    "plt.title(\"Target distribution\")\n",
    "plt.show()\n",
    "\n",
    "textcols = df.selectdtypes(include=[\"object\"]).columns.tolist()\n",
    "if text_cols:\n",
    "    sampletext = \" \".join(df[textcols[0]].dropna().astype(str).values[:10000])\n",
    "    wc = WordCloud(width=800, height=400, backgroundcolor=\"white\").generate(sampletext)\n",
    "    plt.figure(figsize=(12,6))\n",
    "    plt.imshow(wc, interpolation=\"bilinear\")\n",
    "    plt.axis(\"off\")\n",
    "    plt.show()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.x"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
`

---

.github/workflows/python-tests.yml
`yaml
name: Python tests

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          # Optional extras for integration tests; install only if needed
          if [ "${{ env.RUNSLOWTESTS }}" = "true" ]; then
            pip install xgboost lightgbm imbalanced-learn tldextract wordcloud matplotlib seaborn
          fi

      - name: Run pytest (fast tests)
        run: |
          pytest -q

      - name: Run slow integration tests (optional)
        if: env.RUNSLOWTESTS == 'true'
        run: |
          pytest -q -m slow
`

---

requirements.txt
`
pandas
numpy
scikit-learn
pytest
`

Optional extras (install only when needed)
`
xgboost
lightgbm
imbalanced-learn
tldextract
wordcloud
matplotlib
seaborn
`

---

Quick setup checklist
- Save phish.py at project root.  
- Create tests/ and save testphish.py and testintegration_model.py inside it.  
- Save phishinganalysisnotebook.ipynb at project root.  
- Add .github/workflows/python-tests.yml to your repo for CI.  
- Add requirements.txt at project root and install with pip install -r requirements.txt.  
- Run tests locally with pytest -q. To run the slow integration test, install xgboost and run pytest -q -m slow.

If you want, I can now generate a compact README with usage instructions and commands to run locally and in CI.

🛟 Support

If you need enhancements, improvements, or more automation, extend the CLI or GitHub workflows.

 

🎉 Final Note

This README documents your complete autonomous ML + DevOps pipeline.
Your OASIS system is now capable of:

Training

Testing

Healing

Formatting

Releasing

Versioning

Closing

Commenting

Auto-merging

Nightly cleaning

all without human intervention.