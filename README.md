## Disease–Symptom Co-Occurrence Mining (Data Warehousing & Mining Project)

This is a small end-to-end project for a **Data Warehousing and Data Mining** course.  
The goal is to analyze a dataset of *patients, diseases, and symptoms* and mine:

- **Co-occurring symptoms** for a given disease
- **Association rules** between symptoms and diseases
- Simple **statistics and visualizations** for exploration

---

### 1. Project Overview

- **Backend / Mining**: Python (pandas, numpy, mlxtend or custom Apriori)
- **Front-end / UI**: Streamlit web app
- **Data**: CSV file with columns like:
  - `patient_id`
  - `disease`
  - `symptom`

Each row represents that a patient diagnosed with `disease` shows a particular `symptom`.
The app groups rows into transactions per patient (or per disease) to perform co-occurrence mining.

---

### Logical Data Warehouse + OLAP (Included)

This project also includes a **small logical data warehouse layer** (implemented in-memory using pandas):

- **Fact table**: `patient_id, disease, symptom` (each row is a fact/event)
- **Dimensions**: Patient, Disease, Symptom (unique lists for exploration)

In the app, the **OLAP Explorer** tab demonstrates:

- **Slice**: fix one dimension value (e.g., disease = Flu)
- **Dice**: filter multiple values across dimensions (e.g., diseases in {Flu, COVID-19} and symptoms in {Fever, Cough})
- **Roll-up**: fewer dimensions → less detail (e.g., from disease+symptom → disease only)
- **Drill-down**: add a dimension → more detail (e.g., from disease only → disease+symptom)
- **Pivot view**: 2D cube view (e.g., Disease × Symptom with counts)

---

### 2. Folder Structure

```text
Data Warehousing & Mining Project/
├── app.py                      # Main Streamlit app
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/
│   └── sample_disease_symptom.csv
└── src/
    ├── __init__.py
    ├── config.py               # Basic configuration (paths, thresholds)
    ├── data_loader.py          # Load and validate CSV data
    ├── preprocessing.py        # Transform raw rows into transactions
    ├── mining.py               # Co-occurrence + association rule mining
    ├── olap.py                 # Logical warehouse + OLAP operations
    └── visualization.py        # Helper functions for charts/tables
```

---

### 3. Setup & Installation

**Prerequisites**

- Python 3.9+ installed and added to PATH
- `pip` available

**Create virtual environment (recommended)**

```bash

python -m venv venv
venv\Scripts\activate
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

---

### 4. Running the App

From the project root (`Data Warehousing & Mining Project`):

```bash
venv\Scripts\activate   # if not already activated
streamlit run app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`) in your browser.

In the UI you can:

- Upload your own disease–symptom CSV or use the **sample dataset**
- Configure **minimum support** and **minimum confidence**
- View:
  - Frequent **symptom itemsets**
  - **Association rules** (symptom → disease, symptom set → symptom set)
  - Co-occurrence **heatmap** and summary statistics

---

### 5. Data Format

Your CSV should have at least these columns:

- `patient_id` (string or integer)
- `disease` (string)
- `symptom` (string)

Example (wide format, each row = patient, columns are symptoms) is **also** supported via preprocessing
in the app (you can specify which columns are symptom flags).

---

### 6. Implementation Notes (for Report / Viva)

- **Data Warehousing Concepts**
  - We treat the dataset as a small **star schema**:
    - Fact: `patient_symptom_fact(patient_id, disease, symptom)`
    - Dimensions: `disease_dim`, `symptom_dim`, `patient_dim` (logical, implemented via group-bys)
- **Data Mining Technique**
  - Uses **frequent itemset mining (Apriori)** on symptom sets
  - Extracts **association rules** with support, confidence, and lift
  - Allows tuning of thresholds via UI
- **Limitations**
  - Works on small–medium CSVs that fit into memory
  - Co-occurrence is **statistical correlation**, not causation

You can easily extend the project by:

- Adding more visualizations
- Adding demographic attributes (age, gender) as additional dimensions
- Trying different mining algorithms (FP-Growth, etc.)

---

### 7. How to Present This Project

In a demo or presentation, you can:

1. Briefly explain the **problem**: doctors want to see common symptom patterns per disease.
2. Show the **data model** and how patients/diseases/symptoms are stored.
3. Demonstrate the **UI**:
   - Load sample data
   - Change min support/confidence
   - Show how frequent sets and rules change
4. Discuss **findings** from the sample data (e.g., “Fever and Cough frequently co-occur for Disease X”).
5. Mention **future work** and limitations.

