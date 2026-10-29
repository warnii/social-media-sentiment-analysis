# Social Media Sentiment Analysis Tool

A beginner-friendly NLP and machine-learning project that classifies social media text as **Positive**, **Negative**, or **Neutral**.

## Features
- Analyze a single post
- Batch-analyze CSV files with a `text` column
- View sentiment distribution
- Download analyzed results as CSV
- Local processing (no external sentiment API)

## Tech stack
Python, Streamlit, Pandas, scikit-learn, Plotly

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`).

If PowerShell blocks activation, you can run Streamlit without activating the environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## CSV format
For batch analysis, upload a CSV with a column named `text`. Example:

```csv
text
"I love this update!"
"This is disappointing."
"The event starts at 6 PM."
```

## Model
The included example dataset is intentionally small and suitable only for demonstrating the workflow. It uses a TF-IDF vectorizer and Logistic Regression. Do not treat predictions as ground truth. Improve the dataset and evaluate the model before making real-world claims.

Predictions include the model's highest class probability as a confidence score. This is not a calibrated measure of correctness. Blank text is rejected for single-post analysis and skipped in CSV uploads; skipped rows are labeled `Not analyzed` and have no confidence score.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

## Project structure

```text
social-media-sentiment-analysis/
├── app.py
├── sentiment.py
├── tests/
│   └── test_sentiment.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── sample_data.csv
```

## Future improvements
- Add a larger, properly licensed labeled dataset
- Add train/test split and evaluation metrics
- Support language selection
- Add model persistence and retraining
- Add sentiment analysis for cleaned, consented datasets
