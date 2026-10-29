from pathlib import Path
from typing import Iterable, List, Optional, TypedDict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DATA_PATH = Path(__file__).parent / "data" / "sample_data.csv"


class Prediction(TypedDict):
    sentiment: str
    confidence: float


class SentimentAnalyzer:
    def __init__(self, data_path: Path = DATA_PATH) -> None:
        try:
            data = pd.read_csv(data_path)
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"Training data file is empty: {data_path}") from exc

        missing_columns = {"text", "sentiment"} - set(data.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Training data must contain these columns: {missing}")

        data = data.dropna(subset=["text", "sentiment"]).copy()
        data["text"] = data["text"].astype(str).str.strip()
        data["sentiment"] = data["sentiment"].astype(str).str.strip().str.lower()
        data = data[(data["text"] != "") & (data["sentiment"] != "")]

        if data.empty:
            raise ValueError("Training data must contain at least one labeled text.")
        if data["sentiment"].nunique() < 2:
            raise ValueError("Training data must contain at least two sentiment classes.")

        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, strip_accents="unicode")),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        self.model.fit(data["text"].astype(str), data["sentiment"].astype(str))

    @staticmethod
    def _validate_text(text: str, index: Optional[int] = None) -> str:
        if not isinstance(text, str):
            location = f" at index {index}" if index is not None else ""
            raise TypeError(f"Text{location} must be a string.")

        normalized_text = text.strip()
        if not normalized_text:
            location = f" at index {index}" if index is not None else ""
            raise ValueError(f"Text{location} must not be empty.")
        return normalized_text

    def predict(self, text: str) -> Prediction:
        normalized_text = self._validate_text(text)
        probabilities = self.model.predict_proba([normalized_text])[0]
        classes = self.model.classes_
        index = probabilities.argmax()
        return {
            "sentiment": str(classes[index]).title(),
            "confidence": float(probabilities[index]),
        }

    def predict_many(self, texts: Iterable[str]) -> List[Prediction]:
        validated_texts = [
            self._validate_text(text, index) for index, text in enumerate(texts)
        ]
        if not validated_texts:
            return []

        probabilities = self.model.predict_proba(validated_texts)
        classes = self.model.classes_
        results = []
        for row in probabilities:
            index = row.argmax()
            results.append({
                "sentiment": str(classes[index]).title(),
                "confidence": float(row[index]),
            })
        return results
