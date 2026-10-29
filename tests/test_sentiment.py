import csv
import tempfile
import unittest
from pathlib import Path

from sentiment import DATA_PATH, SentimentAnalyzer


class SentimentAnalyzerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.analyzer = SentimentAnalyzer()

    def test_predictions_have_expected_shape_and_valid_confidence(self) -> None:
        result = self.analyzer.predict("I really enjoyed this product!")

        self.assertIn(result["sentiment"], {"Positive", "Negative", "Neutral"})
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_batch_predictions_match_individual_predictions(self) -> None:
        texts = ["Wonderful experience", "The event starts at noon"]

        batch_results = self.analyzer.predict_many(texts)
        individual_results = [self.analyzer.predict(text) for text in texts]

        self.assertEqual(batch_results, individual_results)

    def test_empty_batch_returns_no_predictions(self) -> None:
        self.assertEqual(self.analyzer.predict_many([]), [])

    def test_blank_text_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.analyzer.predict("   ")

    def test_batch_reports_index_of_blank_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "index 1"):
            self.analyzer.predict_many(["valid post", "   "])

    def test_training_data_requires_text_and_sentiment_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "invalid.csv"
            data_path.write_text("message,label\nhello,positive\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "sentiment, text"):
                SentimentAnalyzer(data_path)

    def test_training_data_requires_at_least_two_classes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "single_class.csv"
            with data_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["text", "sentiment"])
                writer.writeheader()
                writer.writerows([
                    {"text": "first post", "sentiment": "positive"},
                    {"text": "second post", "sentiment": "positive"},
                ])

            with self.assertRaisesRegex(ValueError, "at least two"):
                SentimentAnalyzer(data_path)

    def test_sample_data_path_is_the_default_training_data(self) -> None:
        self.assertTrue(DATA_PATH.is_file())


if __name__ == "__main__":
    unittest.main()
