import io
import pandas as pd
import streamlit as st
import plotly.express as px
from sentiment import SentimentAnalyzer

st.set_page_config(
    page_title="Social Media Sentiment Analyzer",
    page_icon="💬",
    layout="wide",
)

@st.cache_resource
def load_analyzer():
    return SentimentAnalyzer()

analyzer = load_analyzer()

st.title("💬 Social Media Sentiment Analyzer")
st.caption("A local NLP project using TF-IDF and Logistic Regression.")

tab_single, tab_batch, tab_about = st.tabs(["Analyze text", "Analyze CSV", "About"])

with tab_single:
    st.subheader("Analyze a post")
    text = st.text_area(
        "Enter a social media post",
        placeholder="Example: I really enjoyed this product!",
        height=130,
    )
    if st.button("Analyze sentiment", type="primary"):
        if not text.strip():
            st.warning("Enter some text first.")
        else:
            result = analyzer.predict(text)
            col1, col2, col3 = st.columns(3)
            col1.metric("Sentiment", result["sentiment"])
            col2.metric("Confidence", f'{result["confidence"]:.1%}')
            col3.metric("Words", len(text.split()))
            st.progress(result["confidence"])
            st.caption("Confidence is the model's estimated probability, not a guarantee.")

with tab_batch:
    st.subheader("Analyze a CSV file")
    st.write("Upload a CSV containing a column named `text`.")
    uploaded = st.file_uploader("Choose CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Could not read the CSV: {exc}")
        else:
            if "text" not in df.columns:
                st.error("CSV must contain a column named `text`.")
            elif df.empty:
                st.warning("The CSV contains no rows to analyze.")
            elif st.button("Analyze uploaded posts"):
                texts = df["text"].fillna("").astype(str).str.strip()
                has_text = texts.ne("")
                if not has_text.any():
                    st.warning("The CSV contains no non-empty text values to analyze.")
                else:
                    results = analyzer.predict_many(texts[has_text].tolist())
                    output = df.copy()
                    output["sentiment"] = "Not analyzed"
                    output["confidence"] = float("nan")
                    output.loc[has_text, "sentiment"] = [r["sentiment"] for r in results]
                    output.loc[has_text, "confidence"] = [r["confidence"] for r in results]
                    skipped_count = int((~has_text).sum())
                    if skipped_count:
                        st.info(f"Skipped {skipped_count} row(s) with empty text.")

                    st.dataframe(output, use_container_width=True)
                    counts = (
                        output.loc[has_text, "sentiment"]
                        .value_counts()
                        .rename_axis("sentiment")
                        .reset_index(name="count")
                    )
                    fig = px.bar(counts, x="sentiment", y="count", title="Sentiment distribution")
                    st.plotly_chart(fig, use_container_width=True)
                    csv_bytes = output.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download analyzed CSV",
                        data=csv_bytes,
                        file_name="sentiment_results.csv",
                        mime="text/csv",
                    )

with tab_about:
    st.subheader("How it works")
    st.markdown("""
    - **TF-IDF** converts text into numerical features.
    - **Logistic Regression** predicts a sentiment class.
    - The model is trained from the small example dataset included with this project.
    - This is a learning/demo model. Its accuracy depends on the training data and it may misunderstand sarcasm, slang, or context.
    - Empty rows in a batch upload are skipped and appear as `Not analyzed`.
    """)
    st.info("No posts are sent to an external API. Analysis runs locally.")
