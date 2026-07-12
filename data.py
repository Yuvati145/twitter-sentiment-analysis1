import streamlit as st
import pandas as pd
import re
import nltk
import altair as alt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ------------------ Page config ------------------
st.set_page_config(page_title="Twitter Sentiment Dashboard", layout="wide")

# ------------------ Safe stopwords loading ------------------
try:
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))
except Exception:
    try:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words("english"))
    except Exception:
        stop_words = {
            "a","an","the","and","or","is","are","was","were","in","on","at","to",
            "for","of","with","that","this","it","i","you","he","she","they","we",
            "be","been","have","has","had","do","does","did","not","but","by","from"
        }

# ------------------ Helpers ------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"#[A-Za-z0-9_]+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = text.lower().strip()
    words = [w for w in text.split() if w and w not in stop_words]
    return " ".join(words)

# ------------------ Load & preprocess data ------------------
@st.cache_data(show_spinner=False)
def load_and_prepare(path: str, sample_n: int = 10000):
    df = pd.read_csv(path)

    # Keep only required columns
    df = df[["target", "text"]].dropna().copy()

    # Clean the tweets
    df["clean_text"] = df["text"].astype(str).apply(clean_text)

    return df

CSV_PATH = "tweets1.csv"
try:
    df = load_and_prepare(CSV_PATH, sample_n=10000)
except FileNotFoundError:
    st.error(f"CSV file not found at path: {CSV_PATH}. Please check the file path.")
    st.stop()

# ------------------ Train model ------------------
@st.cache_resource
def train_model(data: pd.DataFrame):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data["clean_text"])
    y = data["target"].values
    model = LogisticRegression(max_iter=1000, solver="liblinear")
    model.fit(X, y)
    return vectorizer, model

vectorizer, model = train_model(df)

def predict_sentiment(text: str) -> str:
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    return "😊 Positive" if prediction == 1 else "😡 Negative"

# ------------------ Custom CSS ------------------
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg,#0F2027,#203A43,#2C5364); color: #FAFAFA; }
    .glow-title { font-size:32px; text-align:center; font-weight:700; margin-top:8px; }
    .percent-display { 
        text-align: center; 
        font-size: 30px; 
        font-weight: 700; 
        color: #BBDEFB; 
        margin-top: 20px;
        margin-bottom: 30px;
    }
    .pie-container {
        background: rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 10px;
        text-align: center;
        box-shadow: 0 0 20px rgba(66,165,245,0.3);
        position: relative;
        margin-top: -40px; /* move pie chart slightly up to align */
    }
    textarea { border-radius:10px !important; background:#1E293B !important; color:#E2E8F0 !important; }
    div.stButton > button:first-child { background:#1E88E5; color:white; font-weight:600; border-radius:8px; height:2.6em;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ Title ------------------
st.markdown('<div class="glow-title">💬 Twitter Sentiment Dashboard</div>', unsafe_allow_html=True)
st.write("")

# ------------------ User Input ------------------
st.subheader("🔍 Try it Yourself")
user_input = st.text_area("Enter a tweet:", "I absolutely love using AI for text analysis!")

if st.button("Analyze Sentiment"):
    result = predict_sentiment(user_input)
    if "Positive" in result:
        st.balloons()
    else:
        st.snow()
    st.success(f"✨ Sentiment Prediction: **{result}**")

st.markdown("---")

# ------------------ Sample Data ------------------
sample_df = df.sample(n=min(30, len(df)), random_state=42).reset_index(drop=True)
sample_df["Sentiment"] = sample_df["text"].apply(predict_sentiment)
df_live = sample_df[["text", "Sentiment"]].rename(columns={"text": "Tweet"})

# Percentages
pos_count = (df_live["Sentiment"] == "😊 Positive").sum()
neg_count = (df_live["Sentiment"] == "😡 Negative").sum()
total = len(df_live)
pos_percent = (pos_count / total) * 100 if total else 0.0
neg_percent = (neg_count / total) * 100 if total else 0.0

# ------------------ Centered Percent Display ------------------
st.markdown(
    f"""
    <div class="percent-display">
        😊 Positive: {pos_percent:.1f}%   😡 Negative: {neg_percent:.1f}%
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------ Charts ------------------
color_scale = alt.Scale(domain=["😊 Positive", "😡 Negative"], range=["#42A5F5", "#1565C0"])
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Sentiment Distribution")
    bar = (
        alt.Chart(df_live)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Sentiment:N", title="Sentiment", axis=alt.Axis(labelAngle=0)),  # horizontal labels
            y=alt.Y("count():Q", title="Count"),
            color=alt.Color("Sentiment:N", scale=color_scale, legend=None),
            tooltip=["Sentiment", alt.Tooltip("count():Q", title="Count")]
        )
        .properties(width=500, height=400)
    )
    st.altair_chart(bar, use_container_width=True)

with col2:
    st.subheader("🥧 Sentiment Breakdown")
    st.markdown('<div class="pie-container">', unsafe_allow_html=True)

    # Pie chart with inner text labels only
    pie_chart = (
        alt.Chart(df_live)
        .mark_arc(outerRadius=140, innerRadius=40)
        .encode(
            theta=alt.Theta("count():Q"),
            color=alt.Color("Sentiment:N", scale=color_scale, legend=None),
            tooltip=["Sentiment", alt.Tooltip("count():Q", title="Count")]
        )
        .properties(width=420, height=360)
    )

    text_labels = (
        alt.Chart(df_live)
        .mark_text(radius=90, size=16, color="white", fontWeight="bold")
        .encode(
            theta=alt.Theta("count():Q", stack=True),
            text=alt.Text("Sentiment:N")
        )
    )

    # Display pie chart + inner labels
    st.altair_chart(pie_chart + text_labels, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Data Table ------------------
st.markdown("### 📝 Recent Tweets Analyzed")
st.dataframe(df_live.reset_index(drop=True), height=420)
