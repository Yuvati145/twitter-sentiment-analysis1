# 💬 Twitter Sentiment Analysis Dashboard

A Streamlit web app that classifies tweets as **Positive 😊** or **Negative 😡** using a TF-IDF + Logistic Regression model trained on the fly, with an interactive dashboard to explore the results.

## Features

- **Live sentiment prediction** — type any tweet/text into the box and get an instant Positive/Negative prediction, with a fun balloons/snow animation on result.
- **Text cleaning pipeline** — strips URLs, @mentions, #hashtags, punctuation/numbers, and stopwords (via NLTK, with a built-in fallback list if the NLTK download is unavailable) before feeding text to the model.
- **TF-IDF + Logistic Regression** classifier (scikit-learn), trained directly from `tweets1.csv` on app startup and cached with `@st.cache_resource` so it only trains once per session.
- **Interactive visual dashboard** (Altair):
  - Bar chart of sentiment distribution
  - Donut/pie chart breakdown with inline labels
  - Positive/Negative percentage summary
  - Table of a random sample of analyzed tweets with predicted sentiment
- Custom dark-themed UI with gradient background and styled components.

## Tech Stack

| Category            | Tools / Libraries                     |
|----------------------|----------------------------------------|
| Language             | Python                                 |
| Web App / UI         | Streamlit                              |
| Data Handling        | Pandas, NumPy                          |
| Machine Learning      | scikit-learn (TF-IDF, Logistic Regression) |
| NLP / Text Cleaning  | NLTK (stopwords), Regex                |
| Visualization         | Altair, Matplotlib                     |

## Project Structure

```
twitter-sentiment-analysis1/
├── data.py             # Main Streamlit app: preprocessing, model training, dashboard UI
├── tweets1.csv          # Dataset used to train the model (target, text columns)
├── requirements.txt     # Python dependencies
└── .devcontainer/       # Dev container config (for GitHub Codespaces)
```

## Dataset

`tweets1.csv` contains two columns:

| Column   | Description                                  |
|----------|-----------------------------------------------|
| `target` | Sentiment label — `0` = Negative, `1` = Positive |
| `text`   | The raw tweet text                            |

## Getting Started

### Prerequisites
- Python 3.8+

### Installation

```bash
# Clone the repository
git clone https://github.com/Yuvati145/twitter-sentiment-analysis1.git
cd twitter-sentiment-analysis1

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run data.py
```

The app will open in your browser at `http://localhost:8501`.

## How It Works

1. **Load & clean data** — `tweets1.csv` is loaded and each tweet is cleaned (URLs, mentions, hashtags, and non-alphabetic characters removed; text lowercased and stopwords filtered out).
2. **Vectorize** — cleaned text is converted into numerical features using `TfidfVectorizer` (top 5,000 features).
3. **Train** — a `LogisticRegression` classifier is trained on the TF-IDF features against the `target` labels.
4. **Predict** — any new text entered in the app is cleaned the same way, vectorized, and classified as Positive or Negative.
5. **Visualize** — a random sample of tweets is scored and displayed as charts and a table so you can see the model's predictions at a glance.

## Future Improvements

- Support for a Neutral sentiment class (currently binary Positive/Negative)
- Persist the trained model (e.g. with `joblib`) instead of retraining on every app restart
- Swap in a pretrained transformer model (e.g. BERT) for improved accuracy
- Live tweet fetching via the Twitter/X API instead of a static CSV

## Author

**Yuvati Bhabal**
[GitHub](https://github.com/Yuvati145)

## License

This project is open source and available for educational use.
