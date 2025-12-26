# Financial News Sentiment Analysis & Knowledge Graph

A comprehensive system that combines financial news sentiment analysis with graph-based information retrieval, providing an interactive platform for exploring relationships between entities and related news records.

---

## 📋 Problem Statement

Financial news contains valuable insights that can influence investment and business decisions, but manually extracting sentiment and understanding relationships between entities is time-consuming. This project automates sentiment classification and builds an interactive Knowledge Graph to efficiently explore entities and their related news.

---

## ✨ Features

### 📊 Exploratory Data Analysis & Sentiment Classification
- Performed comprehensive EDA on **5,842 financial news records**
- Classified news sentiment as **positive**, **negative**, or **neutral**
- Utilized **NLTK** for text preprocessing and **TF-IDF vectorization** for feature extraction
- Trained and evaluated multiple machine learning models:
  - Logistic Regression
  - Linear SVC
  - Random Forest
  - Naive Bayes
- Implemented **Stratified K-Fold cross-validation** for robust evaluation
- **Best Performance**: Logistic Regression achieved **70% accuracy** and **0.71 F1-score**

### 🧠 Deep Learning Sentiment Classifiers
- Built advanced neural network architectures:
  - **LSTM** (Long Short-Term Memory)
  - **Bi-LSTM** (Bidirectional LSTM)
  - **GRU** (Gated Recurrent Unit)
- Leveraged **Word2Vec embeddings** using gensim for semantic word representations
- Incorporated **Dropout layers** to prevent overfitting
- Applied **Keras callbacks** for training optimization
- **Best Performance**: Bi-LSTM achieved **72% accuracy**

### 🕸️ Named Entity Recognition (NER) & Knowledge Graph
- Extracted financial entities (companies, people, organizations) using **spaCy**
- Constructed a comprehensive **Knowledge Graph** in a Dockerized **Neo4j** container
- Interactive **Streamlit** frontend enables intuitive entity querying
- For each query:
  - Relevant subgraph is dynamically generated using **Cypher** queries
  - Corresponding news records are retrieved and displayed
- Visual graph exploration using **PyVis**

---

## 🛠️ Tech Stack

### Machine Learning & NLP
- **Python** - Core programming language
- **NLTK** - Natural language preprocessing
- **TextBlob** - Sentiment analysis
- **spaCy** - Named Entity Recognition
- **Gensim** - Word2Vec embeddings
- **scikit-learn** - ML models and evaluation

### Deep Learning
- **TensorFlow/Keras** - Neural network frameworks
- **LSTM, Bi-LSTM, GRU** - Recurrent neural architectures

### Graph Database
- **Neo4j** - Graph database management
- **Cypher** - Graph query language
- **Docker** - Neo4j containerization

### Visualization & Frontend
- **Streamlit** - Interactive web interface
- **PyVis** - Graph visualization
- **Matplotlib/Seaborn** - Data visualization

### Data Processing
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations

---

## 📁 Project Structure
