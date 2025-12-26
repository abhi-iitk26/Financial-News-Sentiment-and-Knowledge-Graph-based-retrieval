Financial News Sentiment Analysis & Knowledge GraphA comprehensive system that combines financial news sentiment analysis with graph-based information retrieval, providing an interactive platform for exploring relationships between entities and related news records.📋 Problem StatementFinancial news contains valuable insights that can influence investment and business decisions, but manually extracting sentiment and understanding relationships between entities is time-consuming. This project automates sentiment classification and builds an interactive Knowledge Graph to efficiently explore entities and their related news.✨ Features📊 Exploratory Data Analysis & Sentiment Classification

Performed comprehensive EDA on 5,842 financial news records
Classified news sentiment as positive, negative, or neutral
Utilized NLTK for text preprocessing and TF-IDF vectorization for feature extraction
Trained and evaluated multiple machine learning models:

Logistic Regression
Linear SVC
Random Forest
Naive Bayes


Implemented Stratified K-Fold cross-validation for robust evaluation
Best Performance: Logistic Regression achieved 70% accuracy and 0.71 F1-score
🧠 Deep Learning Sentiment Classifiers

Built advanced neural network architectures:

LSTM (Long Short-Term Memory)
Bi-LSTM (Bidirectional LSTM)
GRU (Gated Recurrent Unit)


Leveraged Word2Vec embeddings using gensim for semantic word representations
Incorporated Dropout layers to prevent overfitting
Applied Keras callbacks for training optimization
Best Performance: Bi-LSTM achieved 72% accuracy
🕸️ Named Entity Recognition (NER) & Knowledge Graph

Extracted financial entities (companies, people, organizations) using spaCy
Constructed a comprehensive Knowledge Graph in a Dockerized Neo4j container
Interactive Streamlit frontend enables intuitive entity querying
For each query:

Relevant subgraph is dynamically generated using Cypher queries
Corresponding news records are retrieved and displayed


Visual graph exploration using PyVis
🛠️ Tech StackMachine Learning & NLP

Python - Core programming language
NLTK - Natural language preprocessing
TextBlob - Sentiment analysis
spaCy - Named Entity Recognition
Gensim - Word2Vec embeddings
scikit-learn - ML models and evaluation
Deep Learning

TensorFlow/Keras - Neural network frameworks
LSTM, Bi-LSTM, GRU - Recurrent neural architectures
Graph Database

Neo4j - Graph database management
Cypher - Graph query language
Docker - Neo4j containerization
Visualization & Frontend

Streamlit - Interactive web interface
PyVis - Graph visualization
Matplotlib/Seaborn - Data visualization
Data Processing

Pandas - Data manipulation and analysis
NumPy - Numerical computations
📁 Project Structurefinancial-news-sentiment-kg/
│
├── data/
│   ├── raw/                          # Raw financial news dataset
│   ├── processed/                    # Cleaned and preprocessed data
│   └── embeddings/                   # Word2Vec models
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory Data Analysis
│   ├── 02_sentiment_ml.ipynb        # Traditional ML models
│   ├── 03_sentiment_dl.ipynb        # Deep Learning models
│   └── 04_ner_kg_construction.ipynb # Entity extraction & graph building
│
├── src/
│   ├── preprocessing/
│   │   ├── text_cleaner.py          # Text cleaning utilities
│   │   └── feature_extraction.py    # TF-IDF and embedding utilities
│   │
│   ├── models/
│   │   ├── ml_classifiers.py        # ML sentiment classifiers
│   │   ├── dl_classifiers.py        # Deep learning models
│   │   └── model_utils.py           # Training and evaluation utilities
│   │
│   ├── ner/
│   │   └── entity_extractor.py      # Named Entity Recognition
│   │
│   ├── knowledge_graph/
│   │   ├── graph_builder.py         # Neo4j graph construction
│   │   ├── cypher_queries.py        # Predefined Cypher queries
│   │   └── graph_utils.py           # Graph utility functions
│   │
│   └── visualization/
│       └── graph_visualizer.py      # PyVis visualization
│
├── app/
│   ├── streamlit_app.py             # Main Streamlit application
│   ├── pages/
│   │   ├── sentiment_analysis.py    # Sentiment analysis interface
│   │   └── knowledge_graph.py       # Knowledge graph explorer
│   └── utils/
│       └── app_helpers.py           # Helper functions for app
│
├── docker/
│   ├── docker-compose.yml           # Docker compose for Neo4j
│   └── Dockerfile                   # Custom Docker configurations
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_graph.py
│
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
├── .gitignore
└── LICENSE
