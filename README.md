# Sentiment Analysis of Product Reviews 🐾

## 📜 Project Overview
This project focuses on performing **sentiment analysis** on product reviews from **Amazon**, specifically for dog-related products. By analyzing textual reviews, the model can classify sentiments into **positive**, **neutral**, or **negative**, providing insights into customer satisfaction.

The project is designed as a **portfolio piece** to showcase data science skills, including:
- Natural Language Processing (NLP)
- Machine learning model development and optimization
- Data visualization and storytelling
- Code modularity and best practices

---

## 🛠️ Features
- **Preprocessing pipeline**: Clean and tokenize text data effectively.
- **Embeddings**: Leverages **FastText** for creating word embeddings.
- **Modeling**: Implements Logistic Regression and Support Vector Machines (SVM).
- **Evaluation**: Provides detailed metrics (Accuracy, F1-score, Precision, Recall) and visualizations like confusion matrices.
- **Exploratory Data Analysis (EDA)**: Uncovers patterns in the dataset using compelling visuals.

---

## 📂 Project Structure
```plaintext
sentiment_analysis/
│
├── data/
│   ├── raw/           # Original raw data
│   ├── processed/     # Processed and cleaned data
│
├── notebooks/         # Jupyter Notebooks for EDA and experiments
│
├── src/               # Source code
│   ├── preprocessing.py        # Text cleaning and tokenization
│   ├── feature_engineering.py  # Embedding generation
│   ├── modeling.py             # Model training and evaluation
│
├── reports/
│   ├── figures/       # Visualizations and graphs
│   ├── final_report.pdf # Final presentation/report
│
├── README.md          # Project documentation
├── requirements.txt   # Python dependencies
├── environment.yml    # Conda environment configuration
└── main.py            # Main pipeline to execute the project

🧪 Technologies Used
Programming: Python
Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, fasttext
Tools: Jupyter Notebook, Git, Visual Studio Code

📊 Results
Model Accuracy: Achieved an accuracy of 83.6% using Logistic Regression and SVM.
Insights:
Positive reviews often mention keywords like "quality" and "price."
Negative reviews frequently highlight "shipping" or "damage."

🤔 Future Work
Deploy the model via an API or interactive web app.
Experiment with deep learning models (e.g., LSTM, BERT) for improved performance.
Explore unbalanced dataset techniques or exclude neutral reviews.