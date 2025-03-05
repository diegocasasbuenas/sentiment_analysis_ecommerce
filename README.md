# Sentiment Analysis of Product Reviews 🐾

### 📜 Project Overview
This project focuses on performing **sentiment analysis** on product reviews from **Amazon**, specifically for pet products. By analyzing textual reviews, the model can classify sentiments into **positive** and **negative**, providing insights into customer satisfaction with sentiment analys. 

The data comes from a **16M-row dataset** available at [Hugging Face](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).

### Data Processing

A **large-scale dataset** was processed using the **RAPIDS** library, specifically **cuDF**, which accelerated the process and allowed us to refine the dataset to **300K well-processed rows** for training the sentiment analysis model.

## Sentiment Labeling

Since the dataset lacks sentiment labels, we assigned labels based on the **star rating**:

- Reviews with **5 and 4 stars** were labeled as **positive sentiment**.
- Reviews with **1 and 2 stars** were labeled as **negative sentiment**.
- **3-star reviews** were removed, as they introduced bias and reduced the model’s ability to predict accurately during testing.


---

### 🛠️ Features  
- **Dataset Balancing**: The original **1,000,000-row dataset** was expanded and balanced by **star rating**, resulting in a **300,000-row dataset**.  
- **Exploratory Data Analysis (EDA)**: Conducted an in-depth analysis to uncover key patterns in the data.  
- **Model Testing**: Evaluated multiple models to determine the most effective approach.  
- **Best Model Selection**: The best-performing model was a **transfer learning approach with RoBERTa**.  
- **Deployment**: Used **FastAPI** and **Streamlit** to deploy and test the final model with real **Amazon reviews**. 🚀


---

### 📂 Project Structure
```
SENTIMENT_ANALYSIS/
├── api/
│   ├── sentiment_api.py
├── data/
│   ├── processed/
│   ├── test_data2.parquet
│   ├── train_data2.parquet
│   ├── raw/
│   │   ├── download_raw.py
├── models/
│   ├── download_models.py
├── notebooks/
│   ├── EDA_amazon_reviews.ipynb
│   ├── model_testing.ipynb
├── reports/
│   ├── figures/
├── src/
│   ├── balance_classes.py
│   ├── fasttext_model.py
│   ├── preprocess_apr.py
│   ├── roberta_model.py
├── streamlit_int/
├── .gitignore
├── README.md
├── requirements.txt
```




🧪 Technologies Used
Programming: Python

Libraries:
Data Manipulation and Processing: pandas, numpy, collections.Counter, joblib, os, Pathlib.
Text Processing and NLP: spaCy, nltk, re, BeautifulSoup, WordCloud, STOPWORDS.
Machine Learning Models: scikit-learn (for data splitting, scaling, dimensionality reduction, and classification), xgboost, fasttext, torch, transformers.
Optimization and Training: torch.nn, torch.optim, Trainer, TrainingArguments.
Model Evaluation: accuracy_score, classification_report.
Data Visualization: matplotlib.pyplot, seaborn, plotly.express, wordcloud, umap.
Others: logging, tqdm for progress bars.
Tools: Jupyter Notebook, Git, Visual Studio Code

📊 Results
Model Accuracy: Achieved an accuracy of 90.0% using RoBERTa model.
Insights:
The training of the sentiment analysis model using transfer learning yielded good results on the Amazon Reviews (Pet Products) dataset, as it was trained with positive and negative reviews. This means that 3-star reviews were excluded, as they were considered a source of bias.

During the development of this project, different models and text preprocessing techniques were tested, and it was found that RoBERTa (a Transformer model) is the best option for sentiment analysis, significantly outperforming traditional neural network models. This makes sense, considering that RoBERTa is a pretrained model specifically designed for text classification, making it more effective.

On the other hand, text preprocessing plays a fundamental role in the success of the model. After experimenting with different approaches, the most effective method was found to be the use of regular expressions, specifically [^a-zA-Z0-9\s'], which removes all characters except letters, numbers, spaces, and apostrophes. This should be combined with removing extra spaces and converting all text to lowercase. A simple yet effective cleaning process will contribute more to the model's performance than any other step.

The use of stopword lists is another key factor that helps preserve important word relationships. Even more crucial is ensuring that negations are not removed, as they can be 
decisive in determining whether a review expresses a positive or negative sentiment.

🤔 Deployment:
The pre-trained model "sentiment_transformer_model", located in the models folder, has been deployed using FastAPI and Streamlit for local testing.

You can find the corresponding code in the api and streamlit_int folders. You can run the appropriate scripts and test the model with real Amazon reviews focused on dog products. 
