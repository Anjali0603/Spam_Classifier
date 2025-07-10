# 📧 Email Spam Classifier 🚫✉️
A machine learning project to classify emails as **Spam** or **Not Spam** using Natural Language Processing and supervised learning models.

---

## 🧠 Problem Statement
With the rise in unwanted and malicious emails, spam detection has become essential. This project builds a classification model that can predict whether an email is spam or not based on its content using the **Spambase dataset** from the UCI Machine Learning Repository.

---

## 📊 Dataset
- **Name:** Spambase Dataset  
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/spambase)  
- **Format:** CSV (`spambase.csv`)  
- **Features:** 57 continuous features (like frequency of specific words/symbols)  
- **Target:** `1 = spam`, `0 = not spam`

Detailed feature descriptions can be found in the `data/spambase.names` file.

---

## 🛠️ Tech Stack
- Python 3.x  
- Pandas, NumPy  
- Scikit-learn  
- NLTK  
- Streamlit  
- Matplotlib/Seaborn (for optional EDA)

---

## 🚀 How to Run

### ⚙️ 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/email-spam-classifier.git
cd email-spam-classifier
