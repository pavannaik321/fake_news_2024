# Fake News Detection System

## 📌 Introduction
The **Fake News Detection System** is a **Streamlit-based web application** that classifies news articles as **Reliable** or **Unreliable** using a trained **Machine Learning model**.

## 🛠️ Installation & Setup
Follow the steps below to set up and run the project.

### 1️⃣ **Create a Conda Environment**
```bash
conda create --name fake_news_env2 python=3.10
```
This command creates a new environment named **fake_news_env2** with Python 3.10.

### 2️⃣ **Activate the Environment**
```bash
conda activate fake_news_env2
```
This activates the **fake_news_env2** environment.

### 3️⃣ **Install Dependencies**
Run the following command to install all required dependencies:
```bash
pip install streamlit scikit-learn==1.6.1 nltk pickle5
```

### 4️⃣ **Download NLTK Stopwords** (Optional)
If NLTK stopwords are missing, download them using:
```python
import nltk
nltk.download('stopwords')
```

### 5️⃣ **Run the Streamlit App**
To start the Fake News Detection web app, run:
```bash
streamlit run app.py
```

## 📌 Usage
1. Open the **Streamlit UI** in your browser (default: `http://localhost:8501`).
2. Enter a news article or text in the provided input field.
3. Click **"Predict"** to classify the news as **Reliable** or **Unreliable**.

## 🚀 Features
- **ML Model**: Uses a trained **DecisionTreeClassifier** for text classification.
- **TF-IDF Vectorization**: Converts text into numerical format before prediction.
- **Streamlit UI**: Interactive web-based user interface.
- **Conda & Python Virtual Environment Support**.

## 🔧 Troubleshooting
### *Check Installed Packages in Conda Environment:*
```bash
conda list
```

### *Deactivate the Environment:*
```bash
conda deactivate
```

### *Reactivate the Environment:*
```bash
conda activate fake_news_env2
```

## 📜 License
This project is for educational purposes. Modify and extend it as needed!

