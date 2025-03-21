# import streamlit as st
# import pickle
# import re
# from nltk.corpus import stopwords
# import nltk
# nltk.download('stopwords')
# from nltk.stem.porter import PorterStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer
# port_stem = PorterStemmer()
# vectorization = TfidfVectorizer()

# vector_form = pickle.load(open('vector.pkl', 'rb'))
# load_model = pickle.load(open('model.pkl', 'rb'))

# def stemming(content):
#     con=re.sub('[^a-zA-Z]', ' ', content)
#     con=con.lower()
#     con=con.split()
#     con=[port_stem.stem(word) for word in con if not word in stopwords.words('english')]
#     con=' '.join(con)
#     return con

# def fake_news(news):
#     news=stemming(news)
#     input_data=[news]
#     vector_form1=vector_form.transform(input_data)
#     prediction = load_model.predict(vector_form1)
#     return prediction



# if __name__ == '__main__':
#     st.title('Fake News Classification app ')
#     st.subheader("Input the News content below")
#     sentence = st.text_area("Enter your news content here", "",height=200)
#     predict_btt = st.button("predict")
#     if predict_btt:
#         prediction_class=fake_news(sentence)
#         print(prediction_class)
#         if prediction_class == [0]:
#             st.success('Reliable')
#         if prediction_class == [1]:
#             st.warning('Unreliable')






import pickle
import re
import nltk
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords
nltk.download('stopwords')

# Load trained model
vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

# NLP Preprocessing
port_stem = PorterStemmer()
def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content).lower().split()
    content = ' '.join([port_stem.stem(word) for word in content if word not in stopwords.words('english')])
    return content

# FastAPI Setup
app = FastAPI()

# CORS (Allows frontend requests)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Input Schema
class NewsRequest(BaseModel):
    news: str

@app.post("/predict")
def predict_fake_news(request: NewsRequest):
    processed_news = stemming(request.news)
    vector_input = vector_form.transform([processed_news])
    prediction = load_model.predict(vector_input)
    return {"prediction": int(prediction[0])}














