import streamlit as st 
import pickle 
import os 
import re 
import nltk 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer   
import pandas as pd 

model_path = os.path.join('Model','model.pkl')
vectorizer_path = os.path.join('Model','vectorizer.pkl') 

st.set_page_config(page_title='Fake & Real Detector')

st.title("Fake or Real News Detector")

text = st.text_area("Enter News Text :")

try :
    model = pickle.load(open(model_path, 'rb')) 
    vectorizer = pickle.load(open(vectorizer_path, 'rb')) 
 
except Exception as e :
    st.error(str(e))


stemmer=PorterStemmer()

def remove_spec(x):
    text = re.sub('[^a-zA-Z]',' ',x) 
    text=text.lower() 
    text=text.split()  

    text=[stemmer.stem(word) for word in text if not word in stopwords.words('english')]
    text=' '.join(text) 
    return text

if len(text) != 0:
    df = pd.DataFrame({'text':text}, index=[0]) 
    df['text'] = df['text'].apply(remove_spec) 
    new_data = vectorizer.transform(df['text'])   
     

if st.button('Predict') :
    prediction = model.predict(new_data) 
    if prediction[0] == 1 :
        st.error('News is Fake ')
    else :
        st.success('News is Real ')  
