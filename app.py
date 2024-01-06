from flask import Flask, render_template,request
import joblib

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
app = Flask(__name__)

# Load the pickled TF-IDF vectorizer
loaded_tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Load the model from the previous file
loaded_model = joblib.load('SVC_classifier.pkl')


def preprocessing(text):
    #tokenize the text
    tokenize_data = word_tokenize(text)
    #replace special characters using regex
    text = re.sub('[^a-zA-Z]', ' ', ' '.join(tokenize_data))
    #lower the text
    text=text.lower()
    #split the text
    text=text.split()
    # Lemmatize the data using stopwords
    stopwords_set = set(stopwords.words('english'))
    text = [WordNetLemmatizer().lemmatize(word) for word in text if word not in stopwords_set]
     # Join lemmatized words back into the string
    cleaned_text = ' '.join(text)
    # Return cleaned text
    return cleaned_text

def predict_data(user_data,model):
    #clean the text
    clean_text=preprocessing(user_data)
    #vectorize the cleaned text
    vectorized_input=loaded_tfidf_vectorizer.transform([clean_text])


    #give the vectorized text to the model
    prediction=model.predict(vectorized_input)
    #return the result
    return prediction[0]

nltk.download('punkt')
nltk.download('stopwords')

@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='POST':
        user_input=request.form.get('user_text')
        result=predict_data(user_input,loaded_model)
        print("Result:", result)
        return render_template('index.html',Result=result)
    return render_template('index.html')

    



if __name__ == '__main__':
    app.run(debug=True)