#import libraries
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import pandas as pd
import re
import nltk
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer#This is used for stemming purpose.
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
#call the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
#Download punkt
nltk.download('punkt')
#download stopwords
nltk.download('stopwords')


def pre_process_text(text):
     # Tokenization
    tokens = word_tokenize(text)

    # Remove non-alphanumeric characters and convert to lowercase
    review = re.sub('[^a-zA-Z0-9]', ' ', ' '.join(tokens))
    review = review.lower()

    # Split into words
    review = review.split()

    # Lemmatization and remove stopwords
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]

    # Join the words back into a string
    review = ' '.join(review)

    return review

    

def vectorize_text(train_text,test_text,max_features=5000):
      # Preprocess the text
    train_text = train_text.apply(pre_process_text)
    test_text = test_text.apply(pre_process_text)

    # Vectorize using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_text)
    X_test_tfidf = tfidf_vectorizer.transform(test_text)
    # Save the TF-IDF vectorizer to a pickle file
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')

    return X_train_tfidf, X_test_tfidf

def map_sentiments(sentiment):
    if sentiment == 'positive':
        return 0
    elif sentiment == 'negative':
        return 1
    else:  # Assuming 'neutral' for any other sentiment
        return 2








#read the train and test dat
#1st lets read the train data
train_df=pd.read_csv(r'C:\Users\HRISHAB\Documents\A.I_projects\twitter_sentiment_analysis\data\train.csv',encoding='ISO-8859-1')
#extracting text and sentiment column in training data
train_df=train_df[['text','sentiment']]

#read the test data
test_df=pd.read_csv('data/test.csv', encoding='ISO-8859-1')
#extracting text and sentiment column in test data
test_df=test_df[['text','sentiment']]

train_df=train_df.dropna()
test_df['text'] = test_df['text'].fillna('unknown') 
test_df['sentiment'] = test_df['sentiment'].fillna('unknown') 


#create train data and test data
X_train, X_test = vectorize_text(train_df['text'], test_df['text'])
#this is for o/p preprocessing
y_train=train_df['sentiment'].map(map_sentiments)
y_test=test_df['sentiment'].map(map_sentiments)

#now lets apply stratified shuffle split
stratified_split_train = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# Split the data for internal validation
#this will contain training data and test data from X_train and y_train
for train_index, val_index in stratified_split_train.split(X_train, y_train):
    X_train_internal, X_val_internal = X_train[train_index], X_train[val_index]
    y_train_internal, y_val_internal = y_train.iloc[train_index], y_train.iloc[val_index]
#now lets give the model the dataset

    
#define the model
model = SVC(kernel='linear')  

#Define the hyperparameter grid to search through
param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto']}

# Create a GridSearchCV object
model= GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3)


    
#give the model your dataset
model.fit(X_train_internal, y_train_internal)
    
#give your model strat_test_data
accuracy_internal = model.score(X_val_internal, y_val_internal)
#check the accuracy of strat_test_data
print(f'Accuracy on internal validation set: {accuracy_internal:.2f}')
#give your model real test data
accuracy_test = model.score(X_test, y_test)

#check the accuracy of the real test data  
print(f'Accuracy on test set: {accuracy_test:.2f}')
    
# Save the model
joblib.dump(model, 'SVC_classifier.pkl')
