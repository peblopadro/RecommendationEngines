import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
import pickle


def load_data(database_filepath):
    """
    Loads data from SQL Database and transforms it for model  training
    
    :param database_filepath: SQL database file (string)
    
    :returns x: Features (dataframe)
    :returns y: Targets (dataframe)
    :returns category_names: Target labels (list)
    """
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('SELECT * FROM df', engine)
    
    bad_samples = []
    for i in df.index:
        if df.loc[i,'related'] == 2:
            bad_samples.append(i)
    good_samples = [i for i in df.index if i not in bad_samples ]   
    df = df.iloc[good_samples,]   

    features = ['message']
    X = df[features]
    
    non_target_cols = ['id','message','original','genre'] 
    target_cols = df.columns[~df.columns.isin(non_target_cols)]
    Y = df[target_cols]
    
    category_names = Y.columns
    return X,Y, category_names


def tokenize(text):
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    return tokens


def build_model():
    """
    Builds a model using Random Forest Classifier. Data is transformed in pipeline using Tokenization, Count Vectorizer,
    Tfidf Transformer and
    
    :return cv: Trained model after performing grid search (GridSearchCV model)
    """

    class StartingVerbExtractor(BaseEstimator, TransformerMixin):
        def starting_verb(self, text):
            sentence_list = nltk.sent_tokenize(text)
            for sentence in sentence_list:
                pos_tags = nltk.pos_tag(tokenize(sentence))
                try:
                    first_word, first_tag = pos_tags[0]
                    if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                        return True
                except:
                    return False
            return False
        def fit(self, x, y=None):
            return self
        def transform(self, X):
            X_tagged = pd.Series(X).apply(self.starting_verb)
            return pd.DataFrame(X_tagged) 
    
    pipeline = Pipeline([
    ('features', FeatureUnion([
        ('nlp_pipeline',  Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize) ),
            ('tfidf', TfidfTransformer() )]) ),
        ('txt_extract', StartingVerbExtractor() )]) ),          
    ('clf', MultiOutputClassifier(RandomForestClassifier() ) )])
    
    parameters = {
    'features__nlp_pipeline__vect__tokenizer': [tokenize,None], # for count vectorizer
    'features__nlp_pipeline__tfidf__norm': ['l1','l2'], # for tfidf
    'clf__estimator__criterion': ['gini','entropy']} # for Random Forest
    
    cv = GridSearchCV(pipeline, param_grid=parameters,scoring='f1_macro')
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):                     
    """
    Measures model's performance on test data and prints out results.
    :param model: trained model (GridSearchCV Object)
    :param X_test: Test features (dataframe)
    :param Y_test: Test targets (dataframe)
    :param category_names: Target labels (dataframe)
    
    :return model_report: Dataframe with Model Performance report
    """             
    Y_pred = model.best_estimator_.predict(X_test)
    #print("\nBest Parameters:", model.best_params_)
    #score = f1_score( Y_test,Y_pred, average='macro')
    
    return print(classification_report(Y_test,Y_pred_cv,target_names=category_names))


def save_model(model, model_filepath):
    """
    Function to save trained model as pickle file.
    :param model: Trained model (GridSearchCV Object)
    :param model_filepath: Filepath to store model (string)
    :return: None
    """
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
