import sys
from sqlalchemy import create_engine
import pandas as pd
import re
import nltk
nltk.download(['punkt', 'wordnet'])
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

def load_data():
    '''
    Load offers dataset from given offers SQLite Database.

    OUTPUT
    X: Dataset containing offer data for train and test
    y: Dataset containing status for X used in train and test
    '''
    engine = create_engine('sqlite:///offers.db')
    offers = pd.read_sql_table('offers', con=engine)
    X = offers.drop(columns=['status', 'time', 'became_member_on'])
    y = offers['status']
    
    return X, y


def build_model():
    '''
    Returns a GridSearchCV model with classifier GradientBoostingClassifier. 
    '''
    param_grid = {
        "max_depth" : [3,4,5],
        "n_estimators": [50,100]
    }

    # initialize grid search
    return GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid=param_grid, scoring = 'f1_weighted', verbose=5)


def evaluate_model(model, X_test, y_test):
    '''
    Accuracy and f1-score for the given test dataset is printed out.

    INPUT
    model: the model to test
    X_test: a dataset used to make predictions on the given model
    y_test: a dataset containing the true responses on X_test, to evaluate the predictions against
    '''
    y_pred = model.predict(X_test)

    print('accuracy_score', accuracy_score(y_test, y_pred))
    print('f1_score      ', f1_score(y_test, y_pred, average='weighted'))



def save_model(model):
    '''
    Save the model as pickle file.
    INPUT
    model: model to save
    '''
    # save the model to disk
    pickle.dump(model, open('starbucks_classifier.pkl', 'wb'))


def main():
    print('Loading data...')
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print('Building model...')
    model = build_model()
    
    print('Training model...')
    model.fit(X_train, y_train)
    
    print('Evaluating model...')
    evaluate_model(model, X_test, y_test)

    print('Saving model...\n    MODEL: starbucks_classifier.pkl')
    save_model(model)

    print('Trained model saved!')


if __name__ == '__main__':
    main()