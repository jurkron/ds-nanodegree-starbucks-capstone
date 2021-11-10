import pandas as pd
import numpy as np

from flask import Flask
from flask import render_template, request, jsonify
import joblib


app = Flask(__name__)

# load portfolio
portfolio = pd.read_json('../data/portfolio.json', orient='records', lines=True)

# load model
model = joblib.load("../starbucks_classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    gender = request.args.get('gender', 'gender_F') 
    income = request.args.get('income', 60000) 
    age = request.args.get('age', 30) 

    # render web page with plotly graphs
    return render_template('master.html', gender=gender, income=income, age=age)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    gender = request.args.get('gender', '') 
    income = request.args.get('income', 60000) 
    age = request.args.get('age', 45) 

    df_predict = pd.DataFrame(np.array([
        [53.0,75000.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [53.0,75000.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [53.0,75000.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [53.0,75000.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [53.0,75000.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [53.0,75000.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [53.0,75000.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [53.0,75000.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0],
        [53.0,75000.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0],
        [53.0,75000.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
        ]),
        columns=['age', 'income',
       'offer_id_0b1e1539f2cc45b7b9fa7c272da2e1d7',
       'offer_id_2298d6c36e964ae4a3e7e9706d1fb8c2',
       'offer_id_2906b810c7d4411798c6938adc9daaa5',
       'offer_id_3f207df678b143eea3cee63160fa8bed',
       'offer_id_4d5c57ea9a6940dd891ad53e9dbe8da0',
       'offer_id_5a8bc65990b245e5a138643cd4eb9837',
       'offer_id_9b98b8c7a33c4b65b9aebfe6a799e6d9',
       'offer_id_ae264e3637204a6fb9bb56bc8210ddfd',
       'offer_id_f19421c1d4aa40978ebb69ca19b0e20d',
       'offer_id_fafdcd668e3743c1bb461111dcafc2a4', 
       'gender_F', 
       'gender_M',
       'gender_O']
    )

    df_predict[gender]=1
    df_predict['income']=int(income)
    df_predict['age']=int(age)

    # use model to predict classification for query
    classification_labels = model.predict(df_predict)
    # model.grid_search.predict_proba()
    portfolio_ids=[
       'offer_id_0b1e1539f2cc45b7b9fa7c272da2e1d7',
       'offer_id_2298d6c36e964ae4a3e7e9706d1fb8c2',
       'offer_id_2906b810c7d4411798c6938adc9daaa5',
       'offer_id_3f207df678b143eea3cee63160fa8bed',
       'offer_id_4d5c57ea9a6940dd891ad53e9dbe8da0',
       'offer_id_5a8bc65990b245e5a138643cd4eb9837',
       'offer_id_9b98b8c7a33c4b65b9aebfe6a799e6d9',
       'offer_id_ae264e3637204a6fb9bb56bc8210ddfd',
       'offer_id_f19421c1d4aa40978ebb69ca19b0e20d',
       'offer_id_fafdcd668e3743c1bb461111dcafc2a4']

    portfolio_type = ['discount','discount','discount','informational','bogo','informational','bogo','bogo','bogo','discount']
    classification_results = dict(zip(zip(portfolio_ids, portfolio_type), classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        gender=gender,
        income=income,
        age=age,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
    