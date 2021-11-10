import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(filename):
    '''
    Loads data from given json file into a dataframe.

    INPUT
    filename - name of the file to load

    OUTPUT
    Returns a single pandas DataFrame
    '''
    return pd.read_json(filename, orient='records', lines=True)


def unwrapDictionary(df, dictVar, col):
    '''
    Unwraps the value column from transcript dataframe into the given column
    
    INPUT:
    df - dataframe that contains the dictionary in column "value"
    dictVar - name of the variable in the dictionary 
    col - name of the column, where the value of the dictVar should be saved into
    '''
    df[col] = [v.get(dictVar) for v in df['value']]


def mergeOfferEvent(offers, offer_df, time_column, status):
    '''
        Merge given dataframes on person and offer_id. 
        Depending on the given timeColumn the already received offer will be found and 
        the status for that entry will be the new given status.
        
        INPUT:
        offers - the main dataframe to merge into
        offer_df - the dataframe to merge
        time_column - name of the time column in that dataset (must not be "time")
        status - the new status to set
    '''
    
    offers = offers.merge(offer_df[['person','offer_id',time_column]], on=['person','offer_id'], how='left')

    offers['duration'] = offers[time_column] - offers['time']
    offers = offers.drop(offers[offers['duration'] < 0].index)
    offers = offers.drop_duplicates(subset=['person','offer_id','time'])

    offers.loc[offers[time_column].notna(), 'status'] = status

    offers = offers.drop(columns=['duration', time_column])
    return offers


def transform_data(transcript, profile):
    '''
        Extracts offer events from transcript, calculates final status of each offer
        and enrich data from profile. 

        INPUT:
        transcript - dataframe containing offer events from customer
        profile - dataframe containing attributes for each customer

        OUTPUT
        Return an offers dataframe. 
    '''
    # Prepare offer_ dataframes
    offer_received = pd.DataFrame(transcript[transcript['event']=='offer received'])
    unwrapDictionary(offer_received, 'offer id', 'offer_id')

    offer_viewed = pd.DataFrame(transcript[transcript['event']=='offer viewed'])
    unwrapDictionary(offer_viewed, 'offer id', 'offer_id')
    offer_viewed['time_viewed'] = offer_viewed['time']

    offer_completed = pd.DataFrame(transcript[transcript['event']=='offer completed'])
    unwrapDictionary(offer_completed, 'offer_id', 'offer_id')
    offer_completed['time_completed'] = offer_completed['time']

    # Build offers dataframe
    offers = pd.DataFrame(offer_received[['person', 'offer_id', 'time']])
    offers['status'] = "Received"

    offers = mergeOfferEvent(offers, offer_viewed, 'time_viewed', 'Viewed')
    offers = mergeOfferEvent(offers, offer_completed, 'time_completed', 'Completed')

    # Merge protfolio into offers
    offers = offers.merge(profile, left_on='person', right_on='id', how='left')

    offers = offers[['offer_id','time','gender','age','became_member_on','income','status']]
    offers = offers.dropna()

    offers = pd.get_dummies(offers, columns=['offer_id', 'gender'])

    return offers


def save_data(offers):
    '''
    Saves given DataFrame into given SQLite database file.

    INPUT
    df: pandas.DataFrame to save
    database_filename: name of the SQLite database
    '''
    engine = create_engine('sqlite:///offers.db')
    offers.to_sql('offers', engine, index=False, if_exists='replace') 


def main():

    print('Loading data...')
    transcript = load_data('./data/transcript.json')
    profile = load_data('./data/profile.json')

    print('Cleaning data...')
    offers = transform_data(transcript, profile)
    
    print('Saving data...')
    save_data(offers)
    
    print('Preprocessed offers saved to database!')


if __name__ == '__main__':
    main()