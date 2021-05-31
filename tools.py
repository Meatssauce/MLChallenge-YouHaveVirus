import os
import json
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk import pos_tag, word_tokenize
import nltk
nltk.download('averaged_perceptron_tagger')

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

def load_data(labels_path, conversations_folder_path, verbose=0):
    ''' 
    Load data from a label file and a folder containing conversations as jsonf files. 

    Returns:
        pandas.DataFrame
    '''
    
    def parse_incomplete_conversation(content, system_questions):
        content = json.dumps(content)[1:-1]

        # get question by index in content, sorted by index in content
        search_results = {content.find(f'\"{question}\": '): question for question in system_questions}
        search_results.pop(-1, None)
        search_results = sorted(search_results.items())
        
        result = {question: '' for question in system_questions}
        for i in range(len(search_results)):
            # extract substring between two subsequent questions as answer to the former question
            k, question = search_results[i][0], search_results[i][1]
            k_next = search_results[i + 1][0] if i < len(search_results) - 1 else len(content)
            result[question] = content[k + len(f'\"{question}\": '):k_next]

        return result

    # Check if path exists.
    if not os.path.isfile(labels_path):
        raise ValueError('Invalid `labels_path` variable! Needs to be a file directory')
    if not os.path.isdir(conversations_folder_path):
        raise ValueError('Invalid `conversations_folder` variable! Needs to be a folder directory')

    # record omitted data
    n_unlabelled = n_incomplete = 0

    # Get labels
    df_labels = pd.read_csv(labels_path)

    # Get labeled conversations
    conversations = {}
    df_conversations = pd.DataFrame()
    file_names = os.listdir(conversations_folder_path)
    for file_name in file_names:
        user_id = file_name[:-len('.json')]
        if user_id not in df_labels['ID'].values:
            n_unlabelled += 1
            continue

        # load conversation file
        file_path = os.path.join(conversations_folder_path, file_name)
        with open(file_path) as f:
            content = json.load(f)

        # handle incomplete conversation
        system_questions = ["How may I help you? ", "What is you body temperature? ", "What Symptoms do you have? ", 
        "Based on your recent travel history (past_week), which catergory do you belong to? ", "Share your Current location? "]
        if list(content.keys()) != system_questions:
            n_incomplete += 1 #TODO: fix invalid conversations instead of skipping them. Then remake infected_train and infected_test with GPT2
            content = parse_incomplete_conversation(content, system_questions)
        
        # parse conversation and add to dictionary
        if not conversations:
            conversations = {k: [v] for k, v in content.items()}
            conversations['ID'] = [user_id]
        else:
            for k, v in content.items():
                conversations[k].append(v)
            conversations['ID'].append(user_id)
    df_conversations = pd.DataFrame(conversations)
    
    df_conversations.rename(columns={
        'How may I help you? ': 'request', 
        'What is you body temperature? ': 'temperature', 
        'What Symptoms do you have? ': 'symptoms',
        'Based on your recent travel history (past_week), which catergory do you belong to? ': 'travel_category',
        'Share your Current location? ': 'location'
        }, inplace=True)

    if verbose == 1:
        print(f'Loading data from \'{labels_path}\' and \'{conversations_folder_path}\'...')
        s = f'Loaded {df_conversations.shape[0]} samples, including {n_incomplete} incomplete instances. Omitted {n_unlabelled} unlabelled samples. '
        s += f'{df_labels.shape[0] - df_conversations.shape[0]} labels unused.'
        print(s)

    return pd.merge(df_labels, df_conversations, on='ID')

class SubsetLikelihood2D:
    def __init__(self, bins):
        self.bins = bins
        self.xedges = None
        self.yedges = None
        self.proba = None
    
    def fit(self, subset, superset):
        assert(len(subset) == len(superset) == 2)

        X, y = subset[0], subset[1]
        X_super, y_super = superset[0], superset[1]
        h, _, _ = np.histogram2d(X, y, bins=self.bins)
        h_super, self.xedges, self.yedges = np.histogram2d(X_super, y_super, bins=self.bins)
        self.proba = h / h_super

        return self

    def score(self, X, y):
        assert(len(X) == len(y))

        scores = []
        for i in X.index:
            x_hist = self._findBin(X[i], self.xedges)
            y_hist = self._findBin(y[i], self.yedges)
            if (x_hist == None or y_hist == None):
                scores.append(-1)
            else:
                scores.append(self.proba[x_hist][y_hist])

        return pd.Series(scores, index=X.index)
    
    def fit_score(self, X, y=None):
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X).score(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y).score(X)
    
    def _findBin(self, k, cuts):
        ''' 
        Find index of first bin where the right edge is greater than k. 
        Except for last cut, where the last bin is return if the cut >= k.
        Returns None if no such edge is found or if the left edge of the first bin is greater than k.
        '''
        if cuts[-1] == k:
            return len(cuts) - 2
        for i in range(len(cuts)):
            if cuts[i] > k:
                return i - 1
        return None

class Preprocessor(TransformerMixin, BaseEstimator):
    def __init__(self, bins):
        self.bins = bins
        self.is_fitted = False
        self.mlb = None
        self.likelihood = None
    
    def _to_celsius(self, deg, inverse=False):
        return (deg - 32) * 5/9 if inverse == False else deg * 9/5 + 32
    
    def clean(self, df):
        # symptoms

        df['symptoms'] = df['symptoms'].str.lower()
        df['symptoms'] = df['symptoms'].str.replace(r'[^a-z_,\s]', '', regex=True)
        df['symptoms'] = df['symptoms'].str.replace(r'\s+', ' ', regex=True)
        df['symptoms'] = df['symptoms'].str.replace(r',\s+', ',', regex=True)
        # TODO: implement more sophisticated pattern matching e.g. match if text distance is below n
        df['symptoms'] = df['symptoms'].str.split(',')
        if not self.is_fitted:
            self.mlb = MultiLabelBinarizer()
            self.mlb = self.mlb.fit(df['symptoms'])
        df_symptoms = pd.DataFrame(self.mlb.transform(df['symptoms']), columns=self.mlb.classes_, index=df.index).astype(bool)
        df = pd.concat([df, df_symptoms], axis=1)
        df = df.drop(columns=['symptoms'])

        # request

        # convert to lower case
        df['request'] = df['request'].str.lower()
       
        # correct spelling

        # remove punctuations
        df['request'] = df['request'].str.replace('[^\w\s\d]', '', regex=True)

        # Stopwords removal and lemmatisation
        lemmatizer = WordNetLemmatizer()
        # all_stopwords = set(stopwords.words())
        # all_stopwords -= {'i', 'someone', 'do', 'not', 'no', 'have'}
        all_stopwords = {'a', 'the', 'these', 'those', 'this', 'that', 'some', 'several', 'few', 'many', 'lot', 'bit', 'bunch', 'of', 'all'}
        df["request"] = [' '.join([lemmatizer.lemmatize(token, tag[0].lower()) if tag[0].lower() in ['a', 'r', 'v', 'n'] else lemmatizer.lemmatize(token) 
            for token, tag in pos_tag(word_tokenize(s)) 
            if token not in all_stopwords]) 
            for s in df["request"]]

        # temperature

        pattern = r'[^.0-9]*([1-9]?[0-9]+(?:\.[0-9]+)?)[^.0-9]*'
        df['temperature'] = df['temperature'].str.extract(pattern)
        df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
        # TODO: extract number words and convert number words to numbers

        # convert temperature to fahrenheit if it is closer to body temperature in celsius
        body_temperature_feh = 97.7
        mid_point = (body_temperature_feh + self._to_celsius(body_temperature_feh)) / 2
        df['temperature'] = df['temperature'].mask(df['temperature'] < mid_point, self._to_celsius(df['temperature'], inverse=True))

        # TODO: implement more sophisticated pattern matching e.g. match if text distance is below n
        # travel category
        pattern = r'^type_([1-9])'
        df['travel_category'] = df['travel_category'].str.extract(pattern)
        df['travel_category'] = pd.to_numeric(df['travel_category'])

        # device location
        pattern = r'^Device location : \[(-?[1-9]?[0-9]?(?:\.[0-9]+)?), -?[1-9]?[0-9]{0,2}(?:\.[0-9]+)?\]$'
        df['latitude'] = df['location'].str.extract(pattern)
        df['latitude'] = pd.to_numeric(df['latitude'])

        is_invalid = df['latitude'].abs() > 90
        df['latitude'] = df['latitude'].mask(is_invalid, np.nan)

        pattern = r'^Device location : \[-?[1-9]?[0-9]?(?:\.[0-9]+), (-?[1-9]?[0-9]{0,2}(?:\.[0-9]+))\]$'
        df['longitude'] = df['location'].str.extract(pattern)
        df['longitude'] = pd.to_numeric(df['longitude'])

        is_invalid = df['longitude'].abs() > 180
        df['longitude'] = df['longitude'].mask(is_invalid, np.nan)

        df = df.drop(columns=['location'])

        if not self.is_fitted:
            self.likelihood = SubsetLikelihood2D(self.bins)
            infected = (df['longitude'][df['GPT2_Prediction'] == 1], df['latitude'][df['GPT2_Prediction'] == 1])
            total = (df['longitude'], df['latitude'])
            self.likelihood = self.likelihood.fit(infected, total)
        scores = pd.Series(self.likelihood.score(df['longitude'], df['latitude']), index=df.index)
        df['likelihood'] = scores
        

        # Impossible body temperature (nan or outside conscious body temperature range (28, 43) °C or (82.4, 109.4) °F )
        # See <Temperature variation> at https://en.wikipedia.org/wiki/Human_body_temperature#:~:text=44%20%C2%B0C%20(111.2%20%C2%B0,damage%2C%20continuous%20convulsions%20and%20shock.
        # dishonesty is a potential cause of the error so flag impossible temperatures
        min_temp, max_temp = 82.4, 109.4    # fahrenheit
        df['invalid_temperature'] = np.where(df['temperature'].isna() | (df['temperature'] < min_temp) | (df['temperature'] > max_temp), True, False)

        # Extract syptoms from self-statement
        key_phrase = r'(?:^|^ |i )(?!not |dont |havent )(?:ive|have|got|experience|suffer from) ((?:(?:symptom|covid19|covid 19|covid|coronavirus|corona virus|corona|\
            difficulty breathe|sortness breath|cough|dry cough|fever|flu|cold|temperature|sore throat|runny nose|headache|loss smell|no sense smell|\
            loss taste|no sense taste|tiredness|fatigue|low energy)(?: |$))+)'
            # unhandled cases: cant breathe, im cough, i am cough, i dont feel well, I feel like voimiting, i want to vomit, i cought cold etc
            # does not handle repeats
        df['symptoms_description'] = df['request'].str.extract(key_phrase)
        df['unspecified_symptoms'] = np.where(df['symptoms_description'].str.contains(r'symptom|covid19|covid 19|covid|coronavirus|corona virus|corona') == True, True, False)
        df['breathing_difficulty'] = np.where(df['symptoms_description'].str.contains(r'sortness breath|difficulty breathe') == True, True, df['breathing_difficulty'])
        df['cough'] = np.where(df['symptoms_description'].str.contains(r'cough') == True, True, df['cough'])
        df['dry_cough'] = np.where(df['symptoms_description'].str.contains(r'dry cough') == True, True, df['dry_cough'])
        df['fever'] = np.where(df['symptoms_description'].str.contains(r'fever|flu|cold|temperature|sore throat|runny nose|headache') == True, True, df['fever'])
        df['loss_of_smell'] = np.where(df['symptoms_description'].str.contains(r'no sense smell|loss smell') == True, True, df['loss_of_smell'])
        df['loss_of_taste'] = np.where(df['symptoms_description'].str.contains(r'no sense taste|loss taste') == True, True, df['loss_of_taste'])
        df['sore_throat'] = np.where(df['symptoms_description'].str.contains(r'sore throat') == True, True, df['sore_throat'])
        df['tiredness'] = np.where(df['symptoms_description'].str.contains(r'tiredness|fatigue|low energy') == True, True, df['tiredness'])
        df = df.drop(columns=['symptoms_description'])
        
        # TODO: skip extract use contains directly?

        return df
    
    def fit(self, X, y):
        self.is_fitted = False
        self.clean(X)
        self.is_fitted = True

        return self

    def transform(self, X, y=None):
        assert(self.is_fitted == True)

        X = self.clean(X)
        self.is_fitted = False

        return X, y
    
    def fit_transform(self, X, y=None):
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y).transform(X)
    
    def get_params(deep=True):
        return {}
    
    def set_params(**params):
        return self

def save(item, file_name):
    with open(file_name, 'wb') as opened_file:
        dill.dump(item, opened_file)

def load(file_name):
    with open(file_name, 'rb') as opened_file:
        item = dill.load(opened_file)
    
    return item