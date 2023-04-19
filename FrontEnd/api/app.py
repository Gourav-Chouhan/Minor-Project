import re
import nltk
import pickle
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask
from flask_cors import CORS
from keras_preprocessing.text import tokenizer_from_json
import json

import re
import nltk
import pickle
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import tokenizer_from_json
import json
import string
import os

import googletrans
from langdetect import detect
from googletrans import Translator

from textblob import TextBlob

from flask import Flask, request
from werkzeug.utils import secure_filename

import snscrape.modules.twitter as sntwitter
import pandas as pd

def user_comment(user):
    try:
        query = f"from:{user}"
        tweets = []
        limit = 10

        #  Remove tweets.csv if it already exists
        # if os.path.exists("./temp/tweets.csv"):
        #     os.remove("./temp/tweets.csv")

        for tweet in sntwitter.TwitterSearchScraper(query).get_items():
            
            if len(tweets) == limit:
                break
            else:
                tweets.append([tweet.content])
                
        df = pd.DataFrame(tweets, columns=['Tweet'])
        # df.to_csv("./temp/tweets.csv", index=False)
        return df
    except Exception as e:
        print(f"An error occurred: {e}")



#-----------------------------------

def cleaning(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('@\S+', '', text)
    return text

# Perform the Contractions on the reviews.
# Example: it won’t be converted as it will not be
def contractions(text):
 text = re.sub(r"won't", "will not",text)
 text = re.sub(r"could't", "could not",text)
 text = re.sub(r"would't", "would not",text)
 text = re.sub(r"\'d", "would",text)
 text = re.sub(r"can\'t", "can not",text)
 text = re.sub(r"n\'t",  "not", text)
 text = re.sub(r"\'re", "are", text)
 text = re.sub(r"\'s", "is", text)
 text = re.sub(r"\'ll", "will", text)
 text = re.sub(r"\'t", "not", text)
 text = re.sub(r"\'ve", "have", text)
 text = re.sub(r"\'m", "am", text)
 return text

# Remove non-alpha characters

def remove_non_alpha(text):
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [re.sub('[^A-Za-z]+', '', token) for token in tokens]
    return " ".join(cleaned_tokens)


stop_words = stopwords.words('english')

def remove_stopwords(text):
    tokens = text.split()
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return " ".join(filtered_tokens)

def lemmatizer_on_text(data):
    wordnet_lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(data)
    # Perform lemmatization on each word
    lemmatized_words = [wordnet_lemmatizer.lemmatize(word) for word in words]
    # Join the lemmatized words back into a single string
    lemmatized_sentence = " ".join(lemmatized_words)
    return lemmatized_sentence;




def translate_to_english(text):
    # Instantiate the translation client
    # translate_client = translate.Client();

    # Detect the language of the input text
    detection = translate_client.detect_language(text)

    # Split the input text into words or tokens
    words = text.split()

    # Translate non-English words to English
    for i, word in enumerate(words):
        if not word.isalpha():
            continue

        if detection['language'] != 'en':
            translation = translate_client.translate(word, target_language='en')
            words[i] = translation['translatedText']

    # Join the words back together to form the translated text
    translated_text = ' '.join(words)

    return translated_text


translator = Translator()

def convert_to_english(text):
    try:
        lang = translator.detect(text)
        print(lang)
        if lang.lang == 'en':
            return text
        translation = translator.translate(text, src=lang, dest='en')
        return translation.text
    except Exception as e:
        print(f"An error occurred: {e}")

#-----------------------------------


def spell_check(text):
    try:
        # create a TextBlob object
        blob = TextBlob(text)
        # correct spelling errors using TextBlob's built-in correct() method
        corrected_text = str(blob.correct())
        return corrected_text
    except Exception as e:
        print(f"An error occurred: {e}")
# Transforming abbreviations
abbreviations = {
    u"he's": "he is", 
    u"there's": "there is", 
    u"We're": "We are", 
    u"That's": "That is", 
    u"won't": "will not", 
    u"they're": "they are", 
    u"Can't": "Cannot", 
    u"wasn't": "was not", 
    u"don\x89Ûªt": "do not", 
    u"aren't": "are not", 
    u"isn't": "is not", 
    u"What's": "What is", 
    u"haven't": "have not", 
    u"hasn't": "has not", 
    u"There's": "There is", 
    u"He's": "He is", 
    u"It's": "It is", 
    u"You're": "You are", 
    u"I'M": "I am", 
    u"shouldn't": "should not", 
    u"wouldn't": "would not", 
    u"i'm": "I am", 
    u"I\x89Ûªm": "I am", 
    u"I'm": "I am", 
    u"Isn't": "is not", 
    u"Here's": "Here is", 
    u"you've": "you have", 
    u"you\x89Ûªve": "you have", 
    u"we're": "we are", 
    u"what's": "what is", 
    u"couldn't": "could not", 
    u"we've": "we have", 
    u"it\x89Ûªs": "it is", 
    u"doesn\x89Ûªt": "does not", 
    u"It\x89Ûªs": "It is", 
    u"Here\x89Ûªs": "Here is", 
    u"who's": "who is", 
    u"I\x89Ûªve": "I have", 
    u"y'all": "you all", 
    u"can\x89Ûªt": "cannot", 
    u"would've": "would have", 
    u"it'll": "it will", 
    u"we'll": "we will", 
    u"wouldn\x89Ûªt": "would not", 
    u"We've": "We have", 
    u"he'll": "he will", 
    u"Y'all": "You all", 
    u"Weren't": "Were not", 
    u"Didn't": "Did not", 
    u"they'll": "they will", 
    u"they'd": "they would", 
    u"DON'T": "DO NOT", 
    u"That\x89Ûªs": "That is", 
    u"they've": "they have", 
    u"i'd": "I would", 
    u"should've": "should have", 
    u"You\x89Ûªre": "You are", 
    u"where's": "where is", 
    u"Don\x89Ûªt": "Do not", 
    u"we'd": "we would", 
    u"i'll": "I will", 
    u"weren't": "were not", 
    u"They're": "They are", 
    u"Can\x89Ûªt": "Cannot", 
    u"you\x89Ûªll": "you will", 
    u"I\x89Ûªd": "I would", 
    u"let's": "let us", 
    u"it's": "it is", 
    u"can't": "cannot", 
    u"don't": "do not", 
    u"you're": "you are", 
    u"i've": "I have", 
    u"that's": "that is", 
    u"i'll": "I will", 
    u"doesn't": "does not",
    u"i'd": "I would", 
    u"didn't": "did not", 
    u"ain't": "am not", 
    u"you'll": "you will", 
    u"I've": "I have", 
    u"Don't": "do not", 
    u"I'll": "I will", 
    u"I'd": "I would", 
    u"Let's": "Let us", 
    u"you'd": "You would", 
    u"It's": "It is", 
    u"Ain't": "am not", 
    u"Haven't": "Have not", 
    u"Could've": "Could have", 
    u"youve": "you have",   
    u"donå«t": "do not", 
}
 
def transform_abb(text):
    for emot in abbreviations:
        text = re.sub(u'('+emot+')', " ".join(abbreviations[emot].replace(",","").split()), text)
    return text

vocab_size = 30000
embedding_dim = 32
max_length = 300
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

with open('./models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('./models/tokenizer_twit.json') as f:
    data = json.load(f)
    tokenizer_twit = tokenizer_from_json(data)

tokenizers = [tokenizer, tokenizer_twit]

# Transforming abbreviations
abbreviations = {
    u"he's": "he is", 
    u"there's": "there is", 
    u"We're": "We are", 
    u"That's": "That is", 
    u"won't": "will not", 
    u"they're": "they are", 
    u"Can't": "Cannot", 
    u"wasn't": "was not", 
    u"don\x89Ûªt": "do not", 
    u"aren't": "are not", 
    u"isn't": "is not", 
    u"What's": "What is", 
    u"haven't": "have not", 
    u"hasn't": "has not", 
    u"There's": "There is", 
    u"He's": "He is", 
    u"It's": "It is", 
    u"You're": "You are", 
    u"I'M": "I am", 
    u"shouldn't": "should not", 
    u"wouldn't": "would not", 
    u"i'm": "I am", 
    u"I\x89Ûªm": "I am", 
    u"I'm": "I am", 
    u"Isn't": "is not", 
    u"Here's": "Here is", 
    u"you've": "you have", 
    u"you\x89Ûªve": "you have", 
    u"we're": "we are", 
    u"what's": "what is", 
    u"couldn't": "could not", 
    u"we've": "we have", 
    u"it\x89Ûªs": "it is", 
    u"doesn\x89Ûªt": "does not", 
    u"It\x89Ûªs": "It is", 
    u"Here\x89Ûªs": "Here is", 
    u"who's": "who is", 
    u"I\x89Ûªve": "I have", 
    u"y'all": "you all", 
    u"can\x89Ûªt": "cannot", 
    u"would've": "would have", 
    u"it'll": "it will", 
    u"we'll": "we will", 
    u"wouldn\x89Ûªt": "would not", 
    u"We've": "We have", 
    u"he'll": "he will", 
    u"Y'all": "You all", 
    u"Weren't": "Were not", 
    u"Didn't": "Did not", 
    u"they'll": "they will", 
    u"they'd": "they would", 
    u"DON'T": "DO NOT", 
    u"That\x89Ûªs": "That is", 
    u"they've": "they have", 
    u"i'd": "I would", 
    u"should've": "should have", 
    u"You\x89Ûªre": "You are", 
    u"where's": "where is", 
    u"Don\x89Ûªt": "Do not", 
    u"we'd": "we would", 
    u"i'll": "I will", 
    u"weren't": "were not", 
    u"They're": "They are", 
    u"Can\x89Ûªt": "Cannot", 
    u"you\x89Ûªll": "you will", 
    u"I\x89Ûªd": "I would", 
    u"let's": "let us", 
    u"it's": "it is", 
    u"can't": "cannot", 
    u"don't": "do not", 
    u"you're": "you are", 
    u"i've": "I have", 
    u"that's": "that is", 
    u"i'll": "I will", 
    u"doesn't": "does not",
    u"i'd": "I would", 
    u"didn't": "did not", 
    u"ain't": "am not", 
    u"you'll": "you will", 
    u"I've": "I have", 
    u"Don't": "do not", 
    u"I'll": "I will", 
    u"I'd": "I would", 
    u"Let's": "Let us", 
    u"you'd": "You would", 
    u"It's": "It is", 
    u"Ain't": "am not", 
    u"Haven't": "Have not", 
    u"Could've": "Could have", 
    u"youve": "you have",   
    u"donå«t": "do not", 
}
 
def transform_abb(text):
    for emot in abbreviations:
        text = re.sub(u'('+emot+')', " ".join(abbreviations[emot].replace(",","").split()), text)
    return text

  
stop_words = set(stopwords.words('english'))

def remove_stop_words(text):
    words = text.split(' ')
    arr = [w for w in words if not w in stop_words]
    return ' '.join(arr)



def clean_text(text):
    text = re.sub(r'http[^ ]* ', '', text)
    text = re.sub(r'#\S* ', '', text)
    text = re.sub(r'@\S* ', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s\s+', ' ', text)
    text = text.lower()
    text = transform_abb(text)
    stopwords_list = stopwords.words('english')
    text = spell_check(text)
    text = ' '.join([word for word in text.split() if word not in stopwords_list])
    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text


# Load the six pre-trained models and store them in a list
models = [
    tf.keras.models.load_model('./models/simple_ann.h5'),  # Index 0
    tf.keras.models.load_model('./models/simple_ann_model_imdb.h5'),  # Index 1
    tf.keras.models.load_model('./models/cnn_model.h5'),         # Index 2
    tf.keras.models.load_model('./models/cnn_model_imdb.h5'),         # Index 3
    tf.keras.models.load_model('./models/bilstm_model.h5'),      # Index 4
    tf.keras.models.load_model('./models/bilstm_model_imdb.h5')       # Index 5
]

# Corresponding model names for each index in the 'models' list
model_names = [
    'simple_ann_model_twit',   # Index 0
    'simple_ann_model_imdb',   # Index 1
    'cnn_model_twit',          # Index 2
    'cnn_model_imdb',          # Index 3
    'bilstm_model_twit',       # Index 4
    'bilstm_model_imdb'        # Index 5
]



classes_twitter = ['Positive', "Neutral",  'Negative']
classes_imdb = ['Positive', 'Negative']


app = Flask(__name__)

cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/predict', methods=['POST'])
def prediction():
    data = request.get_json()
    model_index = data.get('model_index', 4)
    if model_index < 0 or model_index > 5:
        model_index = 4
    model = models[model_index]
    
    
    s = data.get('text', "I am happy")
    sequences = tokenizer.texts_to_sequences([s])
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    x = model.predict(padded)[0]
    if model_index % 2 == 0:
        maxIndex = 0
        if x[0] < x[1]:
            maxIndex = 1
        return classes_twitter[maxIndex] + ", Confidence: " + str(x[maxIndex])
        # return classes_twitter[np.argmax(x)] + ", Confidence: " + str(x[np.argmax(x)])
    else:
        return classes_imdb[np.argmax(x)] + ", Confidence: " + str(x[np.argmax(x)])
    
# Predict all route
@app.route('/predict_all', methods=['POST'])
def prediction_all():
    data = request.get_json()
    text = data.get('text', None)
    if text is None:
        return jsonify({'error': 'No text provided'})
    containsNegative = False
    # list of some negative words
    negative_words = ['not', 'no', 'nevelkasr', 'nothing', 'nowhere', 'noone', 'none', 'not', 'havent', 'hasnt', 'hadnt', 'cant', 'couldnt', 'shouldnt', 'wont', 'wouldnt', 'dont', 'doesnt', 'didnt', 'isnt', 'arent', 'aint']
    for word in negative_words:
        if word in text:
            containsNegative = True
            break
    # text_twit =convert_to_english(text_twit)
    text_twit =spell_check(text)
    text_twit = cleaning(text_twit)
    # text = contractions(text)
    # text = remove_non_alpha(text)
    text_twit = remove_stopwords(text_twit)
    text_twit = lemmatizer_on_text(text_twit)

    text = clean_text(text)

    processed_text = text
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    unseen_tokenized = tokenizer_twit.texts_to_sequences([text_twit])
    unseen_padded = pad_sequences(unseen_tokenized, padding='post', maxlen=100)
    overall_prediction_twit = 0
    overall_prediction_imdb = 0
    results_twit = []
    results_imdb = []

    pred_twit = []
    pred_imdb = []
    for i in range(len(models)):
        model_name = model_names[i]
        model = models[i]
        if i % 2 == 0:
            predictions = model.predict(unseen_padded)[0]
        else:
            predictions = model.predict(padded)[0]
        # predictions = model.predict(padded)[0]
        if containsNegative:
            # swap predictions
            if i % 2 == 0:
                predictions[0], predictions[2] = predictions[2], predictions[0]
                predictions[0] -= 0.01
                predictions[2] -= 0.01
            else :
                predictions[0], predictions[1] = predictions[1], predictions[0]
                predictions[0] -= 0.01
                predictions[1] -= 0.01
        if i % 2 == 0:
            # predictions[2] = 0
            # predictions[3] = 0
            classes = classes_twitter
        else:
            classes = classes_imdb
        
        top_prediction_index = np.argmax(predictions)
        top_prediction = classes[top_prediction_index]
        confidence = round(predictions[top_prediction_index] * 100, 2)
        
        if top_prediction == 'Positive':
            if i % 2 == 0:
                overall_prediction_twit += confidence
            else:
                overall_prediction_imdb += confidence
        else:
            if i % 2 == 0:
                overall_prediction_twit -= confidence
            else:
                overall_prediction_imdb -= confidence
        
        if i % 2 == 0:
            pred_twit.append(top_prediction)
        else:
            pred_imdb.append(top_prediction)
    
        result = {
            'model': model_name,
            'prediction': top_prediction,
            'confidence': confidence
        }
        if i % 2 == 0:
            results_twit.append(result)
        else:
            results_imdb.append(result)


    overall_prediction_twit = round(overall_prediction_twit / 3, 2)
    overall_prediction_imdb = round(overall_prediction_imdb / 3, 2)
    # overall_sentiment = 'Neutral'

    if overall_prediction_twit > 0:
        overall_prediction_sentiment_twit = 'Positive'
    else:
        overall_prediction_sentiment_twit = 'Negative'
    
    if overall_prediction_imdb > 0:
        overall_prediction_sentiment_imdb = 'Positive'
    else:
        overall_prediction_sentiment_imdb = 'Negative'

    best_model_twit = pred_twit.index(max(pred_twit))
    best_model_imdb = pred_imdb.index(max(pred_imdb))

    return jsonify({'processed_text': processed_text, 'twitter': {
        'results': results_twit,
        'overall_prediction': abs(overall_prediction_twit),
        'overall_prediction_sentiment': overall_prediction_sentiment_twit,
        'best_model': best_model_twit
    }, 'imdb': {
        'results': results_imdb,
        'overall_prediction': abs(overall_prediction_imdb),
        'overall_prediction_sentiment': overall_prediction_sentiment_imdb,
        'best_model': best_model_imdb
    }})

@app.route('/sum', methods=['POST'])
def sum_numbers():
    data = request.get_json()
    numbers = data['numbers']
    result = sum(numbers)
    return jsonify({'result': result})

@app.route('/hello', methods=['GET'])
def hello():
    return 'Hello, World!'


#Spelling check
from textblob import TextBlob
def spell_check(text):
    textBlb = TextBlob(text)
    return textBlb.correct().string

@app.route('/spell', methods=['POST'])
def spell_correction():
    return spell_check(request.get_json()['text'])

# Define default endpoint
@app.route('/')
def default():
    return jsonify({
        'message': 'Welcome to the Sentiment Analysis API!',
        'endpoints': {
            '/help': 'Provides information about using the API',
            '/predict_all': 'Predicts sentiment of given text for all models',
            '/predict/{model_index}': 'Predicts sentiment of given text for specific model, default is 4',
        },
        'model_names': {
            '0': 'simple_ann_model_twit',
            '1': 'simple_ann_model_imdb',
            '2': 'cnn_model_twit',
            '3': 'cnn_model_imdb',
            '4': 'bilstm_model_twit',
            '5': 'bilstm_model_imdb'
        }
    })

# Define help endpoint
@app.route('/help')
def help():
    return jsonify({
        'message': 'To predict sentiment of text, send a POST request to /predict with JSON data',
        'json_data': {
            'text': 'Text to analyze (optional, defaults to "I am happy" if not provided)'
        },
        'example': {
            'endpoint': '/predict',
            'JSON data': {
                'text': 'This movie was terrible!'
            }
        },
        'model_names': {
            '0': 'simple_ann_model_twit',
            '1': 'simple_ann_model_imdb',
            '2': 'cnn_model_twit',
            '3': 'cnn_model_imdb',
            '4': 'bilstm_model_twit',
            '5': 'bilstm_model_imdb'
        }
    })




if __name__ == '__main__':
    app.run(debug=True)



