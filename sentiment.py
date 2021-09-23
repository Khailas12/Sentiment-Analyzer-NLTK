from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flask import request
import nltk


text = 'today is a wonderful day'

def sentiment_analyzer():
    # downloading the trained data VADER( Valence Aware Dictionary for Sentiment Reasoning)
    nltk.download('vader_lexicon') 
    sentiment_analyzer = SentimentIntensityAnalyzer()
    score = ((sentiment_analyzer.polarity_scores(str(text))))['compound']

    if score > 0:
        label = 'Positive'

    elif score == 0:
        label = 'Neutral'
        
    else:
        label = 'Negative'

    print(f"The result is : {label}")
    
sentiment_analyzer()

# VADER is a model used for text sentiment analysis that is sensitive to both polarity (positive/negative) and intensity (strength) of emotion. ... The sentiment score of a text can be obtained by summing up the intensity of each word in the text