from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk


text = 'This is a very nice day'


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
    
print(label)