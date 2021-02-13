import pandas as pd
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('Video_games_esrb_rating.csv')
test_data = pd.read_csv('test_esrb.csv')
test_x = test_data.loc[:,[
    'console',
    'alcohol_reference',
    'animated_blood',
    'blood',
    'blood_and_gore',
    'cartoon_violence',
    'crude_humor',
    'drug_reference',
    'fantasy_violence',
    'intense_violence',
    'language',
    'lyrics',
    'mature_humor',
    'mild_blood',
    'mild_cartoon_violence',
    'mild_fantasy_violence',
    'mild_language',
    'mild_lyrics',
    'mild_suggestive_themes',
    'mild_violence',
    'no_descriptors',
    'nudity',
    'partial_nudity',
    'sexual_content',
    'sexual_themes',
    'simulated_gambling',
    'strong_janguage',
    'strong_sexual_content',
    'suggestive_themes',
    'use_of_alcohol',
    'use_of_drugs_and_alcohol',
    'violence'
]]
x = data.loc[:,[
    'console',
    'alcohol_reference',
    'animated_blood',
    'blood',
    'blood_and_gore',
    'cartoon_violence',
    'crude_humor',
    'drug_reference',
    'fantasy_violence',
    'intense_violence',
    'language',
    'lyrics',
    'mature_humor',
    'mild_blood',
    'mild_cartoon_violence',
    'mild_fantasy_violence',
    'mild_language',
    'mild_lyrics',
    'mild_suggestive_themes',
    'mild_violence',
    'no_descriptors',
    'nudity',
    'partial_nudity',
    'sexual_content',
    'sexual_themes',
    'simulated_gambling',
    'strong_janguage',
    'strong_sexual_content',
    'suggestive_themes',
    'use_of_alcohol',
    'use_of_drugs_and_alcohol',
    'violence'
]]
test_y = test_data.loc[:, 'esrb_rating']
y = data.loc[:, 'esrb_rating']
classifier = LogisticRegression()
classifier.fit(x,y)
y_pred = classifier.predict(test_x)
errors = 0
for i in range(len(y_pred)):
    if y_pred[i] != test_y[i]:
        errors+=1
print(y_pred)
print("There are {} errors out of {} tests.".format(errors, len(y_pred)))
