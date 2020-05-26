# Data libraries
import pandas as pd

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white')
sns.set(style='whitegrid', color_codes=True)

# Corpuses and constants
BAD_WORDS_CORPUS = ['die', 'bitch', 'cunt', 'dick', 'faggot', 'slut', 'shit', 'pussy', 'loser', 'kill', 'fuck', 'fat', 'suck', 'whore']
THEIR_CYBERBULLYING_PROBABILITY_VALUES = [['die', 0.37], ['bitch', 0.34], ['cunt', 0.36], ['dick', 0.62], ['faggot', 0.22], ['slut', 0.48], ['shit', 0.28], ['pussy', 0.75], ['loser', 0.08], ['kill', 0.40], ['fuck', 0.52], ['fat', 0.47], ['suck', 0.41], ['whore', 0.53]] # approximate, read by eye from graph
NO_BAD_WORD = 'no_bad_word'


def get_bad_words_from_string(string):
    """
    Returns the bad words found in a string.
    """
    bad_words = []
    for bad_word in BAD_WORDS_CORPUS:
        if bad_word in string:
            bad_words.append(bad_word)

    return bad_words


clean_tweets = pd.read_csv('../Data/Twitter/CleanedCyberBullyingTweetsAll.csv')

"""
We need a new DataFrame to prepare a dataset to which we will add newly computed rows based on the bad_word-s.

Each tweet could possibly contain more bad words and that is why we need to append rows with the
same tweet content if we find more of them.

E.g. from the sentence You re a bitch cunt whore Bless Madonna, we want to have:

639            You re a bitch cunt whore Bless Madonna                1    bitch
640            You re a bitch cunt whore Bless Madonna                1     cunt
641            You re a bitch cunt whore Bless Madonna                1    whore
"""
tweets = pd.DataFrame(columns=['tweet', 'is_cyberbullying', 'bad_word'])

# We go through all original tweets and generate rows based on the unique bad words
for _, tweet_data in clean_tweets.iterrows():
    for bad_word in get_bad_words_from_string(tweet_data['tweet']):
        tweets = tweets.append(pd.Series([tweet_data['tweet'], tweet_data['is_cyberbullying'], bad_word], index=tweets.columns), ignore_index=True)

probabilities = []

# For each bad word, we compute how many times that word occurs in a cyberbullying tweet and divide it by the number of all tweets with that word.
for bad_word in BAD_WORDS_CORPUS:
    tweets_for_word = tweets[tweets['bad_word'] == bad_word]
    bully_tweets_for_word = tweets_for_word[tweets_for_word['is_cyberbullying'] == 1]
    probability = len(bully_tweets_for_word) / len(tweets_for_word)

    probabilities.append([bad_word, probability])

our_probabilities_df = pd.DataFrame(probabilities, columns = ['bad_word', 'probability'])
our_probabilities_df['authors'] = 'Us (small Kasture dataset)'
their_probabilities_df = pd.DataFrame(THEIR_CYBERBULLYING_PROBABILITY_VALUES, columns = ['bad_word', 'probability'])
their_probabilities_df['authors'] = 'Authors (big dataset)'

final_df = pd.concat([our_probabilities_df, their_probabilities_df])
sns.set(rc={'figure.figsize':(11.7,8.27)})
prob_plot = sns.barplot(data=final_df, x='bad_word', y='probability', hue='authors')
prob_plot.set(xlabel='Bad word', ylabel='Probability of cyberbullying')
plt.savefig('../Article/Images/ProbabilityPerWord-BarPlot.pdf')
plt.show()

