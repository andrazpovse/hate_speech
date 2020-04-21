import pandas as pd

if __name__ == '__main__':
    f_negative = open('../Data/Twitter/Cleaned CyberBullying Tweets - Negative.txt')
    negative = f_negative.read().splitlines()
    negative = list(filter(None, negative))

    negative = [(el, 0) for el in negative]

    f_positive = open('../Data/Twitter/Cleaned CyberBullying Tweets - Positive.txt')
    positive = f_positive.read().splitlines()
    positive = list(filter(None, positive))

    positive = [(el, 1) for el in positive]

    data = negative + positive

    df = pd.DataFrame(data, columns = ["tweet", "is_cyberbullying"])

    df.to_csv('../Data/Twitter/CleanedCyberBullyingTweetsAll.csv', index=False)
