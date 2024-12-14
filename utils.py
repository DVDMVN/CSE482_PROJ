import pandas as pd
import numpy as np
import ast

from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess

from sklearn.utils import resample

def load_master_df() -> pd.DataFrame:
    def load_bunch() -> dict[pd.DataFrame]:
        apex = pd.read_csv('processed_data/apex_ad2600_dvd_player_updated.csv')
        canon = pd.read_csv('processed_data/canon_g3_updated.csv')
        nikon = pd.read_csv('processed_data/nikon_coolpix_4300_updated.csv')
        nokia = pd.read_csv('processed_data/nokia_6610_updated.csv')
        nomad = pd.read_csv('processed_data/nomad_jukebox_zen_xtra_updated.csv')
        return {
            "apex": apex,
            "canon": canon,
            "nikon": nikon,
            "nokia": nokia,
            "nomad": nomad
        }

    def get_master_df(sentiments_only: bool = True) -> pd.DataFrame:
        bunch = load_bunch()
        master_df = pd.concat(bunch.values(), ignore_index=True)
        master_df['sentiment_dict'] = master_df['sentiment_dict'].apply(ast.literal_eval)
        if sentiments_only:
            master_df = master_df[master_df['sentiment_dict'].apply(lambda x: bool(x))]
        return master_df

    master_df = get_master_df(sentiments_only = False)

    # Binning as negative neutral positive
    # Define conditions
    conditions = [
        master_df['sentiment_total'] > 0,  # Positive sentiment
        master_df['sentiment_total'] < 0,  # Negative sentiment
        master_df['sentiment_total'] == 0  # Neutral sentiment
    ]

    # Define corresponding labels
    labels = ['positive', 'negative', 'neutral']

    # Create a new column for binned sentiment
    master_df['sentiment_category'] = np.select(conditions, labels)
    master_df['sentiment_category'].value_counts()

    # Tokenization and removal of stopwords
    master_df['sentence'] = master_df['sentence'].apply(lambda x: remove_stopwords(str(x)))
    master_df['tokenized_sentence'] = master_df['sentence'].apply(simple_preprocess)
    return master_df

def balance_master_df_classes(master_df: pd.DataFrame, print_out: bool = True) -> pd.DataFrame:
    positive_df = master_df[master_df['sentiment_category'] == 'positive']
    negative_df = master_df[master_df['sentiment_category'] == 'negative']
    neutral_df = master_df[master_df['sentiment_category'] == 'neutral']

    max_size = 2000

    # Resample each class to match the majority class size
    positive_upsampled = resample(positive_df, replace=True, n_samples=max_size, random_state=42)
    negative_upsampled = resample(negative_df, replace=True, n_samples=max_size, random_state=42)
    neutral_upsampled = resample(neutral_df, replace=True, n_samples=max_size, random_state=42)

    # Combine the upsampled dataframes
    balanced_master_df = pd.concat([positive_upsampled, negative_upsampled, neutral_upsampled])

    # Shuffle
    balanced_master_df = balanced_master_df.sample(frac=1, random_state=42).reset_index(drop=True)

    if print_out: # Sanity check
        print(balanced_master_df['sentiment_category'].value_counts())
    return balanced_master_df