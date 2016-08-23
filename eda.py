import pandas as pd
df = pd.read_json('data/train_new.json')
df.acct_type.value_counts(dropna=False)

# adding a fraud column
df['fraud'] = df['acct_type'].apply(lambda x:
                                    1 if 'fraud' in x else 0)

# filter frauds
df_fraud = df.ix[df.fraud == 1]
