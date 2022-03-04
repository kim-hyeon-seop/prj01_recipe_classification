import pandas as pd
import glob

# df_A = pd.read_csv('./crawling/team_crawling/Menupan.com_Chinese_69.csv')
# print(df_A)
data_paths = glob.glob('./stopwords/*')

print(data_paths)

df = pd.DataFrame()
for data_path in data_paths:
    df_temp = pd.read_csv(data_path, index_col=0)
    df = pd.concat([df, df_temp])

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(df)

df.info()
# print(df['Category'].value_counts())
df.to_csv('./crawling/stopword_final.csv')
