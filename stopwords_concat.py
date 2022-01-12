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

# 이대로 concat하면 인덱스가 없는 채로 저장된 애들 다 nan값으로 바뀌어버려.
# 그래서 아예 따로 파일을 만들어 저장하고, df에 합쳤음.
# :1000 한 것은 한식 데이터가 너무 많아서

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(df)

df.info()
# print(df['Category'].value_counts())
df.to_csv('./crawling/stopword_final.csv')