import pandas as pd
import glob

# df_A = pd.read_csv('./crawling/team_crawling/Menupan.com_Chinese_69.csv')
# print(df_A)
data_paths = glob.glob('./crawling/team_crawling/*')
data_paths_2 = glob.glob('./crawling/team_crawling(2)_noindexes/*')

print(data_paths)
print(data_paths_2)

df = pd.DataFrame()
for data_path in data_paths:
    df_temp = pd.read_csv(data_path, index_col=0)
    df = pd.concat([df, df_temp[:1000]])

# 이대로 concat하면 인덱스가 없는 채로 저장된 애들 다 nan값으로 바뀌어버려.
# 그래서 아예 따로 파일을 만들어 저장하고, df에 합쳤음.
# :1000 한 것은 한식 데이터가 너무 많아서

for data_path_2 in data_paths_2:
    df_temp_2 = pd.read_csv(data_path_2)
    df = pd.concat([df, df_temp_2])


df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(df)
print(df.columns)
df.info()
print(df['Category'].value_counts())
# df.to_csv('./crawling/concat_all_recipes.csv', index=False)