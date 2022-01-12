from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
import pandas as pd
import requests
import re
import time

options = webdriver.ChromeOptions()
options.add_argument('lang=ko_KR')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('disable_gpu')

driver = webdriver.Chrome('./chromedriver', options=options)
headers = {"User-Agent":'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36'}

df_ingredients = pd.DataFrame()
category = ['Korean','Japanese','Chinese','Western']
ingredients = []
ingredient = ''

# 일식

for j in range(1,7):
    url = 'https://www.menupan.com/Cook/recipere.asp?nation=30&page={}'.format(j)
    driver.get(url)
    for i in range(1,16):
        try:
            ingredient = driver.find_element_by_xpath(
                '/html/body/table[2]/tbody/tr/td[3]/table/tbody/tr/td/table/tbody/tr[6]/td/table[{}]/tbody/tr[1]/td/table/tbody/tr/td[2]/table/tbody/tr[3]/td'.format(i)).text
            ingredient = ingredient[5:]
            ingredient = re.compile('[^가-힣a-zA-Z]').sub(' ', ingredient)
            print(ingredient)
            ingredients.append(ingredient)

        except NoSuchElementException:
            ingredient = driver.find_element_by_xpath(
                '/html/body/table[2]/tbody/tr/td[3]/table/tbody/tr/td/table/tbody/tr[6]/td/table[{}]/tbody/tr[2]/td/table/tbody/tr/td[2]/table/tbody/tr[3]/td'.format(
                    i)).text
            ingredient = ingredient[5:]
            ingredient = re.compile('[^가-힣a-zA-Z]').sub(' ', ingredient)
            print(ingredient)
            ingredients.append(ingredient)

print(ingredients)
print(len(ingredients))
df_ingredients_style = pd.DataFrame(ingredients, columns=['ingredients'])
df_ingredients_style['category'] = 'Japanese'
print(df_ingredients_style)
df_ingredients_style.to_csv('./crawling/Menupan.com_Japanese_{}.csv'.format(len(ingredients)), index=False)


# 중식
# try:
#     for j in range(1,6):
#         url = 'https://www.menupan.com/Cook/recipere.asp?nation=40&page={}'.format(j)
#         driver.get(url)
#         for i in range(1,16):
#             try:
#                 ingredient = driver.find_element_by_xpath(
#                     '/html/body/table[2]/tbody/tr/td[3]/table/tbody/tr/td/table/tbody/tr[6]/td/table[{}]/tbody/tr[1]/td/table/tbody/tr/td[2]/table/tbody/tr[3]/td'.format(i)).text
#                 ingredient = ingredient[5:]
#                 ingredient = re.compile('[^가-힣a-zA-Z]').sub(' ', ingredient)
#                 print(ingredient)
#                 ingredients.append(ingredient)
#
#             except NoSuchElementException:
#                 ingredient = driver.find_element_by_xpath(
#                     '/html/body/table[2]/tbody/tr/td[3]/table/tbody/tr/td/table/tbody/tr[6]/td/table[{}]/tbody/tr[2]/td/table/tbody/tr/td[2]/table/tbody/tr[3]/td'.format(
#                         i)).text
#                 ingredient = ingredient[5:]
#                 ingredient = re.compile('[^가-힣a-zA-Z]').sub(' ', ingredient)
#                 print(ingredient)
#                 ingredients.append(ingredient)
# except: pass
#
# print(ingredients)
# print(len(ingredients))
# df_ingredients_style = pd.DataFrame(ingredients, columns=['ingredients'])
# df_ingredients_style['category'] = 'Chinese'
# print(df_ingredients_style)
# df_ingredients_style.to_csv('./crawling/Menupan.com_Chinese_{}.csv'.format(len(ingredients)), index=False)