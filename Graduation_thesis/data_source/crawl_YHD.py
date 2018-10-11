# -*- coding: utf-8 -*-


from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import numpy as np

browser = webdriver.Chrome()
wait = WebDriverWait(browser,10)


#6946605,6946629,25390983397,26669892946,27136250757,28060284115,26664474952,26664647138


browser.get('http://item.yhd.com/11319391377.html')
browser.maximize_window()
browser.execute_script("window.scrollTo(0,700)")

submit = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR,'#sppj')))
submit.click()
time.sleep(5)


def next_page():
    browser.execute_script("window.scrollTo(0,document.body.scrollHeight)")
    time.sleep(2)
    submit = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR,'#pe_page > ul > li.latestnewnextpage')))
    submit.click()
    time.sleep(2)


def get_products():
    browser.execute_script("window.scrollTo(0,document.body.scrollHeight)")
    wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR,'#div_pe_clist'))
    )
    html = browser.page_source
    soup = BeautifulSoup(html,'lxml')
    items = soup.select('div.comment_con  .item.good-comment')
    result = []
    for item in items:
        product = {}
        try:
            product['star'] = int(item.select('dl dt.user_info span')[1]['class'][1][-1])
        except:
            continue
        product['comment'] = item.select('dd.clearfix span.text.comment_content_text')[0].text.strip()
        result.append(product)
    return result


data = get_products()
print("the 1 page has been downloaded!")
for i in range(99):
    next_page()
    temp = get_products()
    data.extend(temp)
    print("the %s page has been downloaded!"%(i+2))

np.save("ninth_YHD.npy", np.array(data))
