# -*- coding: utf-8 -*-

import numpy as np
import json
import time
import requests
import random

headers = {
    #'Referer':'https://item.m.jd.com/product/26654365116.html',
    'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Mobile Safari/537.36'
}

# pro = ['121.231.32.70','117.25.176.213','122.114.31.177','123.114.61.124','14.118.254.193','60.184.172.68',
#        '14.118.254.97','113.200.241.202','171.221.207.243','114.246.241.201','125.121.113.248','218.88.105.38',
#        '115.198.36.2','125.118.74.49','218.14.115.211','175.25.26.117','121.41.171.223','101.132.122.230']

def crawl_info(url,headers):
    result = []
    res = requests.get(url, headers=headers)        #,proxies={'http':random.choice(pro)}
    data = res.text.strip()[10:-1]
    data_js = json.loads(data)
    data_all = data_js['result']['comments']
    for each in data_all:
        result.append({'content':each['content'],'score':each['score']})
    return result

def main():
    result = []
    begin = 1
    end = 13
    for i in range(begin,end):
        url = "https://wq.jd.com/commodity/comment/getcommentlist?sorttype=5&sceneval=2&sku=26659498835&page=%s&pagesize=10" % i
        temp = crawl_info(url,headers)
        result.extend(temp)
        print("the %s loop has finished!"%i)
        time.sleep(random.choice([4,4.5,5,5.5,6,6.5,7]))
        if i % 25 == 0 :
            np.save("sanshith_%s.npy"%(i//25),np.array(result))
            result = []
            print("i will sleep 30s...")
            time.sleep(30)
            print("i wake up!")
            continue
        if i == end - 1:
            np.save("sanshith_%s.npy" % ("end"), np.array(result))

main()

