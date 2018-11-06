# -*- coding: utf-8 -*-
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator

with open('../cutted_umbalanced_data/cutted_umbalanced_data.pkl','rb') as f:
    data = pickle.load(f)

contents = data.content
segmentation = ' '.join(contents)


print(segmentation)
background_image=plt.imread(r"micky.jpg")#加载背景图片,返回的结果为RGB颜色值
print("图片加载成功")


wc=WordCloud(
    width=1024,
    height=768,
    background_color="white",
    mask=background_image,#设置背景图片
    font_path=r"SIMLI.ttf",#加载中文字体路径，不加载中文会出错，英文不会
    max_words=50,
    random_state=5
)
wc.generate_from_text(segmentation)
imag_colors=ImageColorGenerator(background_image)#提取背景图片颜色
wc.recolor(color_func=imag_colors)#设置字体颜色为背景图片颜色
plt.imshow(wc,interpolation='bilinear')
plt.axis('off')
plt.show()
