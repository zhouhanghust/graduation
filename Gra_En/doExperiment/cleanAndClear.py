from nltk.tokenize import WordPunctTokenizer
from nltk.stem.lancaster import LancasterStemmer


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def clean_text(text):
    stopwords = stopwordslist('/Users/zhouxiaohang/PycharmProjects/Graduation/Gra_En/stopwords.txt')
    words = ' '.join([word.lower() for word in WordPunctTokenizer().tokenize(text) if (len(word) > 1 and word.lower() not in stopwords)])
    pure_text = ''
    for letter in words:
        if letter.isalpha() or letter == " ":
            pure_text += letter
    text = pure_text.split()
    st = LancasterStemmer()
    result = [st.stem(word) for word in text]
    return ' '.join(result)


if __name__ == "__main__":
    s = 'I\'m a professional OTR truck driver actually,actually, and I bought a TND 700 at a truck stop hoping to make my life easier.  Rand McNally, are you listening?First thing I did after charging it was connect it to my laptop and install the software and then attempt to update it.  The software detected a problem with my update and wanted my home address so I could be sent a patch on an SD card.  Hello?  I don\'t think I\'m all that unusual; my home address is a PO box that a friend checks weekly and that I might get to check every six months or so.  I live in my truck and at truck stops.  If you need to make a patch available on an SD card then you should send the SD cards to the truck stops where the devices are sold.  I ran the update program multiple times until the program said that the TND 700 was completely updated.I programmed in the height (13\'6")'

    text = clean_text(s)
    print(text)







