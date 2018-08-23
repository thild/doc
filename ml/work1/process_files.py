# nltk.download('punkt')
# nltk.download('stopwords')

from gensim import utils
import nltk
from nltk.corpus import stopwords
from sources import sources

cachedStopWords = set(stopwords.words("english"))

def process_files(sources):
    for source, prefix in sources.items():
        print('Processing ' + 'data/' + source + '...')
        with open('data/' + source, 'w') as outfile:
            with utils.smart_open('data/tmp/' + source) as fin:
                for item_no, line in enumerate(fin):
                    strline = utils.to_unicode(line)
                    # punct = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{}~'   # `|` is not present here
                    # transtab = str.maketrans(dict.fromkeys(punct, ''))
                    # strline = strline.lower().translate(transtab)
                    strline = strline.lower()
                    tokens = nltk.word_tokenize(strline)
                    tagged = nltk.pos_tag(tokens)
                    notnouns = [word for word, pos in tagged if (pos != 'NN' or pos != 'NNP' or pos != 'NNS' or pos != 'NNPS')]
                    strline = ' '.join([word for word in notnouns if word not in cachedStopWords])
                    outfile.write(strline + '\n')

process_files(sources)