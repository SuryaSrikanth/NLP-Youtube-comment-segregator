import re
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


input = 'Movie is bad.'

def pprocess(input):
    pp_i = re.findall("[a-zA-Z]+", input)
    pp = []
    for x in pp_i:
        k = x.lower()
        if k not in stopwords.words('english') and len(x)>2:
            k = lemmatizer.lemmatize(k)
            pp.append(k)

    return pp



# Main algorithm
def iterative_hypernym(pp):
    ss0 = []
    arr_len = len(pp)
    idx = 0
    while len(ss0) == 0:
        ss0 = wn.synsets(pp[idx])
        idx = idx + 1

    # ss1 = wn.synsets(pp[1])[0]
    common_hypernym = ss0
    while idx<arr_len:
        ss1 = []
        while (len(ss1) == 0) or (idx < arr_len):

            ss1 = wn.synsets(pp[idx])
            idx = idx + 1

        if ss1 == []:
            common_hypernym = ss0[0]
        else:        
            common_hypernym = common_hypernym[0].lowest_common_hypernyms(ss1[0], use_min_depth = True)
    
    return common_hypernym


input_pp = pprocess(input)
print(input_pp)

hypernym = iterative_hypernym(input_pp)
print(hypernym)