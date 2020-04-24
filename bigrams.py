# Importing libraries 
import nltk 
import re 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import pandas as pd 
import numpy as np
import csv


def WriteDataToCsv(dataList):
    with open("CsvFile.csv","w", encoding="utf-8", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(dataList)

if __name__ == "__main__":
    inputfile = 'C:\\Users\\yashv\\Desktop\\Merged.txt'
    
    # Input the file  
    txt1 = [] 
    with open(inputfile) as file: 
        txt1 = file.readlines() 

    #stopword removal
    nltk.download('stopwords')
    nltk.download('punkt')
    all_stopwords = set(stopwords.words('english'))
    new_words = ['2014','2016','2009','2017','2012','2011','2007','2006','2005','2018','in','the','x99s','x93','x9d','xe2','x80',',','.','(',')','4k','3-d','&',';','amp','_','[',']','it','also','asu','``','e.g','i','this','new',':','the','we','one','three','two','?','—','dr.','in','$','p.m.','5-12','yes','\'s','arizona','state','university','use','\'','7,000','at','25,000','an','each','that','first','these','but','43,000','53,000','us','most','as','ph.d.','try','\t','b','cv','ph.d.','â','arizona.â','cv\n\nâ','/','univ','y.','w.','v.','z.','university.â','467/567','six','-','h','r','az','co.','ltd.','e','h.','c.-f.','u.s.','u.s.â','universityâ','uk','wi.','x.','x',"na","'",'g','s.']
    new_stopwords_list = all_stopwords.union(new_words)
    for i, line in enumerate(txt1): 
        txt1[i] = ' '.join([x for x in nltk.word_tokenize(line) if ( x not in new_stopwords_list )]) 
    
    # Preprocessing 
    # def remove_string_special_characters(s): 
        
    #     # removes special characters with ' ' 
    #     stripped = re.sub('[^a-zA-z\s]', '', s) 
    #     stripped = re.sub('_', '', stripped) 
        
    #     # Change any white space to one space 
    #     stripped = re.sub('\s+', ' ', stripped) 
        
    #     # Remove start and end white spaces 
    #     stripped = stripped.strip() 
    #     if stripped != '': 
    #             return stripped.lower() 

        
    # Getting bigrams  
    vectorizer = CountVectorizer(ngram_range =(2, 2)) 
    X1 = vectorizer.fit_transform(txt1)  
    features = (vectorizer.get_feature_names()) 
    # Code to change datatype and merge the two outputs
    X2 = X1.toarray()
    X2 = X2.tolist()
    X3 = []
    for row in X2:
        for row2 in row:
            X3.append(row2)
    X5 = [(X3[i], features[i]) for i in range(0, len(X3))]
    X6 = sorted(X5,key=lambda l:l[0], reverse = True)
    WriteDataToCsv(X6)
    
    
    # Applying TFIDF 
    # You can still get n-grams here 
    #vectorizer = TfidfVectorizer(ngram_range = (2, 2)) 
    #X2 = vectorizer.fit_transform(txt1) 
    #scores = (X2.toarray()) 
    #print("\n\nScores : \n", scores)

    # Getting top ranking features 
    #sums = X2.sum(axis = 0) 
    #data1 = [] 
    #for col, term in enumerate(features): 
    #    data1.append( (term, sums[0, col]) )
    #print ("\n\nWords : \n", data1) 
    #ranking = pd.DataFrame(data1, columns = ['term', 'rank']) 
    #words = (ranking.sort_values('rank', ascending = False)) 
    #print ("\n\nWords : \n", ranking.head(7)) 