# Importing libraries 
import nltk 
import re 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import pandas as pd 
import numpy as np

if __name__ == "__main__":
    inputfile = 'C:\\Users\\yashv\\Desktop\\facultylist1TextFile.txt'
    outputfile = 'facultylist1TextFileOutput.txt'
    
    # Input the file  
    txt1 = [] 
    with open(inputfile) as file: 
        txt1 = file.readlines() 

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
    #Write it into file
    with open(outputfile, 'w') as f:
        for item in X6:
            f.write(str(item)+'\n')

    
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