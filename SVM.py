import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split #split data into train and test sets
from sklearn.feature_extraction.text import CountVectorizer #convert text comment into a numeric vector
from sklearn.feature_extraction.text import TfidfTransformer #use TF IDF transformer to change text vector created by count vectorizer
from sklearn.svm import SVC# Support Vector Machine
from sklearn.pipeline import Pipeline #pipeline to implement steps in series
from gensim import parsing # To stem data
from joblib import dump, load #save or load model
from sklearn.metrics import accuracy_score
from pdftotext import convert_pdf_to_string

#Read dataset
df = pd.read_csv("dataset.csv", encoding='latin1')

#applying parsing to text.
def parse(s):
    parsing.stem_text(s)
    #print()
    return s

#Iterate the dataset for analysing
for i in range(0, len(df)):
    df.iloc[i, 1] = parse(df.iloc[i, 1])

    #Seperate data into keyword and taxonomy level
    #X, y = df['word'], df['taxonomy']
    X, y = df['Questions'], df['Category']   

    # Split data in train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Use pipeline to carry out steps in sequence with a single object
    # SVM's rbf kernel gives highest accuracy in this classification problem.
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC(kernel='rbf'))])

    # train model
    text_clf.fit(X_train, y_train)
    #print(X_test)
    predicted = text_clf.predict(X_test)
    
    #print(accuracy_score(y_test, predicted))

    #Build joblib file
    dump(text_clf,'model.joblib')


    
    
    
    

    

