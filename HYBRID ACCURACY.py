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
df = pd.read_csv("test_questions.csv", encoding='latin1')
print(df["Category"].value_counts())
print(df.shape)
print(df.loc[0]['Questions'])
print(len(df.index))
import json

keywords=['cite', 'add', 'acquire', 'analyze', 'abstract', 'appraise',
'define', 'approximate', 'adapt', 'audit', 'animate', 'assess',
'describe', 'articulate', 'allocate', 'blueprint', 'arrange', 'compare',
'draw', 'associate', 'alphabetize', 'breadboard', 'assemble', 'conclude',
'enumerate', 'characterize', 'apply', 'break', 'down', 'budget', 'contrast',
'identify', 'clarify', 'ascertain', 'characterize', 'categorize', 'counsel',
'index', 'classify', 'assign', 'classify', 'code', 'criticize',
'indicate', 'compare', 'attain', 'compare', 'combine', 'critique', 'difference', 'analyse', 'argue',
'label', 'compute', 'avoid', 'confirm', 'compile', 'defend',
'list', 'contrast', 'backup', 'contrast', 'compose', 'determine',
'match', 'convert', 'calculate', 'correlate', 'construct', 'discriminate',
'meet', 'defend', 'capture', 'detect', 'cope', 'estimate', 'differ',
'name', 'describe', 'change', 'diagnose', 'correspond', 'evaluate', 'what', 'how',
'outline', 'detail', 'classify', 'diagram', 'create', 'explain','why'
'point', 'differentiate', 'complete', 'differentiate', 'cultivate', 'grade',
'quote', 'discuss', 'compute', 'discriminate', 'debug', 'hire',
'read', 'distinguish', 'construct', 'dissect', 'depict', 'interpret','recall', 'elaborate', 'customize', 'distinguish', 'design', 'judge',
'recite', 'estimate', 'demonstrate', 'document', 'develop', 'justify',
'recognize', 'example', 'depreciate', 'ensure', 'devise', 'measure', 'record', 'explain', 'derive', 'examine', 'dictate', 'predict',
'repeat', 'express', 'determine', 'explain', 'enhance', 'prescribe', 'reproduce', 'extend', 'diminish', 'explore', 'explain', 'rank',
'review', 'extrapolate', 'discover', 'figure', 'out', 'facilitate', 'rate',
'select', 'factor', 'draw', 'file', 'format', 'recommend', 'state', 'generalize', 'employ', 'group', 'formulate', 'release', 'study', 'give',
'examine', 'identify', 'generalize', 'select',
'tabulate', 'infer', 'exercise', 'illustrate', 'generate','find', 'summarize', 'trace', 'interact', 'explore', 'infer', 'handle', 'support',
'write', 'interpolate', 'expose', 'interrupt', 'import', 'test', 'interpret', 'express', 'inventory', 'improve', 'validate', 'observe', 'factor',
'investigate', 'incorporate', 'verify','configure',
'paraphrase', 'figure', 'layout', 'integrate', 'picture', 'graphically', 'graph', 'manage', 'interface',
'predict', 'handle', 'maximize', 'join', 'review', 'illustrate', 'minimize', 'lecture', 'rewrite','restate', 'interconvert', 'optimize', 'model',
'subtract', 'investigate', 'order', 'modify', 'summarize', 'manipulate', 'outline', 'network', 'translate', 'modify', 'point out', 'organize',
'visualize', 'operate', 'prioritize', 'outline',
'personalize', 'proofread', 'overhaul', 'plot', 'query', 'plan', 'practice', 'relate', 'portray', 'predict', 'select', 'prepare',
'separate', 'prescribe', 'price', 'size up', 'produce', 'process', 'subdivide', 'program',
'train', 'rearrange', 'project', 'transform', 'reconstruct',
'protect', 'refer', 'provide', 'relate',  'reorganize',
'round off', 'revise', 'sequence', 'rewrite', 'show', 'specify', 'simulate', 'summarize', 'sketch', 'write', 'solve', 'subscribe'
,'tabulate', 'transcribe', 'translate', 'use']
def uniques(l1):
    l2=[]
    for x in l1:
        if x not in l2:
            l2.append(x)
    return l2
keywords2=[]
keywords2=uniques(keywords)

with open('count_record.json') as f1:
    count_record = json.load(f1)
with open('Knowledge_count.json') as f2:
    Knowledge_count = json.load(f2)
with open('Comprehension_count.json') as f3:
    Comprehension_count = json.load(f3)
with open('Application_count.json') as f4:
    Application_count = json.load(f4)
with open('Analysis_count.json') as f5:
    Analysis_count = json.load(f5)
with open('Synthesis_count.json') as f6:
    Synthesis_count = json.load(f6)
with open('Evaluation_count.json') as f7:
    Evaluation_count = json.load(f7)

#applying parsing to text.
def parse(s):
    parsing.stem_text(s)
    #print()
    return s

#def print_accuracy(mannual_copy):
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

  #Load joblib file
text_clf = load('model.joblib')
li_swn=[] 
  #Reading the multiple questions and predicting
  #from csv import DictReader
  #DictReader works by reading in the first line of the CSV
  #with open('dataset.csv', 'r', errors='ignore') as read_obj:
      #csv_dict_reader = DictReader(read_obj)
for m in range(0,len(df.index)):
          #li_swn1=[]
          text1 = df.loc[m]['Questions']
          #print('\n',text1)
          text1 = text1.lower()
          from nltk.tokenize import sent_tokenize, word_tokenize
          token = word_tokenize(text1)
          #Prediction
          predicted = text_clf.predict(token)
          predicted_list = tuple(set(predicted))
          #print(predicted_list)
          weights_dict = {}
          weight1 = 0
          weight2 = 0
          weight3 = 0
          weight4 = 0
          weight5 = 0
          weight6 = 0
          for j in keywords2:
            if j in token:
              for i in predicted_list:    
                if i==str('Application'):
                  weight1 += Application_count[j]/count_record[j]
                  weights_dict[i]=weight1
                if i==str('Knowledge'):
                  weight2 += Knowledge_count[j]/count_record[j]
                  weights_dict[i]=weight2
                if i==str('Analysis'):
                  weight3 += Analysis_count[j]/count_record[j]
                  weights_dict[i]=weight3
                if i==str('Comprehension'):
                  weight4 += Comprehension_count[j]/count_record[j]
                  weights_dict[i]=weight4
                if i==str('Synthesis'):
                  weight5 += Synthesis_count[j]/count_record[j]
                  weights_dict[i]=weight5
                if i==str('Evaluation'):
                  weight6 += Evaluation_count[j]/count_record[j]
                  weights_dict[i]=weight6
          if (not (weights_dict)):
            for i in predicted_list:
              if i==str('Knowledge'):
                li_swn.append("Knowledge")
              if i==str('Comprehension'):
                li_swn.append("Comprehension")
              if i==str('Application'):
                li_swn.append("Application")
              if i==str('Analysis'):
                li_swn.append("Analysis")
              if i==str('Synthesis'):
                li_swn.append("Synthesis")
              if i==str('Evaluation'):
                li_swn.append("Evaluation")
                            
          else:
            max=0
            for key in weights_dict:
              if weights_dict[key] > max:
                max = weights_dict[key] 
            for key,value in  weights_dict.items():
              if max == value:   
                if key==str('Knowledge'):
                  li_swn.append("Knowledge")
                  break
                if key==str('Comprehension'):
                  li_swn.append("Comprehension")
                  break
                if key==str('Application'):
                  li_swn.append("Application")
                  break
                if key==str('Analysis'):
                  li_swn.append("Analysis")
                  break
                if key==str('Synthesis'):
                  li_swn.append("Synthesis")
                  break
                if key==str('Evaluation'):
                  li_swn.append("Evaluation")
                  break
          #print(li_swn)
df.insert(2,"HYB_PREDICTION",li_swn,False)
category_list = df['Category'].tolist()
Prediction_list = df['HYB_PREDICTION']
from sklearn.metrics import confusion_matrix
megha = confusion_matrix(category_list, Prediction_list)
print(megha)
#return mannual_copy
#df1 = print_accuracy(df)
#print(df[:10])

total=len(df.index)
def accuracy_test(mannual_copy1):
    acc=0
    for i in range(0,len(mannual_copy1.index)):
        category = mannual_copy1.loc[i]['Category']
        prediction = mannual_copy1.loc[i]['HYB_PREDICTION']
        """print(category)
        print(type(category))
        print(prediction_list)
        print(type(prediction_list))"""
        if category==prediction:
            acc=acc+1
    return acc
c=accuracy_test(df)
a=((float(c))/total)*100
print(a)

#Iterate the dataset for analysing
'''for i in range(0, len(df)):
    df.iloc[i, 1] = parse(df.iloc[i, 1])

    #Seperate data into keyword and taxonomy level
    #X, y = df['word'], df['taxonomy']
    X, y = df['Questions'], df['Category']   

    # Split data in train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Use pipeline to carry out steps in sequence with a single object
    # SVM's rbf kernel gives highest accuracy in this classification problem.
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC(kernel='rbf'))])

    
    text_clf.fit(X_train, y_train)
    
    predicted = text_clf.predict(X_test)
    
    dump(text_clf,'model.joblib')'''


    
    
    
    

    

