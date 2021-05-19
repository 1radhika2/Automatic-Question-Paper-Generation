import pandas as pd
import json
dataframe = pd.read_csv("dataset.csv", encoding='latin1')
keywords=['cite', 'add', 'acquire', 'analyze', 'abstract', 'appraise',
'define', 'approximate', 'adapt', 'audit', 'animate', 'assess',
'describe', 'articulate','why', 'allocate', 'blueprint', 'arrange', 'compare',
'draw', 'associate', 'alphabetize', 'breadboard', 'assemble', 'conclude',
'enumerate', 'characterize','configure', 'apply', 'break', 'down', 'budget', 'contrast',
'identify', 'clarify', 'ascertain', 'characterize', 'categorize', 'counsel',
'index', 'classify', 'assign', 'classify', 'code', 'criticize',
'indicate', 'compare', 'attain', 'compare', 'combine', 'critique', 'difference', 'analyse', 'argue',
'label', 'compute', 'avoid', 'confirm', 'compile', 'defend',
'list', 'contrast', 'backup', 'contrast', 'compose', 'determine',
'match', 'convert', 'calculate', 'correlate', 'construct', 'discriminate',
'meet', 'defend', 'capture', 'detect','restate', 'cope', 'estimate', 'differ',
'name', 'describe', 'change', 'diagnose', 'correspond', 'evaluate', 'what', 'how',
'outline', 'detail', 'classify', 'diagram', 'create', 'explain',
'point', 'differentiate', 'complete', 'differentiate', 'cultivate', 'grade',
'quote', 'discuss', 'compute', 'discriminate','find', 'debug', 'hire',
'read', 'distinguish', 'construct', 'dissect', 'depict', 'interpret','recall', 'elaborate', 'customize', 'distinguish', 'design', 'judge',
'recite', 'estimate', 'demonstrate', 'document', 'develop', 'justify',
'recognize', 'example', 'depreciate', 'ensure', 'devise', 'measure', 'record', 'explain', 'derive', 'examine', 'dictate', 'predict',
'repeat', 'express', 'determine', 'explain', 'enhance', 'prescribe', 'reproduce', 'extend', 'diminish', 'explore', 'explain', 'rank',
'review', 'extrapolate', 'discover', 'figure', 'out', 'facilitate', 'rate',
'select', 'factor', 'draw', 'file', 'format', 'recommend', 'state', 'generalize', 'employ', 'group', 'formulate', 'release', 'study', 'give',
'examine', 'identify', 'generalize', 'select',
'tabulate', 'infer', 'exercise', 'illustrate', 'generate', 'summarize', 'trace', 'interact', 'explore', 'infer', 'handle', 'support',
'write', 'interpolate', 'expose', 'interrupt', 'import', 'test', 'interpret', 'express', 'inventory', 'improve', 'validate', 'observe', 'factor',
'investigate', 'incorporate', 'verify',
'paraphrase', 'figure', 'layout', 'integrate', 'picture', 'graphically', 'graph', 'manage', 'interface',
'predict', 'handle', 'maximize', 'join', 'review', 'illustrate', 'minimize', 'lecture', 'rewrite', 'interconvert', 'optimize', 'model',
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
count_record = {}
dataframe['Questions']=dataframe['Questions'].str.lower()
for i in keywords2:
    count = 0
    for j in dataframe['Questions']:
        from nltk.tokenize import sent_tokenize, word_tokenize
        token = word_tokenize(j)
        if i in token:
            count = count+1
    count_record[i] = count
jsonn = json.dumps(count_record)
f = open("count_record.json","w")
f.write(jsonn)
f.close()

Blooms_levels =['Knowledge','Comprehension','Analysis','Application','Synthesis','Evaluation']

Knowledge_count = {}
Comprehension_count = {}
Analysis_count = {}
Application_count = {}
Synthesis_count = {}
Evaluation_count = {}

for a in Blooms_levels:
    for i in keywords2:
        count1 = 0
        for j in range(0,len(dataframe.index)):
            from nltk.tokenize import sent_tokenize, word_tokenize
            token2 = word_tokenize(dataframe.loc[j]['Questions'])
            if i in token2:
                if dataframe.loc[j]['Category']==a:
                    count1=count1+1
        if a=='Knowledge':
            Knowledge_count[i] = count1
        if a=='Comprehension':
            Comprehension_count[i] = count1
        if a=='Analysis':
            Analysis_count[i] = count1
        if a=='Application':
            Application_count[i] = count1
        if a=='Synthesis':
            Synthesis_count[i] = count1
        if a=='Evaluation':
            Evaluation_count[i] = count1

json1 = json.dumps(Knowledge_count)
f1 = open("Knowledge_count.json","w")
f1.write(json1)
f1.close()

json2 = json.dumps(Comprehension_count)
f2 = open("Comprehension_count.json","w")
f2.write(json2)
f2.close()

json3 = json.dumps(Application_count)
f3 = open("Application_count.json","w")
f3.write(json3)
f3.close()

json4 = json.dumps(Analysis_count)
f4 = open("Analysis_count.json","w")
f4.write(json4)
f4.close()

json5 = json.dumps(Synthesis_count)
f5 = open("Synthesis_count.json","w")
f5.write(json5)
f5.close()

json6 = json.dumps(Evaluation_count)
f6 = open("Evaluation_count.json","w")
f6.write(json6)
f6.close()

print("done")
