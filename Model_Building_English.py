#!/usr/bin/env python
# coding: utf-8

# ## Preprocessing

# In[1]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pkg_resources
import string
import re
import json
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem import WordNetLemmatizer

 
lemma = WordNetLemmatizer()
stop_words = stopwords.words('english')


# In[2]:


uom = ['cm','mm', 'mhz', 'gb', 'ohm', 'w', 'inch', 'g', 'v', 'kg', 'mb', 'kb', 'gb', 'tb', 'ghz']

non_meaningful_name_words = ["11.11", "ge", "ready stock", "readystock", "promotion", "new", "local", "cheapest",
                             "preferred","azz","gaz","ger","aa","ee","jae","gea","gee","zz","ready","stock","zaz","chegg"
                             "ge", "r", "sec","az", "ff", "gf", "offer", "ready stock", "freegift", "free gift"]

def deEmojify(text):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', text)

def clean_data(data):
    lower_msg = data.lower() #lower all charracter
    msg = deEmojify(lower_msg) ##clear emoji
    msg = re.sub('[^a-zA-Z0-9.]+', '  ', msg) ##clean symbol
    msg = re.sub('(.x?)http.*?(.*?)', ' ', msg) #clean url
    word_token = nltk.tokenize.word_tokenize(msg)
    #msg = spell_checker(word_token)
    word_token = clr_stop_words(word_token)
    #lemmatized = lemma_words(clean_stop_words)
    text = ' '.join(map(str, word_token))
    return text

def clr_stop_words(words):  ##clear stop words
    filter_words = []
    for w in words:
        if w not in stop_words:
            filter_words.append(w)

    return filter_words  

def spell_checker(words):
    mispelled = []
    for w in words:
        text = sym_spell.lookup(w,Verbosity.CLOSEST,max_edit_distance=2, include_unknown=True)
        for t in text:
            mispelled.append(t._term)
            
    return mispelled

def lemma_words(words):
    lemma_word = []
    for w in words:
        text = lemma.lemmatize(w)
        lemma_word.append(text)
    
    return lemma_word

def cleanSpecifications(text):
    text = text.lower()
    text = re.sub('category.*components > ', '', text) 
    text = re.sub('ships.*\n', '', text) #clean "Ships From"
    text = re.sub('stock.*[0-9]', '', text) #clean "Number of Stock"
    text = re.sub('warranty.*\n', '', text) #clean warranty
    text = re.sub('duration.*\n', '', text) #clean "Duration"
    text = re.sub('condition used.*\n', '',text)#clean "Condition Used"
    text = re.sub('\n', ' ',text)# clean extra line
    return text

def cleanDescriptions(text):
    text = text.lower()
    text = re.sub('\n', ' ',text)# clean extra line
    text = re.sub('terms and condition.*$', '', text) 
    #text = re.sub('terms and condition*', '', text)
    #m = re.search('terms and condition.*', text)
    #if m:
    #    text = m.group(0)
    #re.match('terms and condition', text).group(0)
    return text

#49 tkkke | 92k ratings | 25.2k sold
def cleanName(text):
    text = text.lower()
    text = re.sub('...\n.*sold', '', text) 
    text = remove_non_meaningful_name_words(text)
    return text


def remove_one_char(text):
    word_token = nltk.tokenize.word_tokenize(text)
    if len(word_token) != 0:
        if len(word_token[0]) == 1:
            return " ".join(word_token[1:])
        else:
            return ' '.join(word_token)
    else:
        return text
    
def filterOutNumberUom(text):
    i = 0
    digit_uom = []
    filteredName = []
    word_token = nltk.tokenize.word_tokenize(text)
    for token in word_token:
        contain_digit_uom = False
        if token in uom:
            contain_digit_uom = True
        else:
            for character in token:
                if character.isdigit():
                    contain_digit_uom = True
        if contain_digit_uom == True:
            digit_uom.append(token)
        else:
            filteredName.append(token)
            
    return [' '.join(filteredName), digit_uom]

def remove_non_meaningful_name_words(text):
    clean = []
    word_token = nltk.tokenize.word_tokenize(text)
    for word in word_token:
        if word not in non_meaningful_name_words:
            clean.append(word)
            
    return ' '.join(clean)


# In[3]:


def removebrand(text):
    text = text.lower()
    token = text.split()
    if ' '.join(token[:3]) == "chargers power supply":
        return "chargers power supply"
    elif ' '.join(token[:2]) == "fans heatsinks":
        return "fans heatsinks"
    elif ' '.join(token[:1]) == "ram":
        return "ram"
    elif ' '.join(token[:2]) == "graphic cards":
        return "graphic cards"
    elif ' '.join(token[:2]) == "thermal paste":
        return "thermal paste"
    elif ' '.join(token[:1]) == "motherboards":
        return "motherboards"
    elif ' '.join(token[:2]) == "desktop casings":
        return "desktop casings"
    elif ' '.join(token[:1]) == "processors":
        return "processors"
    elif ' '.join(token[:2]) == "sound cards":
        return "sound cards"
    elif ' '.join(token[:1]) == "others":
        return "others"
    elif ' '.join(token[:2]) == "software operating":
        return "software operating"
    elif ' '.join(token[:1]) == "software":
        return "software"
    elif ' '.join(token[:2]) == "optical drives":
        return "optical drives"
    else:
        return "others"
    
clean_named = []
digit_uom = []

def separateName_Digit_UOM():
    separates = englishDFClean['Name'].apply(lambda x: filterOutNumberUom(x))
    separates = separates.tolist()
    clean_named = []
    digit_uom = []
    for separate in separates:
        a,b = separate
        clean_named.append(a)
        digit_uom.append(b)
    englishDFClean['Cleaned_Only_Name'] = clean_named
    englishDFClean['digit_uom'] =  digit_uom
    


# In[4]:


englishDF = pd.read_csv("Improve.csv")
englishDFClean = pd.read_csv("Improve.csv")

## Text Cleaning
englishDFClean['Cleaned_Product_Specifications'] = englishDFClean['Product Specifications'].apply(lambda x: cleanSpecifications(x))
englishDFClean['Cleaned_Product_Description'] = englishDFClean['Product Description'].apply(lambda x: cleanDescriptions(x))
englishDFClean['Cleaned_Product_Specifications'] = englishDFClean['Cleaned_Product_Specifications'].apply(lambda x: clean_data(x))
englishDFClean['Cleaned_Product_Description'] = englishDFClean['Cleaned_Product_Description'].apply(lambda x:  clean_data(x))
englishDFClean['Cleaned_Spec+Desc'] = englishDFClean['Cleaned_Product_Specifications'] + " " + englishDFClean['Cleaned_Product_Description']

englishDFClean['Cleaned_Name'] = englishDFClean['Name'].apply(lambda x: cleanName(x))
englishDFClean['Cleaned_Name'] = englishDFClean['Cleaned_Name'].apply(lambda x: clean_data(x))

## separate digit and uom
#testing = englishDFClean['Cleaned_Name'].apply(lambda x: filterOutNumberUom(x))
#testing = testing.tolist()
#clean_named = []
#digit_uom = []
#for test in testing:
#    a,b = test
#    clean_named.append(a)
#    digit_uom.append(b)
#englishDFClean['Cleaned_Name'] = clean_named
#englishDFClean['digit_uom'] =  digit_uom
#
separateName_Digit_UOM()
englishDFClean['Cleaned_Name'] = englishDFClean['Cleaned_Name'].apply(lambda x: remove_one_char(x))
englishDFClean['Cleaned_Spec+Desc'] = englishDFClean['Cleaned_Spec+Desc'].apply(lambda x: clean_data(x))
englishDFClean['Cleaned_Only_Specifications'] = englishDFClean['Cleaned_Product_Specifications'].apply(lambda x: removebrand(x))
#englishDFClean['Cleaned_Specifications'] = englishDFClean['Product Specifications'].apply(lambda x: removebrand(x))


# In[5]:


englishDFClean


# In[83]:


englishDFClean[["Name", "Cleaned_Name"]]


# In[84]:


englishDFClean[['Cleaned_Product_Specifications','Cleaned_Product_Description', 'Cleaned_Spec+Desc']]


# In[23]:


for i in englishDFClean['Product Specifications'].head():
    print(i)
    print("=================================")


# In[22]:


for i in englishDFClean['Cleaned_Product_Specifications'].head():
    print(i)
    print("=================================")


# In[7]:


englishDFClean[['Product Specifications', 'Cleaned_Only_Specifications']]


# ## Product Clustering

# In[6]:


print(englishDFClean['Cleaned_Only_Specifications'].unique())


# In[7]:


englishDFClean['Cleaned_Only_Specifications'].value_counts()


# In[64]:


from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

vectorizer_ngram3 = TfidfVectorizer(ngram_range =(1,3))
tfidf_ngram3 = vectorizer_ngram3.fit_transform(englishDFClean['Cleaned_Spec+Desc'])


# In[9]:


print(tfidf_ngram3)
print(tfidf_ngram3.shape)


# In[38]:


tfidfDF = pd.DataFrame(tfidf_ngram3.toarray(), columns=vectorizer_ngram3.get_feature_names())


# In[39]:


tfidfDF


# In[63]:


from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


Sum_of_squared_distances = []
K = range(2,20)
for k in K:
    km = KMeans(n_clusters=k, max_iter=200, n_init=23)
    km = km.fit(tfidf_ngram3)
    Sum_of_squared_distances.append(km.inertia_)
    clusters = km.predict(tfidf_ngram3)
    silhouette_avg = silhouette_score(tfidf_ngram3,clusters)
    print("For n_cluster = ", k, "The average silhouette_score is :", silhouette_avg)

def plot_elbow(K, Sum_of_squared_distances):
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    
plot_elbow(K, Sum_of_squared_distances)


# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud


model_tfidf_ngram3 = KMeans(n_clusters=13, init='k-means++', max_iter=200, n_init=23)
model_tfidf_ngram3.fit(tfidf_ngram3)
labels_tfidf_ngram3=model_tfidf_ngram3.labels_


spec=englishDFClean['Cleaned_Only_Specifications']
wiki_lst=englishDFClean['Cleaned_Spec+Desc']
englishDFClean['Cluster'] = labels_tfidf_ngram3
wiki_cl=pd.DataFrame(list(zip(spec,labels_tfidf_ngram3)),columns=['Cleaned_Specifications','Cluster'])

result={'cluster':labels_tfidf_ngram3,'wiki':wiki_lst}
result=pd.DataFrame(result)

for k in range(0,13):
   s=result[result.cluster==k]
   text=s['wiki'].str.cat(sep=' ')
   text=text.lower()
   text=' '.join([word for word in text.split()])
   wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
   print('Cluster: {}'.format(k))
   print('Titles')
   titles=wiki_cl[wiki_cl.Cluster==k]['Cleaned_Specifications'] 
   print(titles.value_counts())
   #print(titles.to_string(index=False))
   plt.figure()
   plt.imshow(wordcloud, interpolation="bilinear")
   plt.axis("off")
   plt.show()


# In[11]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.patheffects as path_effects
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import sklearn.metrics as metrics
import matplotlib.patheffects as path_effects
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder,OneHotEncoder,MinMaxScaler,StandardScaler
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,classification_report,roc_auc_score,roc_curve, auc
import time
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
from sklearn import svm
from sklearn.inspection import permutation_importance
import time


# In[12]:


import xgboost as xgb
from sklearn import svm


# In[13]:


X = tfidf_ngram3
y = labels_tfidf_ngram3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[14]:


def model_training(classifier,x_train,x_test,y_train,y_test):
    start = time.time()
    classifier.fit(x_train, y_train)
    stop = time.time()
    trainingTime = stop - start
    print(f"Training time: {trainingTime}s")

    # Predicting the test set
    y_pred = classifier.predict(x_test)

    # Making the confusion matrix and calculating accuracy score
    cm = confusion_matrix(y_test, y_pred)
    ac = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,average='micro')
    recall = recall_score(y_test, y_pred,average='micro')
    f1 = f1_score(y_test, y_pred,average='micro')

    acscore.append(ac)
    acprecision.append(precision)
    acrecall.append(recall)
    acF1.append(f1)
    acTrainingTime.append(trainingTime)

    sns.heatmap(cm,annot=True, fmt='d', annot_kws={'fontsize':20}, cmap="YlGnBu");

    print('Accuracy score: {0:0.3f}'.format(ac))
    print('Precision score: {0:0.3f}'.format(precision))
    print('Recall score: {0:0.3f}'.format(recall))
    print('F1 score: {0:0.3f}'.format(f1))
    print('Training Accuracy: ',classifier.score(x_train, y_train))
    print('Testing Accuracy: ',classifier.score(x_test, y_test))
    target_names = ['churn', 'not churn']
    print(classification_report(y_test, y_pred))
    return classifier
    
def model_roc_curve(classifier,x_train,x_test):
    classifier.fit(x_train, y_train)
    predProb = classifier.predict_proba(x_test)
    preds = predProb[:,1]
    fpr, tpr,threshold = metrics.roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    roc_curve(fpr,tpr,roc_auc)
    
def plotEvaluationMetrics(models,y,title):
    plt.rcParams['figure.figsize']=15,8 
    sns.set_style("darkgrid")
    ax = sns.barplot(x=models, y=y, palette = "rocket", saturation =1.5)
    plt.xlabel("Classifier Models", fontsize = 20 )
    plt.ylabel("% of Accuracy", fontsize = 20)
    plt.title(title, fontsize = 20)
    plt.xticks(fontsize = 13, horizontalalignment = 'center', rotation = 0)
    plt.yticks(fontsize = 13)
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        xPlot, yPlot = p.get_xy() 
        ax.annotate(f'{height:.3%}', (xPlot + width/2, yPlot + height*1.02), ha='center', fontsize = 'x-large')
    plt.show()
    
def trainingTimePlot(models,y,title):
    plt.rcParams['figure.figsize']=15,8 
    sns.set_style("darkgrid")
    ax = sns.barplot(x=models, y=acTrainingTime, palette = "rocket", saturation =1.5)
    plt.xlabel("Classifier Models", fontsize = 20 )
    plt.ylabel("Fitting / Training Time", fontsize = 20)
    plt.title("Fitting / Training Time of different Classifier Models", fontsize = 20)
    plt.xticks(fontsize = 13, horizontalalignment = 'center', rotation = 0)
    plt.yticks(fontsize = 13)
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        xPlot, yPlot = p.get_xy() 
        ax.annotate(f'{height:.2}''s', (xPlot + width/2, yPlot + height*1.02), ha='center', fontsize = 'x-large')
    plt.show()


# ## Dirty Train

# In[75]:


acscore = []
acprecision = []
acrecall = []
acF1 = []
acTrainingTime = []


# In[76]:


DTC = DecisionTreeClassifier(random_state=42)
dtc_model = model_training(DTC,X_train, X_test, y_train, y_test)


# In[77]:


RFC = RandomForestClassifier(random_state = 42)
rfc_model = model_training(RFC,X_train, X_test, y_train, y_test)


# In[78]:


m_xgb = xgb.XGBClassifier(random_state = 42)
xgb_model = model_training(m_xgb,X_train, X_test, y_train, y_test)


# In[79]:


from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()
mnb_model = model_training(mnb,X_train, X_test,y_train,y_test)


# In[80]:


from sklearn import svm

SVM = svm.SVC(random_state=42)
svm_model = model_training(SVM,X_train, X_test,y_train,y_test)


# In[81]:


lr = LogisticRegression(random_state=42)
lr_model = model_training(lr,X_train, X_test,y_train,y_test)


# In[82]:


CVMean = []
CVSD = []
models = ["DTC","RDF","XGB","MNB","SVM","LR"]
tupleModel = [dtc_model,rfc_model,xgb_model,mnb_model,svm_model,lr_model]
a = 0
for model in models:
    model = cross_val_score(tupleModel[a],X, y,cv=5,scoring='accuracy')
    print(models[a],":",model)
    CVMean.append(model.mean())
    CVSD.append(model.std())
    a+=1


# In[86]:


plotEvaluationMetrics(models,acscore,"Accuracy Score")


# In[83]:


plotEvaluationMetrics(models,CVMean,"CVMean Score")


# In[84]:


plotEvaluationMetrics(models,acrecall,"Recall Score")


# In[85]:


plotEvaluationMetrics(models,acprecision,"Precision Score")


# In[87]:


plotEvaluationMetrics(models,acF1,"F1 Score")


# In[88]:


trainingTimePlot(models,acTrainingTime,"Training Time")


# ## Grid Search

# In[113]:


from sklearn.model_selection import GridSearchCV

model_params = {
    'DecisionTree':{
        'model':DecisionTreeClassifier(random_state = 42),
        'params':{
            'criterion':['gini','entropy'],
            'max_depth':[1,5,10,15,20,25],
            'max_features':['auto', 'sqrt', 'log2']
        }
    },
    'RandomForest':{
        'model':RandomForestClassifier(random_state = 42),
        'params':{
            'criterion':['gini','entropy'],
            'n_estimators':[10,50,100,150,200],
            'max_depth':[1,5,10,15,20,25],
            'max_features': ['auto', 'sqrt', 'log2']
        }
    },
    'XGBoost':{
        'model':xgb.XGBClassifier(random_state = 42),
        'params':{
            'min_child_weight': [1, 5, 10],
            'gamma': [0.1,0.5, 1, 1.5, 2],
            'subsample': [0.5, 1.0, 1.5,2.0],
            'colsample_bytree': [0.5, 1.0, 1.5,2.0],
            'max_depth': [3, 4, 5]
        }
    },
    'SVM':{
        'model':svm.SVC(random_state = 42, degree=3),
        'params':{
            'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'poly', 'sigmoid','linear']
        }
    },
    'Logistic Regression':{
        'model':LogisticRegression(random_state = 42),
        'params':{
            'solver': ['newton-cg', 'lbfgs', 'liblinear'],
              'penalty': ['l2', 'l1', 'elasticnet', 'none'],
              'C': [0.01,0.1, 1.0,10,100]
        }
    }
}

#x_process_concat = pd.concat([X_train,X_test])
#y_concat =  np.concatenate((y_val, y_test), axis=None)
scores = []
for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5,return_train_score=False)
    clf.fit(X, y)
    scores.append({
        'model':model_name,
        'best_score': clf.best_score_,
        'best_params':clf.best_params_,
        'refit_time':clf.refit_time_
         
    })
        
GridSearchdf = pd.DataFrame(scores, columns=['model','best_score','best_params','refit_time'])
GridSearchdf


# In[114]:


for best_params in GridSearchdf['best_params']:
    print(best_params)


# ## GridSearch CV Train

# In[96]:


acscore = []
acprecision = []
acrecall = []
acF1 = []
acTrainingTime = []


# In[97]:


DTC = DecisionTreeClassifier(random_state=42,criterion = 'gini', max_depth = 25, max_features = 'auto')
dtc_model = model_training(DTC,X_train, X_test, y_train, y_test)


# In[98]:


RFC = RandomForestClassifier(random_state = 42,criterion = 'gini', max_depth = 25, max_features = 'auto', n_estimators = 100)
rfc_model = model_training(RFC,X_train, X_test, y_train, y_test)


# In[99]:


m_xgb = xgb.XGBClassifier(random_state = 42,colsample_bytree=1.0,gamma=0.1,max_depth=5,min_child_weight=1,subsample=0.5)
xgb_model = model_training(m_xgb,X_train, X_test, y_train, y_test)


# In[100]:


from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()
mnb_model = model_training(mnb,X_train, X_test,y_train,y_test)


# In[101]:


from sklearn import svm

SVM = svm.SVC(random_state=42, degree=3, C=10, gamma=1, kernel='sigmoid')
svm_model = model_training(SVM,X_train, X_test,y_train,y_test)


# In[102]:


lr = LogisticRegression(random_state=42, C=0.01, penalty='none', solver='newton-cg')
lr_model = model_training(lr,X_train, X_test,y_train,y_test)


# In[104]:


CVMean = []
CVSD = []
models = ["DTC","RDF","XGB","MNB","SVM","LR"]
tupleModel = [dtc_model,rfc_model,xgb_model,mnb_model,svm_model,lr_model]
a = 0
for model in models:
    model = cross_val_score(tupleModel[a],X, y,cv=5,scoring='accuracy')
    print(models[a],":",model)
    CVMean.append(model.mean())
    CVSD.append(model.std())
    a+=1


# In[105]:


models = ["DTC","RDF","XGB","MNB","SVM","LR"]
plotEvaluationMetrics(models,CVMean,"CVMean Score")


# In[43]:


# Visualising the accuracy score of each classification model
#print(len(models))
#print(len(acscore))
plotEvaluationMetrics(models,acscore,"Accuracy Score")


# In[44]:


# Visualising the accuracy score of each classification model
plotEvaluationMetrics(models,acprecision,"Precision Score")


# In[45]:


# Visualising the accuracy score of each classification model
plotEvaluationMetrics(models,acrecall,"Recall Score")


# In[46]:


# Visualising the accuracy score of each classification model
plotEvaluationMetrics(models,acF1,"F1 Score")


# In[47]:


# Visualising the accuracy score of each classification model
trainingTimePlot(models,acTrainingTime,"Training Time")


# In[61]:


import joblib

def save_model(model, fileName):
    # Save Model to file in the current working directory
    joblib.dump(model, fileName)
    
def load_model(fileName):    
    # Load from file
    joblib_model = joblib.load(fileName)
    print(joblib_model,' ','\n')
    
    return joblib_model


file_name = "xgb_model.pkl"
(save_model(xgb_model,file_name))

file_name = "svm_model.pkl"
save_model(svm_model,file_name)

file_name = "lr_model.pkl"
save_model(lr_model,file_name)

file_name = "vectorizer_ngram3.pkl"
save_model(vectorizer_ngram3,file_name)


# In[56]:


englishDFClean.to_csv(r'Result2.csv', index = False)

