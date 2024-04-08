from extract_face_image.calculate_block_result import Viedeo2img_class,Video_file,Block_img
from sklearn.model_selection import train_test_split
import pandas as pd 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np 
from IPython.display import display


def get_texture(df):
    get_texture_class=Block_img(df,image_size)
    return pd.Series({f'B_GLCM' : get_texture_class.B_input(image_size),
                      f'G_GLCM' : get_texture_class.G_input(image_size),
                      f'R_GLCM' : get_texture_class.R_input(image_size),
                      f'V_GLCM' : get_texture_class.HSV_input(image_size),
                      f'Y_GLCM' : get_texture_class.YCrCb_input(image_size)})


def df_clean(df):
    main_reference_run =df.iloc[:,4].copy()
    for col in ['B','G','R','V','Y']:
        temp = pd.DataFrame(df[f'{col}_GLCM'].tolist(),columns=[f'{col}_GLCM_{i}' for i in range(75)])
        main_reference_run = pd.concat([main_reference_run,temp],axis=1)
    main_reference_run=main_reference_run.loc[~main_reference_run.index.isin([3511, 5541, 6982, 20989, 23887, 23888, 23889])]

    return main_reference_run
    




def model_training(df):
    # split to train and test 
    xtrain ,xtest,ytrain ,ytest= train_test_split(df.drop('label', axis=1),df["label"],test_size = 0.2,random_state=10, stratify=df["label"])
    
    # Create classifiers
    lr = LogisticRegression()
    gnb = GaussianNB()
    svclassifier = SVC(kernel='rbf')
    rfc = RandomForestClassifier()

    clf_list = [
        (lr, "Logistic"),
        (gnb, "Naive Bayes"),
        (svclassifier, "SVC"),
        (rfc, "Random forest"),
    ]


    for i, (clf, name) in enumerate(clf_list):
        print(f'----------------------{name} train result----------------------')
        clf.fit(xtrain,ytrain)
        y_pred = clf.predict(xtrain)
        np.mean((ytrain-y_pred)==0)
        display(pd.DataFrame(confusion_matrix(ytrain,y_pred, labels=np.unique(y_pred)),columns=['Predict Fake','Predict Real'],index=['True Fake','True Real'])) ## row = true  value | column = predict value 
        print(classification_report(ytrain,y_pred))
        
        print(f'----------------------{name} test result----------------------')
        y_pred = clf.predict(xtest)
        display(pd.DataFrame(confusion_matrix(ytest,y_pred, labels=np.unique(y_pred)),columns=['Predict Fake','Predict Real'],index=['True Fake','True Real'])) ## row = true  value | column = predict value 
        print(classification_report(ytest,y_pred))