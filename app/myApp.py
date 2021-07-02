import streamlit as st
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
st.set_page_config(layout="wide")

@st.cache
def load_data():
    test = pd.read_csv("/home/princeba/LIVRE_PERSO/01_travailPerso/AI-Engenierie/openclassRoom/projet/projet07/projet_07_model_scoring/app/test.csv")
    train = pd.read_csv("/home/princeba/LIVRE_PERSO/01_travailPerso/AI-Engenierie/openclassRoom/projet/projet07/projet_07_model_scoring/app/Train.csv")
    return test, train

test, train = load_data()



st.title("SCORE CRÉDIT D'UN CLIENT ET VISUALISATION")

st.write("""
# Le score du crédit d'un profil

Cette application permet aux chargé.e.s de clientèle de calculer le score du crédit d'un profil d'un dataset

""")



st.sidebar.header('Selectionner un client')

client = sorted(test['SK_ID_CURR'].unique())
client = ["aucun"]+client

ID = st.sidebar.selectbox('ID du client', client)
if ID != "aucun":
    load_clf = pickle.load(open('/home/princeba/LIVRE_PERSO/01_travailPerso/AI-Engenierie/openclassRoom/projet/projet07/projet_07_model_scoring/app/clf.pkl', 'rb'))

    profil = test.loc[test['SK_ID_CURR']== ID, "CNT_CHILDREN" :]

    prediction = load_clf.predict(profil)
    prediction_proba = load_clf.predict_proba(profil)

    st.subheader('Prediction')
    st.write(prediction)

    st.subheader('Prediction Probability')
    st.write(prediction_proba)

    st.subheader('Le client parmis les clients')
    st.write("Statut du client :", str(train.loc[train['SK_ID_CURR']== ID, "NAME_FAMILY_STATUS"].values))
    st.write("nombre d'enfant :", str(profil['CNT_CHILDREN'].values))

else:
    st.subheader('Dashboard')
    def user_input_features():
        
        salaire = st.sidebar.slider('Salaire', train["AMT_INCOME_TOTAL"].describe()['min'],train["AMT_INCOME_TOTAL"].describe()['75%'],train["AMT_INCOME_TOTAL"].mean())
        ancienete = st.sidebar.slider('Ancienneté', train["DAYS_EMPLOYED"].describe()['min'],train["DAYS_EMPLOYED"].describe()['max'],train["DAYS_EMPLOYED"].astype('float64').mean())
       
        data = {'DAYS_EMPLOYED': ancienete,
                'AMT_INCOME_TOTAL': salaire}
       
        return data

    data = user_input_features()

    clientS = test.loc[(data['AMT_INCOME_TOTAL']<=test['AMT_INCOME_TOTAL']) & (data['DAYS_EMPLOYED']>=test['DAYS_EMPLOYED'])]
    clientDash = clientS.sort_values(by='AMT_INCOME_TOTAL', axis=0, ascending=True)['SK_ID_CURR'].unique()
    ID_2 = st.sidebar.selectbox('ID du client', clientDash)
    profil_2 = test.loc[test['SK_ID_CURR']== ID_2, "CNT_CHILDREN" :]
    st.write("Statut du client :", str(train.loc[train['SK_ID_CURR']== ID_2, "NAME_FAMILY_STATUS"].values))
    st.write("Nombre d'enfant :", str(profil_2['CNT_CHILDREN'].values))
    st.write("Revenu :", str(profil_2['AMT_INCOME_TOTAL'].values))

   




def affichageMoyenne(client="aucun"):
    trainNondef = train[train["TARGET"]==0]
    plt.figure(figsize=(20,10))
    valMaxi = train["AMT_INCOME_TOTAL"].describe()['75%']
    plt.subplot(2, 2, 1)
    train.loc[train["AMT_INCOME_TOTAL"]<valMaxi,"AMT_INCOME_TOTAL"].hist()
    plt.axvline(trainNondef["AMT_INCOME_TOTAL"].mean(), color='red', label="moyenne des client")
    if client != "aucun":
        plt.axvline(profil["AMT_INCOME_TOTAL"].mean(), color='black', label="client")
    plt.legend()
    plt.title("Revenus Total")

    valMaxi = train["AMT_CREDIT"].describe()['max']
    plt.subplot(2, 2, 2)
    train.loc[train["AMT_CREDIT"]<valMaxi,"AMT_CREDIT"].hist()
    plt.axvline(trainNondef["AMT_CREDIT"].mean(), color='red', label="moyenne des client")
    if client != "aucun":
        plt.axvline(profil["AMT_CREDIT"].mean(), color='black', label="client")
    plt.legend()
    plt.title("Montant crédit")

    plt.subplot(2, 2, 3)
    valMaxi = train["DAYS_EMPLOYED"].describe()['max']
    train.loc[train["DAYS_EMPLOYED"]<valMaxi,"DAYS_EMPLOYED"].hist()
    plt.axvline(trainNondef["DAYS_EMPLOYED"].astype('float64').mean(), color='red', label="moyenne des client")
    if client != "aucun":
        plt.axvline(profil["DAYS_EMPLOYED"].mean(), color='black', label="client")
    plt.legend()
    plt.title("Ancieneté")

    plt.subplot(2, 2, 4)
    valMaxi = train["credSURrevenu"].describe()['75%']
    train.loc[train["credSURrevenu"]<valMaxi,"credSURrevenu"].hist()
    plt.axvline(trainNondef["credSURrevenu"].astype('float64').mean(), color='red', label="moyenne des client")
    if client != "aucun":
        plt.axvline(profil["credSURrevenu"].mean(), color='black', label="client")
    plt.legend()
    plt.title("poids du crédit sur le revenus")

    return st.pyplot(plt)

def affichageDash():
    trainNondef = train[train["TARGET"]==0]
    plt.figure(figsize=(20,10))
    valMaxi = clientS["AMT_INCOME_TOTAL"].describe()['75%']
    plt.subplot(1, 2, 1)
    clientS.loc[clientS["AMT_INCOME_TOTAL"]<valMaxi,"AMT_INCOME_TOTAL"].hist()
    plt.axvline(trainNondef["AMT_INCOME_TOTAL"].mean(), color='red', label="moyenne des client")
    
    plt.axvline(profil_2["AMT_INCOME_TOTAL"].mean(), color='black', label="client")
    plt.legend()
    plt.title("Revenus Total")


    plt.subplot(1, 2, 2)
    valMaxi = train["DAYS_EMPLOYED"].describe()['max']
    clientS.loc[clientS["DAYS_EMPLOYED"]<valMaxi,"DAYS_EMPLOYED"].hist()
    plt.axvline(trainNondef["DAYS_EMPLOYED"].astype('float64').mean(), color='red', label="moyenne des client")
   
    plt.axvline(profil_2["DAYS_EMPLOYED"].mean(), color='black', label="client")
    plt.legend()
    plt.title("Ancieneté")



    return st.pyplot(plt)

if ID != "aucun":
    affichageMoyenne(client=ID)

else:
    affichageDash()
