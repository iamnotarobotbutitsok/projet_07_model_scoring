import streamlit as st
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("SCORE CRÉDIT D'UN CLIEN")

st.write("""
# Le score du crédit d'un profil

Cette application permet aux chargé.e.s de clientèle de calculer le score du crédit d'un profil d'un dataset

""")

test = pd.read_csv('test.csv')
train = pd.read_csv("Train.csv")

st.sidebar.header('Selectionner un client')

client = sorted(test['SK_ID_CURR'].unique())

ID = st.sidebar.selectbox('ID du client', client)

load_clf = pickle.load(open('clf.pkl', 'rb'))

profil = test.loc[test['SK_ID_CURR']== ID, "CNT_CHILDREN" :]

prediction = load_clf.predict(profil)
prediction_proba = load_clf.predict_proba(profil)

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.subheader('Le client parmis les clients')

def affichageMoyenne(i=0):
    trainNondef = train[train["TARGET"]==0]
    plt.figure(figsize=(20,10))
    valMaxi = train["AMT_INCOME_TOTAL"].describe()['75%']
    plt.subplot(2, 2, 1)
    train.loc[train["AMT_INCOME_TOTAL"]<valMaxi,"AMT_INCOME_TOTAL"].hist()
    plt.axvline(trainNondef["AMT_INCOME_TOTAL"].mean(), color='red', label="moyenne des client")
    plt.axvline(profil["AMT_INCOME_TOTAL"].mean(), color='black', label="client")
    plt.legend()
    plt.title("Revenus Total")

    valMaxi = train["AMT_CREDIT"].describe()['max']
    plt.subplot(2, 2, 2)
    train.loc[train["AMT_CREDIT"]<valMaxi,"AMT_CREDIT"].hist()
    plt.axvline(trainNondef["AMT_CREDIT"].mean(), color='red', label="moyenne des client")
    plt.axvline(profil["AMT_CREDIT"].mean(), color='black', label="client")
    plt.legend()
    plt.title("Montant crédit")

    plt.subplot(2, 2, 3)
    valMaxi = train["DAYS_EMPLOYED"].describe()['max']
    train.loc[train["DAYS_EMPLOYED"]<valMaxi,"DAYS_EMPLOYED"].hist()
    plt.axvline(trainNondef["DAYS_EMPLOYED"].astype('float64').mean(), color='red', label="moyenne des client")
    plt.axvline(profil["DAYS_EMPLOYED"].mean(), color='black', label="client")
    plt.legend()
    plt.title("Ancieneté")

    plt.subplot(2, 2, 4)
    valMaxi = train["credSURrevenu"].describe()['75%']
    train.loc[train["credSURrevenu"]<valMaxi,"credSURrevenu"].hist()
    plt.axvline(trainNondef["credSURrevenu"].astype('float64').mean(), color='red', label="moyenne des client")
    plt.axvline(profil["credSURrevenu"].mean(), color='black', label="client")
    plt.legend()
    plt.title("poids du crédit sur le revenus")

    return st.pyplot(plt)


affichageMoyenne(i=0)
