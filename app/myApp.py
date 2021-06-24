import streamlit as st
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb


st.title("SCORE CRÉDIT D'UN CLIEN")

st.write("""
# Le score du crédit d'un profil

Cette application permet aux chargé.e.s de clientèle de calculer le score du crédit d'un profil d'un dataset

""")

test = pd.read_csv('test.csv')

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
