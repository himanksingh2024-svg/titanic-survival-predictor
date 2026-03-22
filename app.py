import streamlit as st
import pickle
import numpy as np

with open('titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('🚢 Titanic Survival Predictor')
st.write('Would you have survived the Titanic?')

pclass = st.selectbox('Passenger Class', [1, 2, 3])
sex = st.selectbox('Sex', ['Male', 'Female'])
age = st.slider('Age', 1, 80, 25)
sibsp = st.number_input('Siblings/Spouses aboard', 0, 8, 0)
parch = st.number_input('Parents/Children aboard', 0, 6, 0)
fare = st.number_input('Fare paid', 0.0, 520.0, 32.0)
embarked = st.selectbox('Port of Embarkation', ['Southampton', 'Cherbourg', 'Queenstown'])
title = st.selectbox('Title', ['Mr', 'Mrs', 'Miss', 'Master', 'Rare'])

sex = 0 if sex == 'Male' else 1
embarked = {'Southampton': 0, 'Cherbourg': 1, 'Queenstown': 2}[embarked]
title = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}[title]
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

if st.button('Predict!'):
    features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked, family_size, is_alone, title]])
    result = model.predict(features)[0]
    if result == 1:
        st.success('✅ You would have SURVIVED!')
    else:
        st.error('❌ You would NOT have survived.')
