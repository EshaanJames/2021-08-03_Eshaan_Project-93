import body as body
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)

df.head()
df = df.dropna()
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})
df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen': 2})

X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

svc_model = SVC(kernel='linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

rf_clf = RandomForestClassifier(n_jobs=-1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)


@st.cache()
def prediction(model, island, bill_depth_mm, bill_length_mm, flipper_length_mm, body_mass_g, sex):
    predict_species = model.predict([[island, bill_depth_mm, bill_length_mm, flipper_length_mm, body_mass_g, sex]])
    species = predict_species[0]
    if species == 0:
        return "Adelie"
    elif species == 1:
        return "Chinstrap"
    else:
        return 'Gentoo'


st.title("Penguin Classification web app.")
st.sidebar.title("Select the parameters values here.")
bill_length = st.sidebar.slider("bill_length_mm", float(df['bill_length_mm'].min()),
                                float(df['bill_length_mm'].max()))
bill_depth = st.sidebar.slider("bill_depth_mm", float(df['bill_depth_mm'].min()), float(df['bill_depth_mm'].max()))
flipper_length = st.sidebar.slider("flipper_length_mm", float(df['flipper_length_mm'].min()),
                                   float(df['flipper_length_mm'].max()))
body_mass = st.sidebar.slider("body_mass_g", float(df["body_mass_g"].min()), float(df['body_mass_g'].max()))
sex_value = st.selectbox("Sex", ["Male", "Female"])
if sex_value == "Male":
    sex_value = 0
else:
    sex_value = 1

island_value = st.selectbox("island", ['Biscoe', 'Dream', 'Torgersen'])
if island_value == "Biscoe":
    island_value = 0
elif island_value == "Dream":
    island_value = 1
else:
    island_value = 2
classifier = st.sidebar.selectbox("Classifier",
                                  ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))

if st.sidebar.button("Predict"):
    if classifier == 'Support Vector Machine':
        species_type = prediction(svc_model, island_value, bill_depth, bill_length, flipper_length, body_mass,
                                  sex_value)
        score = svc_model.score(X_train, y_train)

    elif classifier == 'Logistic Regression':
        species_type = prediction(log_reg, island_value, bill_depth, bill_length, flipper_length, body_mass, sex_value)
        score = log_reg.score(X_train, y_train)

    else:
        species_type = prediction(rf_clf, island_value, bill_depth, bill_length, flipper_length, body_mass, sex_value)
        score = rf_clf.score(X_train, y_train)

    st.write("Species predicted:", species_type)
    st.write("Accuracy score of this model is:", score)
