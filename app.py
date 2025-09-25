from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    iris=load_iris()
    df=pd.DataFrame(iris.data,columns=iris.feature_names)
    df['species']=iris.target
    return df,iris.target_names

df,target_name=load_data()

model=RandomForestClassifier()
model.fit(df.iloc[:,:-1].values,df.iloc[:,-1].values)

st.title("IRIS Dataset Classification")
st.write("Adjust the sepal features and petal features from the side bar and let the model predict the species.")
sl=st.sidebar.slider("Sepal Length",float(df['sepal length (cm)'].min()),float(df['sepal length (cm)'].max()))
sw=st.sidebar.slider("Sepal Width",float(df['sepal width (cm)'].min()),float(df['sepal width (cm)'].max()))
pl=st.sidebar.slider("Petal Length",float(df['petal length (cm)'].min()),float(df['petal length (cm)'].max()))
pw=st.sidebar.slider("Petal Width",float(df['petal width (cm)'].min()),float(df['petal width (cm)'].max()))

input_data=[[sl,sw,pl,pw]]


y_pred=model.predict(input_data)

spec=target_name[y_pred[0]]


st.subheader(f"Predicted Species is:{spec}")


footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with 💖 by <a style='display: block; text-align: center;' href="https://drive.google.com/file/d/1OJr-X3b18ZBHCGc8W0KnpBn-mbJG9J5K/view?usp=sharing" target="_blank">Ayush Shukla</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
