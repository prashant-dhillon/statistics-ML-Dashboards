# Author: Prashant DHILLON

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics

df=pd.read_csv("housing.csv")
df_describe = df.describe()
df_corr = df.corr()
df_corr = df_corr.style.background_gradient(cmap= 'inferno').set_precision(4)

def removeOutliers(dataframe,column):
        column = "total_rooms" 
        des = dataframe[column].describe()
        desPairs = {"count":0,"mean":1,"std":2,"min":3,"25":4,"50":5,"75":6,"max":7}
        Q1 = des[desPairs['25']]
        Q3 = des[desPairs['75']]
        IQR = Q3-Q1
        lowerBound = Q1-1.5*IQR
        upperBound = Q3+1.5*IQR
        print("(IQR = {})Outlier are anything outside this range: ({},{})".format(IQR,lowerBound,upperBound))
        data = dataframe[(dataframe [column] < lowerBound) | (dataframe [column] > upperBound)]

        print("Outliers out of total = {} are \n {}".format(df[column].size,len(data[column])))
        outlierRemoved = df[~df[column].isin(data[column])]
        return outlierRemoved

def linear_regression():
    st.markdown("### Performing Linear Regression with Statistics")

    """
    LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize 
        the residual sum of squares between the observed targets in the dataset, and the targets 
            predicted by the linear approximation.
    """

    df_outliersRemoved = removeOutliers(df,"total_rooms")    
    labelEncoder = LabelEncoder()
    imputer = SimpleImputer(np.nan,strategy ="median")
    imputer.fit(df.iloc[:,4:5])
    df.iloc[:,4:5] = imputer.transform(df.iloc[:,4:5])
    df["ocean_proximity"] = labelEncoder.fit_transform(df["ocean_proximity"])
    df_ind = df.drop("median_house_value",axis=1)
    df_dep = df["median_house_value"]
    X_train,X_test,y_train,y_test = train_test_split(df_ind,df_dep,test_size=0.2,random_state=42)
    independent_scaler = StandardScaler()
    X_train = independent_scaler.fit_transform(X_train)
    X_test = independent_scaler.transform(X_test)
    #initantiate the linear regression
    linearRegModel = LinearRegression(n_jobs=-1)
    #fit the model to the training data (learn the coefficients)
    linearRegModel.fit(X_train,y_train)
    st.markdown("#### Calculating the intercept and coefficients")
    st.write("Intercept is "+str(linearRegModel.intercept_))
    st.write("coefficients  are "+str(linearRegModel.coef_))
    #predict on the test data
    st.markdown("#### Prediction on test data")
    y_pred = linearRegModel.predict(X_test)
    test = pd.DataFrame({'Predicted':y_pred,'Actual':y_test})
    fig= plt.figure(figsize=(16,8))
    test = test.reset_index()
    test = test.drop(['index'],axis=1)
    plt.plot(test[:50])
    plt.legend(['Actual','Predicted'])
    plt.title("comparison between actual and predicted values")
    st.pyplot(plt)
    plt.clf()

    st.write('Linear Regression Model Score: {}'.format(linearRegModel.score(X_train,y_train)))

    st.markdown("##### How the score is calculated")

    """
    The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares ((y_true - y_pred) *^ 2).sum() and v is the total sum of squares ((y_true - y_true.mean()) *^ 2).sum(). The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.

    We calculate the score between X_train and y_train because the library internally uses lr.predict(X_train) to get y_train'
    """
    
def random_forest_regression():
    st.markdown("### Performing Random Forest Regression with Statistics")

    df_outliersRemoved = removeOutliers(df,"total_rooms")    
    labelEncoder = LabelEncoder()
    imputer = SimpleImputer(np.nan,strategy ="median")
    imputer.fit(df.iloc[:,4:5])
    df.iloc[:,4:5] = imputer.transform(df.iloc[:,4:5])
    df["ocean_proximity"] = labelEncoder.fit_transform(df["ocean_proximity"])
    df_ind = df.drop("median_house_value",axis=1)
    df_dep = df["median_house_value"]
    X_train,X_test,y_train,y_test = train_test_split(df_ind,df_dep,test_size=0.2,random_state=42)
    independent_scaler = StandardScaler()
    X_train = independent_scaler.fit_transform(X_train)
    X_test = independent_scaler.transform(X_test)
    rfReg = RandomForestRegressor(30)
    rfReg.fit(X_train,y_train)

    st.markdown("#### Prediction on test data")
    rfReg_y_pred = rfReg.predict(X_test)
    test = pd.DataFrame({'Predicted':rfReg_y_pred,'Actual':y_test})
    fig= plt.figure(figsize=(16,8))
    test = test.reset_index()
    test = test.drop(['index'],axis=1)
    plt.plot(test[:50])
    plt.legend(['Actual','Predicted'])
    plt.title("comparison between actual and predicted values")
    st.pyplot(plt)
    plt.clf()

    st.write('Random Forest Model Score: {}'.format(rfReg.score(X_train,y_train)))
    st.markdown("##### How the score is calculated")

    """
    The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares ((y_true - y_pred)*^2).sum() and v is the total sum of squares ((y_true - y_true.mean()) *^ 2).sum(). The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.

    We calculate the score between X_train and y_train because the library internally uses lr.predict(X_train) to get y_train'
    """

def disp():
    st.markdown("# California Housing Price: statistical learning Project")
    st.markdown("## The aim of this project is to Perform Exploratory data analysis and predict median housing prices in California")
    st.markdown("## You can start by selecting a option in sidebar")
    st.markdown("### Developed by Prashant Dhillon")

def EDA():
    st.markdown("# Exploratory data analysis")

    """
    Below we will perform some Exploratory Data Analysis to do some initial investigations on our dataset, 
        so that we can discover patterns,to spot anomalies and to check 
            assumptions with the help of summary statistics and graphical representations.
    """
    
    st.markdown("### Stats about our dataset")

    st.dataframe(df_describe)

    # Plotting against categorical variable 
    st.markdown("### To visualize scatter plot of the features and labels which you are interested in")

    cat = st.multiselect('Show house per category?', df['ocean_proximity'].unique())
    col1 = st.selectbox('Which feature on x?', df.columns[0:9])
    col2 = st.selectbox('Which feature on y?', df.columns[0:9])
    if not cat:
        cat = ['NEAR OCEAN']
    else:
        pass
    new_df = df[(df['ocean_proximity'].isin(cat))]
    fig3 = px.scatter(new_df, x =col1,y=col2, color='ocean_proximity')
    
    st.plotly_chart(fig3)

    
    st.markdown("### Histogram : Representation of the distribution of data")
    feature = st.selectbox('Which feature you want to plot?', df.columns[0:9])
    # Filter dataframe
    new_df2 = df[feature]
    fig2 = px.histogram(new_df2, marginal="rug")
    st.plotly_chart(fig2)


    st.markdown("### Correrlation Matrix")

    """
    Correlation is a measure of the strength of a linear relationship between two quantitative variables. 
    Below two terms are used to inference the results positive correlation and negative correlation. 

    -> Positive correlation is a relationship between two variables in which both variables move in the same direction. 
        This is when one variable increases while the other increases and visa versa.
    """
    
    "-> Dark shades represents positive correlation while lighter shades represents negative correlation."
    st.write(df_corr)

    # Geographical data Plotting
    st.markdown("### House price based of geographical co-ordinates")
    plt.figure(figsize=(10,5))
    plt.scatter(df["longitude"],df["latitude"],c=df["median_house_value"], 
                s=df["population"]/50, alpha=0.2, cmap="Oranges")
    plt.colorbar()
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("House price based of geographical co-ordinates")
    st.pyplot(plt)
    plt.clf()

    st.markdown("### Barplot of categorical column")
    plt.figure(figsize=(7,4))
    sns.countplot(data=df,x='ocean_proximity', palette = "inferno")
    st.pyplot(plt)
    plt.clf()



def main():
    
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["", "Display the DataSet", "Perform Exploratory data analysis", "Apply Machine Learning"])
    if app_mode == "":
        disp() 
    elif app_mode == "Display the DataSet":
        st.markdown("# Dataset")
        """
        The data contains 20,640 observations on 10 variables. 
        This dataset contains the average house value as target variable and the following input variables (features): 
            average income, housing average age, average rooms, average bedrooms, population, average occupation, latitude, longitude and ocean proximity
        """
        st.dataframe(df)
    elif app_mode == "Perform Exploratory data analysis":
        EDA()
    elif app_mode == "Apply Machine Learning":
        st.markdown("## Applying Machine Learning on dataset")
        alg = ['None', 'Linear Regression', 'Random Forest Regression']
        reg = st.selectbox('Which algorithm you want to apply?', alg)
        if reg == 'None':
            "Please select one of the two algorithms to see results"
            pass
        elif reg=='Linear Regression':
            linear_regression()
        elif reg == 'Random Forest Regression':
            random_forest_regression()
    

if __name__ == "__main__":
    main()
