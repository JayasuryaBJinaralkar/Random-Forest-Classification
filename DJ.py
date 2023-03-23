import streamlit as st
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
def label_encode(df, cols):
    le = LabelEncoder()
    for col in cols:
        df[col] = le.fit_transform(df[col])
    return df

def onehot_encode(df, cols):
    ohe = OneHotEncoder()
    for col in cols:
        encoded = pd.DataFrame(ohe.fit_transform(df[[col]]).toarray(), columns=[f'{col}_{i}' for i in range(ohe.categories_[0].shape[0])])
        df = pd.concat([df, encoded], axis=1)
        df.drop(col, axis=1, inplace=True)
    return df
st.title('Model Deployment')
st.image("https://www.analyticsinsight.net/wp-content/uploads/2021/02/ML-Frameworks-1024x576.jpg")

def main():
    st.title("Upload a Dataset")

    # Create a file uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the dataset
        data = pd.read_csv(uploaded_file)

        # Display the raw data
        st.subheader('Raw Data')
        st.write(data)

# Create a checkbox for data preprocessing
    show_preprocessing = st.checkbox("Data Preprocessing")

    # Create a checkbox for data transformation
    show_transformation = st.checkbox("Data Transformation")

    # Create a checkbox for data visualization
    show_visualization = st.checkbox("Data Visualization")

    # Create a checkbox for model building
    show_model_building = st.checkbox("Model Building")
    

    # Display the selected sections
    if show_preprocessing:
        # Add code for data preprocessing section here
        st.write("Data preprocessing section")
        # Display null values
        if st.checkbox('Display Null Values'):
                st.write(data.isnull().sum())
                # Drop null values
                if st.checkbox('Drop Null Values'):
                    data.dropna(inplace=True)
                    st.write('Null values dropped')
                    
     # Check for outliers
    if st.checkbox('Check for outliers'):
            st.write('## Outliers')
            sns.boxplot(x=data['Tenure'])
            st.pyplot()   
   

     # Remove outliers
    if st.checkbox('Remove outliers'):
         Q1 = data['Tenure'].quantile(0.25)
         Q3 = data['Tenure'].quantile(0.75)
         IQR = Q3 - Q1
         df = data[(data['Tenure'] >= Q1 - 1.5*IQR) & (data['Tenure'] <= Q3 + 1.5*IQR)]

         st.success('Outliers successfully Removed!')
           
    if show_transformation:
        # Add code for data transformation section here
        st.write("Data transformation section")
        # Encode categorical variables
        if st.checkbox('Encode Categorical Variables'):
                categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
                encoding_method = st.radio('Select an encoding method', ('Label Encoding', 'One-Hot Encoding'))
                if encoding_method == 'Label Encoding':
                    label_encoder = LabelEncoder()
                    for col in categorical_cols:
                        data[col] = label_encoder.fit_transform(data[col].astype(str))
                    st.write('Label encoding completed')
                elif encoding_method == 'One-Hot Encoding':
                    onehot_encoder = OneHotEncoder()
                    encoded_cols = pd.DataFrame(onehot_encoder.fit_transform(data[categorical_cols]).toarray())
                    data = pd.concat([data, encoded_cols], axis=1)
                    data.drop(categorical_cols, axis=1, inplace=True)
                    st.write('One-hot encoding completed')

    if show_visualization:
        # Add code for data visualization section here
        st.write("Data visualization section")
        plot_options = st.multiselect("Select the plot(s) to display", ( 'Scatter Plot', 'Heatmap'))
    
        if 'Scatter Plot' in plot_options:
            st.write("Scatter Plot")
            x_axis = st.selectbox("Select the feature for x-axis", data.columns)
            y_axis = st.selectbox("Select the feature for y-axis", data.columns)
            plt.figure(figsize=(12,6))
            sns.scatterplot(x=data[x_axis], y=data[y_axis])
            st.pyplot()
        
        if 'Heatmap' in plot_options:
            st.write("Heatmap")
            plt.figure(figsize=(12,6))
            sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
            st.pyplot()

    if show_model_building:
        # Add code for model building section here
        st.write("Model building section")
        st.subheader("Model Building")
        target = st.selectbox("Select the target variable", data.columns)
        X = data.drop(columns=[target])
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        st.write("Model trained successfully")

        # Model Evaluation
        st.subheader("Model Evaluation")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write("Accuracy:", accuracy)


if __name__ == '__main__':
    main()