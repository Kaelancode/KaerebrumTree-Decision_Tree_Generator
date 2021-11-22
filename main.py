import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
#from KaelanML.mlcode.RF.KaetreeStreamlit import DecisionTreeClassifier as DT
from KaetreeStreamlit import DecisionTreeClassifier as DT
from graphviz import Digraph
import base64
import io
import re
import uuid
import numpy as np

st.set_page_config(layout="wide")
st.title('Kaerebrum Tree Code Generator (Classifier)')
st.markdown('- This app uses the Kaerebrum decision tree alogrithm for machine learning')
st.markdown('- It trains on the input data to match the given classification and build decision rules to make predictions for new data')
st.markdown('- The alogrithm is developed(Project Kaerebrum) to be able to train on categorical as well as continous data')
st.markdown('- The aglorithm other features include code generator base on the decision rules built during the training as well as informative material design tree maps. ')

st.sidebar.markdown(' **Select from Sample Data or Upload csv file**')
sample_data = st.sidebar.selectbox('1) Sample_data ',('Iris', 'Titanic', 'Breast Cancer'))
st.sidebar.warning("Please ensure targets/outputs to be at the last column and data in each column is of the same data type. In case of error please check dataset for nan or null values")
uploaded_file = st.sidebar.file_uploader("2) Upload CSV")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    sample_data = 'Uploaded'

selected_data = datasets.load_breast_cancer() if sample_data == 'Breast Cancer' else pd.read_csv(r'IRIS.csv') if sample_data == 'Iris' else pd.read_csv(r'trainst.csv') if sample_data == 'Titanic' else df

X = pd.DataFrame(selected_data.data) if sample_data == "Breast Cancer" else selected_data.iloc[:, 0:-1]
y = pd.DataFrame(selected_data.target) if sample_data == "Breast Cancer" else selected_data.iloc[:, -1]

st.info('Data is split into 80% for training and 20% for testing')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

st.markdown('## Preview of Training Data ')
col1, col2 = st.columns(2)
col1.text('Features/Inputs')
col1.dataframe(X_train)
col2.text('Targets/Outputs')
col2.dataframe(y_train)
st.write(f'Rows: {X_train.shape[0]}| Columns: {X_train.shape[1]}')

st.markdown('## Preview of Testing Data ')
colt1, colt2 = st.columns(2)
colt1.text('Features/Inputs')
colt1.dataframe(X_test)
colt2.text('Targets/Outputs')
colt2.dataframe(y_test)
st.write(f'Rows: {X_test.shape[0]}| Columns: {X_test.shape[1]}')

st.markdown('## Tune parameters')
param_col1, param_col2, param_col3 = st.columns(3)

depth = param_col1.text_input('Max Depth of Tree',  10)
max_features = param_col1.selectbox('No of features considered for each split', (reversed(list(range(1, X.shape[1]+1)))))
Measure = param_col1.selectbox('Measure information', ('gini', 'entropy'))

Min_samples = param_col2.text_input('Min samples to split',  2)
Min_leaf = param_col2.text_input('Min samples in each leaf',  1)

Max_leaf_nodes = param_col3.text_input('Max leaf nodes (0 for no limit)', 0)
Split = param_col3.selectbox('Select which feature to evaluate base on best split citeria or random selection', ('best', 'rand'))

st.markdown('## Setup Tree Map Ouput')
tree_col1, tree_col2, tree_col3 = st.columns(3)
eng = tree_col1.selectbox('Layout engine of Tree', ('dot', 'neato', 'twopi', 'circo', 'sfdp'))
resolut = tree_col2.selectbox('Resolution of image', (96, 128, 192, 256, 300))
img_type = tree_col3.selectbox('Output file format', ('png', 'pdf', 'jpg'))

TreeGen = DT(min_samples=int(Min_samples), max_depth=int(depth), min_leaf= int(Min_leaf), n_feats= int(max_features), max_leaf_nodes=int(Max_leaf_nodes), split=Split, measure=Measure)

trained = False
 #st.button('Train', key=None, help=None, on_click=None, args=None, kwargs=None)
with st.spinner('Training...'):

    if st.button('Train'):
        TreeGen.train(X_train, y_train)
        trained = True
        #b = a.__str__()
        #st.graphviz_chart(a)
        #st.download_button(label='Download', data=r'my_graph.png', file_name='img.png')
        if trained:
            dot = TreeGen.tree.view(engine=eng, res=resolut, output=img_type)
            st.success('Done!')

            st.markdown('## Validation')
            predictions = TreeGen.predict(X_test)
            predictionsDF = pd.DataFrame(predictions, columns=['Predictions'])
            compare1, compare2, compare3 = st.columns(3)
            compare2.dataframe(predictionsDF)
            compare3.dataframe(y_test.reset_index())
            compare1.markdown('#### Results')
            compare1.write(f'Accuracy: {(sum(predictions == y_test.values.flatten()) / y_test.values.shape[0])*100} %')

            st.markdown('## Tree Map')
            st.info('Leaves shade changes according to the purity of the node: "deep green": 0, green: < 0.1 , light green: <0.3, yellow <0.7 and orange >0.7')
            ftype = 'png' if img_type == 'png' else 'pdf' if img_type == 'pdf' else 'jpg'
            source = 'my_graph.'+ftype
            st.image(source, width=None, output_format=ftype)


            button_uuid = str(uuid.uuid4()).replace("-", "")
            button_id = re.sub(r"\d+", "", button_uuid)
            custom_css = f"""
            <style>
                #{button_id} {{
                    display: inline-flex;
                    align-items: center;
                    justify-content: left;
                    background-color: rgb(119, 198, 110);
                    color: rgb(38, 39, 48);
                    padding: .25rem .75rem;
                    position: relative;
                    text-decoration: none;
                    border-radius: 4px;
                    border-width: 1px;
                    border-style: solid;
                    border-color: rgb(230, 234, 241);
                    border-image: initial;
                }}
                #{button_id}:hover {{
                    border-color: rgb(246, 51, 102);
                    color: rgb(246, 51, 102);
                }}
                #{button_id}:active {{
                    box-shadow: none;
                    background-color: rgb(246, 51, 102);
                    color: white;
                    }}
            </style> """
            st.sidebar.text_area('Code', TreeGen.tree.code())
            button_text = 'Code Download'
            text_file = open("code.txt", "w")
            text_file.write(TreeGen.tree.code())
            text_file.close()
            with open("code.txt", "rb") as file:
                bufcod = file.read()
            b64cod = base64.b64encode(bufcod).decode()
            href = f'<a download="code.txt" id="{button_id}" href="data:file/txt;base64,{b64cod}">{button_text}</a><br><br>'
            st.sidebar.markdown(f"<p style='text-align: centre;;'> {custom_css + href} </p>", unsafe_allow_html=True)

            with open(source, "rb") as file:
                buf = file.read()
            b64 = base64.b64encode(buf).decode()

            download_filename = source
            button_text = 'Tree Map Download'
            href = f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br><br>'
            st.markdown(f"<p style='text-align: left;;'> {custom_css + href} </p>", unsafe_allow_html=True)

            st.markdown('#### Report')
            my_expander = st.expander('Click to view')
            rec = np.atleast_2d(TreeGen.record)
            my_expander.text(f'Total nodes: {TreeGen._node_count}, Total split nodes: {sum(rec[:,1])}, Total leaf nodes: {sum(rec[:, 2])}')
            my_expander.dataframe(TreeGen.records())
            my_expander.write('Cont: Continous data | Cat: Categorical data')
            my_expander.dataframe(TreeGen.legend())
            my_expander.write('Feature importance features in descending order')
            my_expander.dataframe(TreeGen.top_features(1))






                # The rest of the fields
