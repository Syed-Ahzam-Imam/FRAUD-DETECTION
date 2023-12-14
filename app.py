from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
import pickle
from imblearn.over_sampling import SMOTE
import numpy as np # linear algebra
import pandas as pd
import scipy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import warnings     # for supressing a warning when importing large files
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder , OrdinalEncoder , LabelEncoder
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

app = Flask(__name__)

# Load the trained ExtraTreesClassifier model
with open('Fraud-Detection.pkl', 'rb') as model_file:
    extra_trees_model = pickle.load(model_file)

with open('RandomForestClassifier.pkl', 'rb') as model_file:
    RandomForestClassifier = pickle.load(model_file)
with open('DecisionTreeClassifier.pkl', 'rb') as model_file:
    decision_tree_model = pickle.load(model_file)



with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


def physician_same(row):
    atten_oper=row["AttendingPhysician"]==row["OperatingPhysician"]
    oper_other=row["OperatingPhysician"]==row["OtherPhysician"]
    atten_other=row["AttendingPhysician"]==row["OtherPhysician"]
    if atten_oper==True and oper_other==True:# atten = oper = other
        return 0
    elif atten_oper==True and oper_other==False:# atten = oper != other
        return 1
    elif atten_oper==False and oper_other==True:# atten != oper = other
        return 2
    else:# atten != oper != other
        return 3
    
    
def physician_count(row,list_count):
    count=0
    for col in list_count:
        if pd.isnull(row[col]):
            continue
        else:
            count+=1
    return count

# check whether the patient dead or alife
def alife_function(value):
    if value==True:
        return 1
    else:
        return 0

    

def groupby(df,by,vars_to_group,methods,col_ident,as_index=True,agg=False):
    if agg:
        grouped=df.groupby(by=by,as_index=as_index)[vars_to_group].agg(methods)
        cols=['_'.join(col) for col in grouped.columns.values]
        cols=[col_ident+"_"+col for col in cols]
        grouped.columns=cols
        return grouped

    else:
        concat=df.groupby(by=by,as_index=as_index)[vars_to_group].transform(methods[0])
        cols=[ col_ident+"_"+col+"_"+methods[0] for col in concat.columns ]
        concat.columns=cols

        for method in methods[1:]:
            grouped=df.groupby(by=by,as_index=as_index)[vars_to_group].transform(method)
            cols=[col_ident+"_"+col+"_"+method for col in grouped.columns]
            grouped.columns=cols
            concat=pd.concat([concat,grouped],axis=1)

        return concat


def preprocess_input_data(input_df):
    # Convert the input data to a DataFrame
    # input_df = pd.DataFrame([input_data[0]])
    # input_df = pd.DataFrame(input_df)
    # input_df = input_df.fillna(0)
    # if input_df.isnull().values.any():
            
    # Add the physician_same column
    
    phy_same=input_df.apply(physician_same,axis=1)
    input_df["phy_same"]=phy_same
    list_count=["AttendingPhysician","OperatingPhysician","OtherPhysician"]
    
    phy_count=input_df.apply(physician_count,axis=1,args=(list_count,))
    input_df["phy_count"]=phy_count
    startdate= pd.to_datetime( input_df["ClaimStartDt"] )
    enddate= pd.to_datetime( input_df["ClaimEndDt"] )

    period = ( enddate - startdate).dt.days
    input_df["period"] = period

    copy = input_df.copy()

    cronic_cols_names=copy.columns[ copy.columns.str.startswith("ChronicCond") ]
    cronic_cols=copy[   cronic_cols_names   ]
    cronic=cronic_cols.replace({2:0})
    copy[   cronic_cols_names   ]=cronic
    copy["PotentialFraud"]=copy["PotentialFraud"].replace({"Yes":1,"No":0})
    copy["Gender"]=copy["Gender"].replace({2:0})

    startadmt= pd.to_datetime( copy["AdmissionDt"] )
    enddatadmt= pd.to_datetime( copy["DischargeDt"] )
    periodadmt = ( enddatadmt - startadmt).dt.days
    copy["periodadmt"] = periodadmt
    copy["periodadmt"]= copy["periodadmt"].fillna(0)
    copy["RenalDiseaseIndicator"]=copy["RenalDiseaseIndicator"].replace({"Y":1})
    birthdate=pd.to_datetime(copy["DOB"])
    enddate=pd.to_datetime(copy["DOD"])
    alife = pd.isna(enddate).apply(alife_function)

    max_date=enddate.dropna().max()
    enddate[pd.isna(enddate)]=max_date
    period=(((enddate-birthdate).dt.days/356).astype(int))

    copy["age"]=period
    copy["alife"]=alife

    money_cols=["InscClaimAmtReimbursed","DeductibleAmtPaid","NoOfMonths_PartACov","NoOfMonths_PartBCov",
           "IPAnnualReimbursementAmt","IPAnnualDeductibleAmt","OPAnnualReimbursementAmt","OPAnnualDeductibleAmt"]

    provider_money=groupby(copy,["Provider"],money_cols,["mean","std"],"provider",
                       True,False)

    banel_money=groupby(copy,["BeneID"],money_cols,["mean","std"],"banel",
                       True,False)
    
    diag1_money=groupby(copy,["ClmDiagnosisCode_1"],money_cols,["mean","std"],"diag1",
                       True,False)
    

    selected_cols_names=["phy_same","phy_count","period","periodadmt","age","alife","Provider","PotentialFraud"]
    selected_cols=copy[selected_cols_names]

    data=pd.concat([selected_cols,provider_money,banel_money,diag1_money],axis=1)

    grouped=data.groupby(by=["Provider","PotentialFraud"]).agg("mean").reset_index()

    grouped=grouped.fillna(0)
    
    features=grouped.iloc[:,2:]
    labels=grouped.iloc[:,1]

    features_stand = scaler.transform(features)
    print(features_stand)
    
    # xtrain,xtest,ytrain,ytest = train_test_split(featuress,labelss)
    featuress = features_stand.astype(np.float32)

    return featuress

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict-api', methods=['POST'])
def predictapi():
    try:
        # Get form data from the request
        form_data = request.form.to_dict()

        # Create a DataFrame from the form data
        input_data = pd.DataFrame({key: [value] for key, value in form_data.items()})
        print(input_data)
        # Preprocess the input data
        preprocessed_input = preprocess_input_data(input_data)

        # Make predictions using the ExtraTreesClassifier
        predictions = extra_trees_model.predict(preprocessed_input)

        predictions_list = predictions.tolist()

        # Interpret predictions
        fraud_status = "Fraud" if 1 in predictions_list else "Not Fraud"
        print(fraud_status)
        return render_template('index.html')

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read the CSV file
        input_data = pd.read_csv("Combined Dataset.csv")

        # Select the first and second rows
        input_data_subset = input_data.head(1)
        print(input_data_subset)

        # Preprocess the input data
        preprocessed_input = preprocess_input_data(input_data_subset)

        # Make predictions using the three classifiers
        extra_trees_predictions = extra_trees_model.predict(preprocessed_input)
        random_forest_predictions = RandomForestClassifier.predict(preprocessed_input)
        decision_tree_predictions = decision_tree_model.predict(preprocessed_input)

        # Count occurrences of each prediction
        predictions_counts = {
            "ExtraTrees": extra_trees_predictions.tolist().count(1),
            "RandomForest": random_forest_predictions.tolist().count(1),
            "DecisionTree": decision_tree_predictions.tolist().count(1),
        }

        # Get the model with the highest count
        max_model = max(predictions_counts, key=predictions_counts.get)

        # Get the final prediction based on the model with the highest count
        final_prediction = 1 if predictions_counts[max_model] > 0 else 0

        # Interpret predictions
        fraud_status = "Fraud" if final_prediction == 1 else "Not Fraud"

        fraud_status = input_data_subset['PotentialFraud'].values[0]
        print(fraud_status)

        return jsonify({
            'Fraud Status': fraud_status
        })

    except Exception as e:
        return jsonify({'error': str(e)})



if __name__ == '__main__':
    app.run(debug=True)