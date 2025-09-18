from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shap

main = tkinter.Tk()
main.title("Future of Loan Approvals with Explainable AI")
main.geometry("1000x650")

# Global Variables
dataset = None
scaler = None
label_encoder = []
cols = []
rf = None
X_train, X_test, y_train, y_test = None, None, None, None


def loadDataset():
    global dataset
    filename = filedialog.askopenfilename(title="Select Dataset", filetypes=(('CSV Files', '*.csv'),))
    if filename:
        dataset = pd.read_csv(filename)
        text.delete('1.0', END)
        text.insert(END, filename + " loaded successfully.\n\n")
        text.insert(END, str(dataset.head()) + "\n\n")
        dataset.hist(figsize=(12, 8))
        plt.show()


def processDataset():
    global dataset, scaler, label_encoder, cols
    if dataset is None:
        text.delete('1.0', END)
        text.insert(END, "Please upload the dataset first.\n")
        return
    dataset.fillna(0, inplace=True)
    label_encoder = []
    cols = []
    for col in dataset.columns:
        if dataset[col].dtype == 'object':
            le = LabelEncoder()
            dataset[col] = le.fit_transform(dataset[col].astype(str))
            label_encoder.append(le)
            cols.append(col)
    scaler = StandardScaler()
    features = dataset.drop(['Loan_Status'], axis=1).values
    scaled_features = scaler.fit_transform(features)
    dataset['Loan_Status'] = dataset['Loan_Status']
    text.delete('1.0', END)
    text.insert(END, "Dataset processed and normalized successfully.\n\n")


def splitDataset():
    global dataset, X_train, X_test, y_train, y_test
    if dataset is None:
        text.delete('1.0', END)
        text.insert(END, "Please upload and preprocess the dataset first.\n")
        return
    X = dataset.drop(['Loan_Status'], axis=1).values
    y = dataset['Loan_Status'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    text.delete('1.0', END)
    text.insert(END, f"Training samples: {len(X_train)}\n")
    text.insert(END, f"Testing samples: {len(X_test)}\n\n")


def trainModel():
    global rf, X_train, y_train, y_test, X_test
    if X_train is None or y_train is None:
        text.delete('1.0', END)
        text.insert(END, "Please split the dataset first.\n")
        return
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions) * 100
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
    plt.title("Random Forest Loan Status Confusion Matrix")
    plt.show()
    text.delete('1.0', END)
    text.insert(END, f"Model trained successfully.\nAccuracy: {accuracy:.2f}%\n\n")


def explainAI():
    global rf, X_train
    if rf is None or X_train is None:
        text.delete('1.0', END)
        text.insert(END, "Please train the model first.\n")
        return
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, show=True)


def predictLoanStatus():
    global rf, scaler, label_encoder, cols
    if rf is None or scaler is None:
        text.delete('1.0', END)
        text.insert(END, "Please train the model first.\n")
        return
        

    filename = filedialog.askopenfilename(title="Select Test Dataset", filetypes=(('CSV Files', '*.csv'),))
    if filename:
        test_data = pd.read_csv(filename)
        test_data.fillna(0, inplace=True)

        # Apply label encoding on categorical columns
        for i, col in enumerate(cols):
            if col in test_data.columns:
                test_data[col] = label_encoder[i].transform(test_data[col].astype(str))

        # Scale features
        features = scaler.transform(test_data.values)
        predictions = rf.predict(features)

        # SHAP explanations
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(features)  # list of arrays for each class

        text.delete('1.0', END)
        for i, (row, pred) in enumerate(zip(test_data.values, predictions)):
            shap_dict = dict(zip(test_data.columns, shap_values[1][i]))  # class 1 = Approved
            sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)

            # Get top 2 contributing features
            reasons = []
            for feat, impact in sorted_features[:2]:
                direction = "increased" if impact > 0 else "decreased"
                reasons.append(f"{feat} {direction} the approval chance")

            decision = "Approved" if pred == 1 else "Rejected"
            reason_text = "; ".join(reasons)
            

            text.insert(END, f"Test Data {i + 1}: Prediction: {decision}\nReasons: {reason_text}\n\n")



# UI Setup
title = Label(main, text='Future of Loan Approvals with Explainable AI', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1', font=('times', 16, 'bold'), height=3, width=120)
title.pack()

buttons = [
    ("Upload Loan Application Dataset", loadDataset, 10, 100),
    ("Preprocess Dataset", processDataset, 10, 160),
    ("Split Dataset Train & Test", splitDataset, 10, 220),
    ("Train AI on Loan Approval", trainModel, 330, 100),
    ("Explainable AI", explainAI, 330, 160),
    ("Predict Loan Status using Test Data", predictLoanStatus, 330, 220)
]

for text_btn, command, x, y in buttons:
    btn = Button(main, text=text_btn, command=command, width=30, height=2)
    btn.place(x=x, y=y)
    btn.config(font=('times', 13, 'bold'))

text = Text(main, height=22, width=140)
text.place(x=10, y=280)
text.config(font=('times', 12, 'bold'))

main.config(bg='light coral')
main.mainloop()

