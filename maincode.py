import pandas as pd  
data = pd.read_csv("archive\loan_data_1.csv")
print(data)

#data analysis

print(data.head())
print(data.tail())

print(data.info())
print(data.isnull().sum())

print(data.duplicated().sum())

# data preprocessing

##remove irrelevent data

data=data.drop(["Unnamed: 0","Loan_ID"], axis=1)
print(data.head())

##missing values handling 
print(data.isnull().sum())
data["Gender"]=data["Gender"].fillna(data["Gender"].mode()[0])
data["Dependents"]=data["Dependents"].fillna(data["Dependents"].mode()[0])
data["Education"]=data["Education"].fillna(data["Education"].mode()[0])
data["Self_Employed"]=data["Self_Employed"].fillna(data["Self_Employed"].mode()[0])
data["ApplicantIncome"]=data["ApplicantIncome"].fillna(data["ApplicantIncome"].mean())
data["CoapplicantIncome"]=data["CoapplicantIncome"].fillna(data["CoapplicantIncome"].median())
data["LoanAmount"]=data["LoanAmount"].fillna(data["LoanAmount"].mean())
data["Loan_Amount_Term"]=data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].mode()[0])
data["Credit_History"]=data["Credit_History"].fillna(data["Credit_History"].mode()[0])


print(data.isnull().sum())

##label encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

columns=["Gender","Married","Education","Self_Employed","Property_Area","Dependents","Loan_Status"]
for i in columns:
    data[i]=le.fit_transform(data[i])

print(data)


#split data into features and target
x=data.drop("Loan_Status",axis=1)
y=data["Loan_Status"]

#split data into training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

#logistic regression model
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)

#predict on test data
y_predict=model.predict(x_test)
print(y_predict)

#model evaluation
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_predict)
print("Accuracy:", accuracy)

