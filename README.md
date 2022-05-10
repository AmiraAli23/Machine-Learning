# Machine-Learning
Unit 11 - Risky Business

![image](https://user-images.githubusercontent.com/99091066/167738122-a4b8a91f-2baf-481b-9d3a-396b4e04ca07.jpeg)

This assignment evaluates several machine learning models in order to predict credit risk. 

## Resampling

Using the lending data [file](https://github.com/AmiraAli23/Machine-Learning/blob/0c634c9c7b35c9e3c2690e3f67006cd333b118a2/lending_data.csv) , we analyzed the value types for each column and converted the columns with non-numerical values to dummy variables using the `pd.get_dummies` function.

We set X as all of the columns besides loan status, and Y as loan status, since we are trying to determine whether an investor is low or high risk.

We check how many low and high risk individuals are in the data set. 

```python 
# Check the balance of our target values
y.value_counts(normalize=False)

```

<img width="255" alt="Screen Shot 2022-05-10 at 7 09 02 PM" src="https://user-images.githubusercontent.com/99091066/167739158-29c5c246-2e51-47c1-81a1-446e6aefa314.png">

  > There are 75,036 low risk and 2,500 high risk individuals.


### Data Pre-Processing 

We first import `StandardScaler` from `sklearn.preprocessing` and fit the training data. 

````python

scalex = data_scaler.fit(X_train)

````

### Simple Logistic Regression

After importing `LogisticRegression` from `sklearn.linear_model` , we fit the data and calculate the balanced accuracy based on the `y_test` and `y_pred` variables. We also display the confusion matrix and print the imbalanced classification report. 

```python 
balanced_accuracy_score(y_test, y_pred)
```

<img width="314" alt="Screen Shot 2022-05-10 at 7 16 38 PM" src="https://user-images.githubusercontent.com/99091066/167739896-85e0b6b4-2d65-48c3-956a-a4f95b18d55b.png">


  > Using this model, we calculate a balanced accuracy score of  ~ 0.80415


```python
confusion_matrix(y_test, y_pred)
````

<img width="242" alt="Screen Shot 2022-05-10 at 7 17 52 PM" src="https://user-images.githubusercontent.com/99091066/167739982-bb468ef7-95f6-40e4-8dd9-bb0b6e364787.png">

```python
print(classification_report_imbalanced(y_test, y_pred))
````


<img width="637" alt="Screen Shot 2022-05-10 at 7 18 52 PM" src="https://user-images.githubusercontent.com/99091066/167740080-34aa14a6-c802-4f4c-bf1b-ee9f31defe68.png">


  > F1 scores are the means between precision and recall, and determine the model's accuracy. The average f1 score for this model is 0.74. This is a decent score, however it could be higher. 
















