# Machine-Learning
Unit 11 - Risky Business

![image](https://user-images.githubusercontent.com/99091066/167738122-a4b8a91f-2baf-481b-9d3a-396b4e04ca07.jpeg)

This assignment looks into evaluating several machine learning models in order to predict credit risk. 

## Resampling

Using the lending data [file](https://github.com/AmiraAli23/Machine-Learning/blob/0c634c9c7b35c9e3c2690e3f67006cd333b118a2/lending_data.csv), we analyzed the value types for each column and converted the columns with non-numerical values to dummy variables using the `pd.get_dummies` function.

We set X as all of the columns save for `loan_status`, which is the target variable, and was set as the Y. This model was used to assess the risk of individual investors in the dataset.


```python 
# Check the balance of our target values
y.value_counts(normalize=False)

```

<img width="255" alt="Screen Shot 2022-05-10 at 7 09 02 PM" src="https://user-images.githubusercontent.com/99091066/167739158-29c5c246-2e51-47c1-81a1-446e6aefa314.png">

  > There are 75,036 low risk and 2,500 high risk individuals.


### Data Pre-Processing 

We first imported `StandardScaler` from `sklearn.preprocessing` and fitted the training data. 

````python

scalex = data_scaler.fit(X_train)

````

### Simple Logistic Regression

After importing `LogisticRegression` from `sklearn.linear_model`, we fitted the data and calculated the balanced accuracy based on the `y_test` and `y_pred` variables. We also displayed the confusion matrix and printed the imbalanced classification report. 

```python 
balanced_accuracy_score(y_test, y_pred)
```

<img width="314" alt="Screen Shot 2022-05-10 at 7 16 38 PM" src="https://user-images.githubusercontent.com/99091066/167739896-85e0b6b4-2d65-48c3-956a-a4f95b18d55b.png">


  > Using this model, we calculated a balanced accuracy score of ~ 0.80415


```python
confusion_matrix(y_test, y_pred)
````

<img width="242" alt="Screen Shot 2022-05-10 at 7 17 52 PM" src="https://user-images.githubusercontent.com/99091066/167739982-bb468ef7-95f6-40e4-8dd9-bb0b6e364787.png">

```python
print(classification_report_imbalanced(y_test, y_pred))
````


<img width="637" alt="Screen Shot 2022-05-10 at 7 18 52 PM" src="https://user-images.githubusercontent.com/99091066/167740080-34aa14a6-c802-4f4c-bf1b-ee9f31defe68.png">


  > F1 scores are the means between precision and recall, and determine the model's accuracy. The average F1 score for this model is 0.74. This is a decent score, however it could be higher. 



### Oversampling

We resampled the data with `RandomOversampler` from `imblearn.over_sampling` and performed similar steps as above. 

```python
y_pred1 = logreg.predict(x_scaledtest) 
balanced_accuracy_score(y_test, y_pred1)
````

<img width="300" alt="Screen Shot 2022-05-10 at 7 29 55 PM" src="https://user-images.githubusercontent.com/99091066/167740976-471148f5-b545-4915-a5db-656758b7a9b0.png">

  > The balanced accuracy score for this test was significantly higher than the previous test, at ~ 0.99464. 


<img width="632" alt="Screen Shot 2022-05-10 at 7 30 58 PM" src="https://user-images.githubusercontent.com/99091066/167741057-a320fb3d-e8a8-4313-baec-fb71f1379f8f.png">

  > The F1 scores for this test are much higher at an average of 0.99. 


### SMOTE Oversampling

We import `SMOTE` from `imblearn.over_sampling`

```python
y_predsmote = modelsmote.predict(x_scaledtest)
balanced_accuracy_score(y_test, y_predsmote)
````

<img width="348" alt="Screen Shot 2022-05-10 at 7 34 40 PM" src="https://user-images.githubusercontent.com/99091066/167741333-a9ea767c-9945-4d47-b1f5-c1e49230a50a.png">

  > The balanced accuracy using SMOTE is similar to RandomOversampler, at ~0.994668.


<img width="636" alt="Screen Shot 2022-05-10 at 7 35 51 PM" src="https://user-images.githubusercontent.com/99091066/167741437-bc711db0-dc2a-4bea-97df-8de03e30a7af.png">

  > This model also generates a high F1 score at 0.99.


### Undersampling

We imported `ClusterCentroids` from `imblearn.under_sampling`

 ```python 
y_predcluster=modelcc.predict(x_scaledtest)
balanced_accuracy_score(y_test, y_predcluster)
````

<img width="341" alt="Screen Shot 2022-05-10 at 7 39 46 PM" src="https://user-images.githubusercontent.com/99091066/167741735-8c0a39c2-c518-4c37-99ab-ea57b7ffbaaa.png">

  > The model generates a good balanced accuracy score of ~0.99328, slightly lower compared to the other test.


<img width="622" alt="Screen Shot 2022-05-10 at 7 41 57 PM" src="https://user-images.githubusercontent.com/99091066/167741904-fea4fd7a-45fe-47fb-be6c-2c854de24b4a.png">

  > The F1 score on this test is high at 0.99.

### Combination (Over and Under) Sampling

We imported `SMOTEEN` from `imblearn.combine`

```python 
y_predsm = modelsm.predict(x_scaledtest)
balanced_accuracy_score(y_test, y_predsm)
````

<img width="305" alt="Screen Shot 2022-05-10 at 7 45 33 PM" src="https://user-images.githubusercontent.com/99091066/167742180-a6012663-038e-4343-9f49-8752e6c36365.png">

  > The balanced accuracy score is high, and seems to be identical to the SMOTE Oversampling model.


<img width="624" alt="Screen Shot 2022-05-10 at 7 46 21 PM" src="https://user-images.githubusercontent.com/99091066/167742243-f7b982ab-5856-4afc-a396-6cddc149df36.png">

> F1 score is also high.

## Resampling Conclusions

1. The models with the highest balanced accuracy scores are the SMOTE and SMOTEEN models.

2. Each model had a recall score of 0.99, however only the RandomOversampler, SMOTE, and SMOTEEN models had a perfect score for `high_risk` recall at 1. 

3. All of the models had the same geometric score of 0.99

## Ensemble Learning

Using the LoanStats data [file](https://github.com/AmiraAli23/Machine-Learning/blob/ac947c33ae67956b80a752d22a7a27cda6df81a9/LoanStats_2019Q1.csv), we  cleaned the data by dropping all null values. We then converted all non-numerical values as dummy variables in order to properly run the tests.


We then assigned X as all columns except `loan_status`, which was the Y.

In this dataset, `68470` individuals are considered high risk while `347` are considered low risk.

### Data Pre-Processing

Using `StandardScaler` from `sklearn.preprocessing` , we fitted the training data. 

````python 
scalex=data_scaler.fit(X_train)
x_scaled = scalex.transform(X_train)
x_scaledtest=scalex.transform(X_test)

````

## Ensemble Learners

### Balanced Random Forest Classifier

We imported `BalancedRandomForestClassifier` from `imblearn.ensemble` 

````python
 
brf = BalancedRandomForestClassifier(n_estimators=100, random_state=1)
brf.fit(x_scaled, y_train)
y_pred = brf.predict(x_scaledtest) 

````

<img width="305" alt="Screen Shot 2022-05-11 at 7 45 59 PM" src="https://user-images.githubusercontent.com/99091066/167965101-5ebc31b0-2d88-4d86-a955-88adc8f30817.png">

  >  The balanced accuracy score for this test is ~ 0.7887. 

<img width="614" alt="Screen Shot 2022-05-11 at 7 47 47 PM" src="https://user-images.githubusercontent.com/99091066/167965286-17458ee7-e417-45ce-bdba-12185e810fd3.png">

  > The F1 score for this model is high at 0.94.


### Easy Ensemble Classifier

We imported `EasyEnsembleClassifier` from `imblearn.ensemble`

````python

eec = EasyEnsembleClassifier(n_estimators=100, random_state=1)
eec.fit(x_scaled, y_train)
y_pred1 = eec.predict(x_scaledtest) 

````


<img width="294" alt="Screen Shot 2022-05-11 at 7 52 40 PM" src="https://user-images.githubusercontent.com/99091066/167965742-33e60c04-12a6-4eb3-bf68-b02330ea8125.png">

  > The balanced accuracy score is significantly higher than the previous model at ~0.9253.

<img width="609" alt="Screen Shot 2022-05-11 at 7 53 32 PM" src="https://user-images.githubusercontent.com/99091066/167965824-2c3a3a0c-aafd-4c4b-afb2-c31cb3d8e72a.png">

  > The f1 score is higher than the previous model at 0.97.


## Ensemble Conclusions

1. The EasyEnsemble model had the higher balanced accuracy score ( 0.788 vs 0.9253 )

2. The EasyEnsemble model had a higher recall score (0.9 vs 0.94)

3. The EasyEnsemble model has a higher geomteric score (0.78 vs 0.93)

4. The top three features in order of importance are `total_rec_prncp` , `last_pymnt_amnt`, and `total_pymnt_inv`. 
































