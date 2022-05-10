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
y.value_counts(normalize=False)````

<img width="255" alt="Screen Shot 2022-05-10 at 7 09 02 PM" src="https://user-images.githubusercontent.com/99091066/167739158-29c5c246-2e51-47c1-81a1-446e6aefa314.png">

  > There are 75,036 low risk and 2,500 high risk individuals.



