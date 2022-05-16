# Neural Network Charity Analysis

## Overview

The purpose of this analysis is to analyize a dataset from Alphaber Soup and create a binary classifier with machine learning and neural networks, to predict whether applicants will be successful if funded by Alphabet Soup.

The dataset comes from a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization.

Afterwards, we try to optimize the original model to increase its accuracy.

### Resources

- Data source: AlpahbetSoup's dataset from '/Resources/charity_data.csv'.

- Software use to perform the analysis: Jupyter Notebook v6.4.5 with Scikit-learn v0.24.2 and Tensorflow 2.0

## Results

The original dataset contains 34,219 records and 12 columns.

![Initial DataFrame](/Resources/datasetOriginal.png)

### Data Preprocessing

- The target variable is the **"IS_SUCCESSFUL"** column as this column shows if the funding was succesful (value of 1) or not (value of 0). Since this column is already binary, there is no need to apply any transformations.

```python:
y = application_df.IS_SUCCESSFUL.values
```

![Target Variable](/Resources/targetVariable.png)

- There are two columns: **'EIN'** and **'NAME'** that refer to the internal case number and the name of the Company that applied for funding, and should not be part of the analysis, thus removed from the data.

```python:
# Drop the non-beneficial ID columns, 'EIN' and 'NAME'.
application_df = application_df.drop(columns=['EIN', 'NAME'])
```

- The rest of the nine columns should be considered as posible targets for our model. But they need to be preprocessed.

    1. By running the unique method on each column we can see that there are some columns that contain too many values and, because they are categorical, they need to be binned:

    ![Features](/Resources/features.png)

    2. We performed a binning process on the **'APPLICATION_TYPE'** and **'CLASSIFICATION'**, resulting in fewer unique values:

    |APPLICATION_TYPE                                      |CLASSIFICATION                                         |
    |:-----------------------------------------------------|:-----------------------------------------------------:|
    |![Application type](/Resources/applicationBinning.png)|![Classification](/Resources/classificationBinning.png)|

    3. Next, we need to encode all our categorical to binary (numerical) values using the OneHotEncoder from Scikit Learn.

    ```python:
    application_cat = application_df.dtypes[application_df.dtypes == "object"].index.tolist()
    enc = OneHotEncoder(sparse=False)
    encode_df = pd.DataFrame(enc.fit_transform(application_df[application_cat]))
    encode_df.columns = enc.get_feature_names(application_cat)
    ```
    ![Encoded Dataframe](/Resources/datasetEncoded.png)



### Compiling. Training, and Evaluating the Model

The original model with the previous steps resulted in a accuracy of 0.5324, but when applied to the test variables, the accuracy dropped to 0.4625:

![Accuracy Original Model](/Resources/accuracyOriginal.png)


- For the optimized model, we tried the following:

    1. After analyzing the features, we decided to drop the **'CLASSIFICATION'** feature, because it had too many unique values that could be affecting the weights of our model.

    ```python:
    application_df = application_df.drop(columns=["CLASSIFICATION"])
    ```

    2. We also changed binning intervals for the **'APPLICATION_TYPE'** and of **'AFFILIATION'** to reduce the number of categories.

    |APPLICATION_TYPE                                 |AFFILIATION                                    |
    |:------------------------------------------------|:---------------------------------------------:|
    |![Application new](/Resources/applicBinning2.png)|![Classification](/Resources/affilBinning2.png)|    


    3. Increase the number of neurons.  For the first attempt, we also increased the number of neurons to 90 in the first hidden layer and to 45 in the second hidden layer, with the same activation functions.

    ![Attempt 1 Model](/Resources/nnSummary.png)


    4. Then we added a third layer with 15 nodes and the same activation funtions.

    ![Attempt 2 Model](/Resources/nn2Summary.png)


    5. Finally, we returned to the two hidden layer model, but changed the first layer's activation function to the **leaky-relu**, to see if allowing for negative values could improve the accuracy.

    ![Attempt 3 Model](/Resources/nn3Summary.png)


- We did increase our model performance, but couldn't reach the 75% treshold:

    **Optimized model Attempt 1** 
    ![Attempt 1 results](/Resources/nnResults.png)

    **Optimized model Attempt 2**
    ![Attempt 1 results](/Resources/nn2Results.png)

    **Optimized model Attempt 3***
    ![Attempt 1 results](/Resources/nn3Results.png)


## Summary

- As can be seen from the optimization attempts, we got better results than in the original. Increasing the number of neurons did improve the accuracy, and slightly the loss. Increasing the nuerons, decreased the loss and the improvement in accuracy was higher, but not that much. Finally, changing the activation functions yielded a slightly better accuracy but a very high loss, hence maybe overfitting the model. From these, we conclude that the better one was Attempt 2 with **more neurons**, **2 hidden layers** and **activation function 'relu'** in both layers.

    |Model    |Hidden Layers|Neurons      |Activation function       |Loss    |Accuracy|
    |:-------:|:-----------:|:-----------:|:------------------------:|:------:|:------:|
    |Original |Two          |80 and 40    |relu, relu, sigmoid       |0.8111  |0.4625  |
    |Attempt 1|Two          |90 and 45    |relu, relu, sigmoid       |0.8263  |0.6804  |
    |Attempt 2|Three        |90, 45 and 15|relu, relu, relu, sigmoid |0.6724  |0.5439  |
    |Attempt 1|Two          |90 and 45    |relu, relu, sigmoid       |2.6434  |0.6921  |

- As recommendation, since the data has many categorical data, it would be prudent to try a random forest model boosting or even a combination of logistic regression models.