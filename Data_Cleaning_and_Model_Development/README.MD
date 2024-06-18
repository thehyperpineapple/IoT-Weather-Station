# Data Processing and Model Development
This section will discuss the data processing and model development section of the application

## Dataset Used
A Kaggle dataset was used for this application due to the lack of time to create a specific dataset for the task at hand

Link to Dataset - https://www.kaggle.com/datasets/muthuj7/weather-dataset

## Data Description and Cleaning Process
Several steps taken to clean the data:

- Drop Unnecessary Columns:
    - Columns such as "Formatted Date", "Daily Summary", "Visibility (km)", "Loud Cover", "Wind Speed (km/h)", "Wind Bearing (degrees)", and "Precip Type" were removed from the dataframe using the drop method.


- Remove Specific Rows Based on Condition:
    - Rows where the "Summary" column had the value "Dangerously Windy and Partly Cloudy" were removed. This step was repeated for other specific values in the "Summary" column such as "Foggy", "Drizzle", "Breezy and Foggy", "Windy and Mostly Cloudy", "Dry", and "Humid and Partly Cloudy".

- Random Sampling and Removal:
    - A more targeted approach was used to filter entries by a specific label (e.g., "Summary") and value (e.g., "Partly Cloudy", "Mostly Cloudy", "Overcast"). For each category, a specified number of entries were randomly sampled and then removed from the dataset.

## Results

Different Machine Learning and Deep Learning models have been used on this dataset and have yielded the following results 

| Algorithm                                                     |Accuracy  |
|:-:                                                            |:-:|
|Logistic Regression                                            | 41.8% |
|Support Vector Classifier (One vs One)                         | 42.4% |
|K-Nearest Neighbours                                           | 39.2% |
|Gaussian Naive Bayes                                           | 39.3% |
|Decision Trees                                                 | 35.9% |
|Random Forest                                                  | 41.9% |
|Artifical Neural Network (ANN) [ 64-64-128-3 Architecture ]    | 46.1% |
|Artifical Neural Network (ANN) [ 32-256-3 Architecture ]       | 41.7% |
|Recurrent Neural Network (RNN) [ 32-64-128-3 Architecture ]    | 43.8% |

## Inference

- The data has been analyzed using a variety of machine learning and deep learning models.
- The models include Logistic Regression, Support Vector Classifier (One vs One), K-Nearest Neighbours, Gaussian Naive Bayes, Decision Trees, Random Forest, and different architectures of Artificial Neural Networks (ANNs) and a Recurrent Neural Network (RNN).
- Accuracy scores range from 35.9% to 46.1%, with Decision Trees performing the least accurately and an ANN with a 64-64-128-3 architecture achieving the highest accuracy.
The table indicates that deep learning models (ANN and RNN) have generally performed better than traditional machine learning models on this dataset, with the most complex ANN architecture yielding the best results.

## Conclusion

The dataset may have complex patterns that are better captured by deep learning models. However, all the accuracies seem to be relatively low, indicating that there may be room for further model optimization or that the dataset is particularly challenging. 

A plausible solution to this would be to create a Dataset more suited with the parameters the project is working with.
