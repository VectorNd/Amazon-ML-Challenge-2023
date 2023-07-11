# Amazon ML Challenge 2023

## Objective 
The goal is to develop a machine learning model that can predict the length dimension of a product. Product length is crucial for packaging and storing products efficiently in the warehouse. Moreover, in many cases, it is an important attribute that customers use to assess the product size before purchasing. However, measuring the length of a product manually can be time-consuming and error-prone, especially for large catalogs with millions of products.

## Dataset 
- The dataset folder contains the following files:
  1.  **Train.csv: 2249698 x 6**
  2.   **Test.csv : 734736 x 5** 
- Access to the product title, description, bullet points, product type ID, and product length for 2.2 million products to train and test submissions . Note that there is some noise in the data.

## Evaluation Metric 
- **Score** = max( 0 , 100*(1-metrics.mean_absolute_percentage_error(actual,predicted)))

## Approach
-  We use supervised approach (classification or regression) based on Deep Learning in which product vector representation is being learned during loss minimization.
- The advantages of this approach are numerous (1) it doesn’t require special treatment and complex preprocessing of product attributes including product titles and other textual information (2) it solves the main supervised task and generates product vector representations in parallel (3) the main supervised task can be then improved by updating the knowledge about similar products.
- The data contains all types of features: numeric, categorical, and textual, which usually appear in e-commerce data. We have also several challenges here: since the seller is free to provide any information about the product, the product description can be completely empty or misspelled. Product names can also be written with typos.
- The Deep Learning solution has following architecture :



 
    ![modelong](https://github.com/VectorNd/Amazon-ML-Challenge-2023/assets/111004091/1e3e029e-a1e3-4e3a-9d1e-773c3fba791e)







  - The input consists of all sorts of features: numeric, categorical, and text. Categorical input is translated into lower-dimensional space through Embedding Layer. Text input (product title, product description) is first tokenized into words and characters and transformed into lower-dimensional space through Embedding Layer. Learning product titles and descriptions on a character level may increase matches for misspelled products or products having slight differences in text input. GRU/LSTM Layer returns a hidden state of the last word or character in the sequence. Finally, all layer outputs are concatenated into a dense layer and additional dense or skip-connection layers can be defined on top. Two layers highlighted in green play an important role in our downstream task.
  - The target variable price was first log-transformed and then transformed with scikit-learn PowerTransformer
  - Model Parameters are summarised below :








    - Finally Training is done to predict the Product Length. 
