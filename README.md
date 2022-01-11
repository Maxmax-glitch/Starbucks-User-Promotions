# Starbucks-User-Promotions Project  
Captstone Project based on Starbucks Transaction&amp;Promotion dataset focusing on user groups and promotional strategy 

This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks.
The task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.

## Motivation & Problem Statement
Customers will respond differently to different offer types. From the business perspective it is favorable to provide customers with the type of offers that increases their spending the most. While minimizing giving discounts to those that would have spent money anyway. To do so we need a better understanding of the customer base.
Goal: Perform Demographic & Value-based customer segmentation to tailor offers to customers more precisely.
Perform a data exploration to better understand the available data
Create groups with unsupervised clustering method based on demographic, account information
Include customer behavior during times of no promotion as their normal behavior (amount of visits & average amount spent per visit) in the clustering
Calculate impact of promotional offer types depending on each customer group

## Structure
- Problem Statement
- Metrics
- Data Exploration & Visualization
- Data Pre-Processing
- Modeling
- Evaluation and Validation
- Outlook & Reflection


## Libraries & Packages
Following libraries & packages were used in this project:
importpandas as pd
import numpy as np
import math
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import calinski_harabaz_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
import seaborn as sns

## Files in this repository
data folder - Raw data folder
  - portfolio.json - Containing offer ids and meta data about each offer
  - profile.json - Demographic data for each customer
  - transcript.zipx - Records for transactions, offers received, offers viewed, and offers completed (must be unpackaged before use)
pictures - Folder with graphical analysis from script used in blog post
LICENSE - standard MIT license
README.md - description of project & contents
Starbucks_Capstone_notebook-functionalized.ipynb - Final code, fully functionalized
Starbucks_Capstone_notebook-raw.ipynb - Working code, but not functionalized 


## How to use: 
There are two python files available:
  1. Starbucks_Capstone_notebook-functionalized contains all code and refactored with separate functions
  2. Starbucks_Capstone_notebook-raw contains the initial code before functionalizing. Only for information or if you want to walk through step by step

data 
- data available in JSON files, transcript data is zipped and must be unzipped before use
 
Blog post
- A discussion of the project and results is available on Medium: https://medium.com/@hartmann.max/customer-segmentation-offer-strategy-starbucks-data-set-c361b03c732


## Results
### Model Evaluation and Validation
To get a baseline model the standard k-means algorithm is used. Each categorical variable is one not encoded and all numerical variables are standardized.
The main hyperparameter of k-mean clustering is the number of clusters k used. To judge whether if the number of cluster is properly chosen we can utilize the Silhouette Score and try to maximize it.

Silhouette Score
If the ground truth labels are not known, evaluation must be performed using the model itself. The Silhouette Coefficient (sklearn.metrics.silhouette_score) is an example of such an evaluation, where a higher Silhouette Coefficient score relates to a model with better defined clusters.
Advantages
- The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering. Scores around zero indicate overlapping clusters.
- The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.
Drawbacks
- The Silhouette Coefficient is generally higher for convex clusters than other concepts of clusters, such as density based clusters like those obtained through DBSCAN.
Source: https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient

The largest score is achieved with four clusters, but mainly we can tell that a higher number of clusters (10 and above) achieves significantly lower scores.
To further analyze a good number of clusters for the k-means algorithm we can visualize the scores and clusters individually. The following plots show a low-dimensional representation (achieved through Principal component analysis (PCA) after clustering).
After review the number of clusters = 4 is chosen. We can see that there are some outliers that are being grouped into the main clusters that could benefit from a more refined clustering, but at the same time the additional clusters 5 & 6 are not defining these to us interesting points as individual clusters.

After predicting the clusters we can analyze how each group is behaving without any promotions and how they are responding to each type of offering (Buy-one-get-one BOGO, informational & discount):
The following groups and behavior were identified:
Group 0: Characterized through a above average income, female and high spending per visit and mostly long-term customers with an higher account age. This group spends the most without promotions per visit. This group is spending most per visit when no promotions are given. Though promotions and especially discounts increase transactions per week and might therefore still be viable after adjusting for costs.
Group 1: Really the only differentiating characteristic is the gender male. All other characteristics were distributed evenly. Group 1 spends less than Group 0, but has a similar amount of transactions per week. Group 1 is most positively responding to discounts while BOGO has almost no effect and could even be negative after subtracting costs. Informational promotion seems to only work for this Group as it decreases the spending across all groups, but increases the times visited, but only in Group 1 it has a net positive effect.
Group 2: This group has unknown gender, age and income. All accounts are older. The average visits are normal, but each with a very low average spending compared to other groups. Group 2 is spending almost no money per visit but has similar amount of transactions. Any promotional activity is not really adding value to the business. Giving no promotions might be the best cause of action.
Group 3: The last group has below average income, medium age and identifies predominantly as other gender. Their spending and visits without promotions are on average in between those of group 0 & 1. Finally Group 3 is best responding to BOGO offers and Discounts with BOGO offers slightly ahead in weekly total amount spend.

#### Hypothesis 1:
H0: There are no hidden groups of similar customers in the dataset.
H1: There are hidden groups/clusters of similar customers in the dataset.

H0 can be rejected. The k-means clustering algorithm measured through the silhouette score as well as visualization of clusters with PCA and original data show clear groups of customers. A clustering with k-means and number of clusters = 4 showed most promising results. The clusters are heavily influenced by the gender (female, male, other, unknown), so more demographic information and a finer clustering might lead to more specific sub-groups. Additionally, there were significant outliers in terms of total spending which were not yet captured in their own group, but are from a business perspective of course especially interesting. Either to convert existing customers into this group or to attract more customers which fit their criteria.


#### Hypothesis 2:
H0: The identified clusters are responding indifferently to the available promotional offer types.
H1: The identified clusters are responding differently to the available promotional offer types.

H0 can also be rejected. The spending habits of all customers during promotions were analyzed, compared to each other and within themselves to their normal spending habits. The key metrics are average amount spent per visit, average visits per week and total amount spent per week. The results were visualized.
Conclusion

### Reflection
This capstone project was done as part of the Udacity Data Science Nanodegree. The project was a valuable learning experience and I am looking forward to applying similar approaches in real life context. The data transformations were quite challenging, as I had only limited experience working with JSON as data sources and dictionaries as part of a data frame.

### Improvement
Additional clustering methods could be used although k-means is giving already satisfactory results as a baseline model. Further I would be interested to analyze the data more based on the business goals. Do we want to drive interactions, revenue or gross profit (revenue - promotional cost)?
The clusters are currently mostly defined by the genders while other demographic and spending habits seem to play a lower role in the definition of groups. For a first model based on simulated data this is okay. In reality a further analysis might be warranted to look deeper into customer segments. The ethics of doing customer segmentation mainly on gender are also debatable considering current discussions of ethics, bias and discrimination in machine learning.

### Acknowledgements
In this project I build directly on what I have learned in the Udacity Nanodegree Data Scientist which was a great preparation for this task. Additionally I acknowlodge the open source community and available packages I used during the project, without them it would not have been possible for me. 
