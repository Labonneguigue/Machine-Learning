## Machine Learning Online Course from **Columbia University**

Here is a repo where you can find my work on several machine learning algorithms implementation in Python.

## Linear Regression (Project 1)

...

## Bayes Classifier

...

## K-Means & Gaussian Mixture Model

The idea here is that our input data points are in a X.csv file. We do not have missing values.

1.  K-Means algorithms

First, I generated randomly K centroids. I wasn't very happy with my results so I looked for a better initialization of the centroids and I've found the K-means++ algorithm.
I choose the first centroid randomly. Then, iteratively I calculate the distance D from every point to that centroid. I then choose the next centroid with a probability weighted by the D squared. This increases the probability to end up with centroids far from each other. The further each initial centroids are, the higher the probability to end up with true clusters in the end.

Expectation Step : I assign each data entry to the closest centroid. Closest meaning here the squared Euclidian distance.

Maximization Step : I update the mean of each centroids by computing the empirical average of each data points.

By successively executing the Expectation and Maximization steps of the EM algorithm, I calculate the K centroids and assign each data points to one of them.
The output is the centroids-[iteration].csv files.


![alt tag](https://github.com/Labonneguigue/Machine-Learning/tree/master/Images/kmeans.png)


2.  Maximum Likelihood EM for the Gaussian Mixture Model (GMM)

We treat the cluster assignments of each data point as the auxiliary data (missing data).
