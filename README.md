## Machine Learning Online Course from **Columbia University**

Here is a repo where you can find my work on several machine learning algorithms implementation in Python.

## Ridge Regression and Active Learning (Project 1)

1.  Ridge Regression

**Problem:** In this part I implement a ridge regression algorithm (ℓ2-regularized least squares linear regression). By maximizing the objective function we find the unknown parameters wRR.

![ScreenShot](Images/rr.tiff)

**Solution:** The parameters wRR are calculated that way:

![ScreenShot](Images/rr-solution.tiff)

Once wRR is learned from the training data (X, Y), I predict the output Y from a test sample using :

![ScreenShot](Images/rr-prediction.tiff)

2.  Active Learning

**Problem:**

**Solution:**


## Bayes Classifier (Project 2)

...

## K-Means & Gaussian Mixture Models (Project 3)

The idea here is that our input data points are in a X.csv file. We do not have missing values.

1.  K-Means algorithms

**Problem:** We try to find K centroids  {μ1,…,μK}  and the corresponding assignments of each data point  {c1,…,cn}   where each  ci∈{1,…,K}   and c_i indicates which of the K clusters the observation x_i belongs to. The objective function that we seek to minimize can be written as

![ScreenShot](Images/minimize_kmeans.tiff)

**Solution:** First, I generated randomly K centroids. I wasn't very happy with my results so I looked for a better initialization of the centroids and I've found the K-means++ algorithm. The implementation is the following :

I choose the first centroid randomly. Then, iteratively I calculate the distance D from every point to that centroid. I then choose the next centroid with a probability weighted by the distance D squared. This effectively increases the probability to end up with centroids far from each other. The further each initial centroids are, the higher the probability to end up with true clusters in the end.

I then successively iterate through the following steps:

Expectation Step : I assign each data entry to the closest centroid. Closest meaning here the squared Euclidian distance.

Maximization Step : I update the mean of each centroids by computing the empirical average of each data points.

By successively executing the Expectation and Maximization steps of the EM algorithm, I calculate the K centroids and assign each data points to one of them.
The output is the centroids-[iteration].csv files.


![ScreenShot](Images/kmeans.png)

We can see on the picture that each color represents one cluster.


2.  Maximum Likelihood EM for the Gaussian Mixture Model (GMM)

**Problem:** Now the data is generated as follows :

![ScreenShot](Images/gmm.tiff)

In other words, the ith observation is first assigned to one of K clusters according to the probabilities in vector π, and the value of observation xi is then generated from one of K multivariate Gaussian distributions, using the mean and covariance indexed by ci.
We treat the cluster assignments of each data point as the auxiliary data (missing data of the EM algorithm).

**Solution:** Here is the solution I implemented.

![ScreenShot](Images/gmm.png)

Phi for a certain data point would be the probabilities to belong to each cluster. Its sum is 1. It really shows here the soft clustering dimension of the GMM algorithm. I use the plug-in classifier to evaluate this function further detailed here.

![ScreenShot](Images/phi.png)

The initialization of π, mu and sigma are really impacting the efficiency of the solution. The way I did is using the results from the previous K-means++ algorithm.
