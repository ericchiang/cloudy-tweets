cloudy-tweets
=============

Machine Learning solution for Kaggle.com's "<a herf="http://www.kaggle.com/c/crowdflower-weather-twitter">Partly Sunny with a Chance of Hashtags</a>"

<h3>Methodology</h3>

Feature space is comprised of word frequencies for m most common words (excluding stop words like: 'is','a','the',etc.). Use of Linear Ridge Regression for classification. Classifier predominantly prefered because of speed.

<h3>Resuts</h3>

Cross validation estimates for total RMSE across all 15 features is about 0.169 when n = 270 (aka feature space incoperates the 270 most common words). Current leader (2013-11-04) has attained around a 0.146 RMSE. Have only been able to bump m up to 270 before my computer maxed out its active memory. I need to upgrade.

UPDATE (2013-11-05): scipy.sparse matrices implemented. Have had sucessful runs using 600 features, though generating test and training folds now takes signifigantly longer. Took an hour to split 600 features into 10 folds on my machine with no parallelization. Each fold is writen to file then retrieved individually for cv to maintain low active memory use; at this point this package definitely does not optimize for speed. Will try for 1000 features next.
MSE: 0.164 (m = 600; Ridge.alpha = 1e-7)
