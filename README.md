cloudy-tweets
=============

Machine Learning solution for Kaggle.com's "<a herf="http://www.kaggle.com/c/crowdflower-weather-twitter">Partly Sunny with a Chance of Hashtags</a>"

<h3>Methodology</h3>

Feature space is comprised of word frequencies for m most common words (excluding stop words like: 'is','a','the',etc.). Use of Linear Ridge Regression for classification. Classifier predominantly prefered because of speed.

<h3>Resuts</h3>

Cross validation estimates for total RMSE across all 15 features is about 0.169 when n = 270 (aka feature space incoperates the 270 most common words). Current leader (2013-11-04) has attained around a 0.146 RMSE. Have only been able to bump m up to 270 before my computer maxed out its active memory. I need to upgrade.
