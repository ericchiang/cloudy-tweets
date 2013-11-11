cloudy-tweets
=============

Machine Learning solution for Kaggle.com's 
"<a herf="http://www.kaggle.com/c/crowdflower-weather-twitter">
Partly Sunny with a Chance of Hashtags</a>"

Current place: 26th
Current score: 0.15970 (RMSE)

<h3>Overview</h3>

Sentiment classification of tweets! Each tweet is linked to a confidence
interval of 15 unique sentiments.

Sentiments include (see data/variableNames.txt for details):
<ul>
  <li>current weather</li>
  <li>future forecast</li>
  <li>hot</li>
  <li>rain</li>
  <li>snow</li>
</ul>

<h3>Methodology</h3>

Simple pipeline of feature generation to classification.

Feature space is comprised of word frequencies for m most common words and only
considers unigram information. Use of linear ridge regression for 
classification. Classifier predominantly prefered because of speed.

<h3>Resuts</h3>

Cross validation estimates for total RMSE across all 15 features is about 0.169
when n = 270 (aka feature space incoperates the 270 most common words).
Current leader (2013-11-04) has attained around a 0.146 RMSE. Have only been
able to bump m up to 270 before my computer maxed out its active memory. I need
to upgrade.
  Cross validaiton RMSE: 0.169 (m = 270; alpha = 1e-7; folds = 10)

UPDATE (2013-11-05):<br>
scipy.sparse matrices implemented. Have had successful runs using 600 features,
though generating test and training folds now takes signifigantly longer. Took
an hour to split 600 features into 10 folds on my machine with no
parallelization. Each fold is writen to file then retrieved individually for cv
to maintain low active memory use; at this point this package definitely does
not optimize for speed. Will try for 1000 features next.<br>
  Cross validation RMSE: 0.164 (m = 600; ridge regression; alpha = 1e-7)

UPDATE (2013-11-06):<br>
Successful run with 1000 features. Memory failure when attempting 2000.
Considering work arounds.<br>
  Cross validation RMSE: 0.160 (m = 1000; alpha = 1e-7; folds = 10)

UPDATE (2013-11-07):<br>
Observations now split into train/test folds before being mapped to feature
space. This means fold splitting is preformed on a numpy array rather than a 
large sparce matrix  which creates a faster and lower memory use pipeline. 
Successful runs with 2000 and 3000 features. Very clearly hitting diminishing
returns when considering expansions of the feature space.<br>
  Cross validation RMSE: 0.1578 (m = 3000; alpha = 1e-7; folds = 10)

UPDATE (2013-11-08):<br>
First two submission to the Kaggle.com leaderboards. First sumbission used a
feature space of the m most frequent words in the training data, the second
considered the m most frequent words in the combination of the training and
test. The first strategy proved marginally better though both resulted in a
surprisingly high error score by Kaggle as compared to cross validation. Will
discuss those discrepancies in detail under the 'Details/Musing' header since
the differences are far more signifigant than what might be allowable.<br>
  Kaggle RMSE: 0.16350 (m = 3000; alpha = 1e-7) 

UPDATE (2013-11-09):<br>
Signifigant expansion of rare word mapping. Numeric and alphanumeric data
mapped to a larger set of possible values. Emoticon information now preserved
as inspired by Go, Bhayani, and Huang demonstration of its usefulness in 
Twitter sentiment classification [1]. Exclamation and question marks presence
also collected which were both noted by Pang, Lee and Vaithyanathan to be an
unobvious source of information to humans (grad students) picking indicator
words [2]. Details under the 'Details/Musing' header.<br>
  Cross validation RMSE: 0.1578 (m = 3000; alpha = 1e-7; folds = 3)<br>
  Cross validation RMSE: 0.1576 (m = 4000; alpha = 1e-7; folds = 3)<br>
  Cross validation RMSE: 0.1586 (m = 5000; alpha = 1e-7; folds = 3)<br>
  Kaggle RMSE: 0.16298 (m = 3000; alpha = 1e-7)<br>
  Kaggle RMSE: 0.16314 (m = 4000; alpha = 1e-7)<br>
  Kaggle RMSE: 0.16370 (m = 5000; alpha = 1e-7)<br>

UPDATE (2013-11-10):<br>
Considering four new modificaitons after yesterday's results:<br>
Increasing the alpha value to favor variance over bias. Considering word 
presense rather than word count for feature generation [1].Reworking
altercation of scores predicited out of range (above 1.0 or bellow 0.0). Trying
different classifiers.

UPDATE (2013-11-11):<br>
Elastic net tested and imediately showed improvements to ridge. Takes
significantly longer to run than ridge but is still completes in a reasonable
amount of time. Attempted to test support vector machines regression since it
has shown to be successful as a sentiment classifier. Unfortunately, time
complexity is an issue since it took roughly twice the time of elastic net in
inital testing. Am now using a free EC2 micro instance on AWS to run 
regressions since I need to use my laptop for other work. If anyone wants to
donate cycles message me! Currently placed 26th with elastic net results.<br>
  Kaggle RMSE: 0.16014 (m = 3000; elastic net; alpha = 1e-5)<br>
  Kaggle RMSE: 0.15970 (m = 4000; elastic net; alpha = 1e-5)


<h3>Details/Musings</h3>

2013-11-07:<br>
To be honest I've been surprised by how successful my approach has been so far.
There has been very little thought put into parsing tweets or improving
information retrieval pre-regression. The only efforts so far have been mapping
rare words; grouping similar low frequency words which would intuitively hold
similar meanings. The binning effors which have been implemented so far pertain
to numeric and alphanumeric words.
<br>Currently bins:
<ul>
  <li>Alphanumeric</li>
  <li>Numeric
    <ul>
      <li>Over 90</li>
      <li>Between 90 and 50</li>
      <li>Between 50 and 10</li>
      <li>Less than 10</li>
    </ul>
  </li>
</ul><br>
Inital efforts to further partition bins failed to improve the accuracy of the
hypothesis. Though admittingly, those efforts took place  when the feature 
space was an order of magnitude smaller.

2013-11-08:<br>
My initial hypothesis to explain why the cross validation errors were so
different from the Kaggle scores is that the high number of folds lead to a
high trian/test ratio, allowing the training data to better encompass the test.
Will run 3 fold CV (rather than 10 fold) against future submission to test this
theory. Very adamant in bringing CV in line with Kaggle since CV results will
be critical in detemining if different variables require different classifiers.

Also, for documentation, all reported scores will now specify if they were
produced by cross validation or by blind test (what Kaggle does). Number of
folds will also be noted. This should have been the policy from the beginning,
but it's hard to voluntarily admit that your model might not be as good as it
appears. Luckily, Kaggle provides the benchmark and self reporting of scores
remains a rare phenomenon in other fields.

2013-11-09<br>
Current rare word mapping strategies.

Before stripping tweets of punctuation and splitting into words, the string is
check for exclamation marks, question marks and emoticons. Positive emoticons
are mapped to ':)', negative emoticons are mapped to ':(', neutral emoticons 
are mapped to ':/', and winky faces ';)' are mainted. Links, mentions, and RT's
are now also preserved (they were previously not). Exclamation and question
mark methodology taken from [1]. Emoticon methodology taken from [2].

After tweets are cleared of punctuation and split each individual word is
determined to either be numeric, alphanumeric, or alphabetical. If the word is
numeric, it is rounded down to the nearest ten, grouping numbers of similar
size together. If the word is alphanumierc, the numbers are removed and a
special '#num' tag is appended to the word. This has had the very pleasing
result of producing common strings such as 'mph #num', 'pm #num', and 'f #num'
which arise organically from the program rather than having to be selected for.


<h3>References</h3>
[1] B. Pang, L. Lee, and S. Vaithyanathan. Thumbs up? Sentiment classification
using machine learning techniques. 2002
[2] A. Go, R. Bhayani, and L. Huang. Twitter Sentiment Classification using
Distant Supervision. 2009
