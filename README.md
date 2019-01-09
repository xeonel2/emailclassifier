# Email Classifier

**-** _Venkatakrishnan Parthasarathy_

## Problem Statement: Train a ML model to be able to classify emails into spam and non-

spam/ham with a given data set to train and validate the model.

## Approach: To have a stable and good model, a series of procedure are followed. This is the

following approach taken in this assignment:

```
(1) Organize the data set and make it ready for preparation
(2) Go through all the data and clean them
(3) Shuffle the data and split into training and test set
(4) On the training set, extract the features that might contribute to the classification
(5) Get insights on how the data set is based on the features extracted
(6) Do a second round of cleaning based on the data extracted to make the model more
unbiased
(7) Extract the features again for training
(8) Train models based on the features and do cross validation
(9) Choose the model with the best accuracy and save it for testing
(10) Test the model for it’s accuracy and show metrics based on the results
```
## Design of the python script:

- The script should be able to execute the requested step only so it’s easier to debug and
    fast


- Command line arguments for the step are taken to accomplish this
- The output of one step is stored in a file using pickle so it can just be read by the other
    step by calling the program with the respective arguments
- Storing the objects of intermediate steps in a file allows us to skip doing previous steps
    again which effectively saves a considerable amount of time

## Data preparation and cleaning:

- All the Ham and spam mails are put into two directories ("dataset/allspam/*.txt")
    and ("dataset/allham/*.txt")
- A function “Initialize” is defined to do the preliminary cleaning steps
- Traverse through the files and make a list of all the words in the email body
- When looking at the content of the mails, even though there is no meaningful message,
    some mails just have the word “Subject:” in the starting
- The first element in all the lists is the word “Subject:”. So removing it from all the
    messages when traversing


- Dispose of emails that become empty after removing the word “subject”.
- Stop words are removed from the messages to make the feature list smaller and cleaner


- The list of the mails are put in a pandas data-frame with columns ‘content’ and ‘spam’
    which are email body and Boolean type spam/ham respectively

```
The data frame is shuffled for even distribution of spam and ham across the data frame
and then split between test and training data (70%:30%)
```
- The resulting data-sets are written to a file for the next step
- Now the training data-set can be used for exploratory analysis and test data-set should not
    be touched

## Data extraction for exploration:

- Load the training data-set from file and separate ham and spam again.
- The email content is now filled with list of words. Converting it to space separated
    sentences for use with CountVectorizer()


- CountVectorizer is used to get the bag of words so we can explore the word frequencies
    in different types of emails
- Dump the word frequencies in a file for visualization in next step.
- Get the length of emails and dump is to a file so further analysis can be done on that as
    well


## Data exploration for insights:

- Load the dumped pandas series of word frequencies in spam and ham
- By looking at the most frequent words in the ham and spam, we can notice quite a few
    words are present in both the ham and spam. The following image shows the common
    words highlighted in yellow among the top 30 most frequent words.
- These common words would definitely not help improving the accuracy of the
    classification model. So we add these to the list of stop words and perform cleaning and
    extraction of data again to get more insights regarding the data.


- Even though some words like 1, 100, 1000 are present in both the types, we are not going
    to be removing them. There is a possibility that smaller numbers are usually used in
    transnational or ham messages and numbers like 1000000000000 could possible be used
    more frequently in spam mails.
- We will also not be removing words like “WWW” because spam email might contain
    URLs more frequently than ham ones.
- Plotting the word frequencies after second round of cleaning, using pygal plotting library,
    - If we
       look at
       the bar
       plots,
       we can
       notice
       that the
       word
       “enron”
       which
       is the
       name of
       the


```
company is having a very huge footprint on the ham plot. If we want to train a generic
email classifier, we would need to exclude company specific words like this.
```
- However, we are assuming that this classification model is particularly for that company
    and hence for better accuracy of the model, keeping it.
- The maximum number of times a frequent word occurs in a spam email is usually (
    - 5000) way less that ham’s (5000 – 40000)
- Lets look at the length of the emails. The average length calculated is the average number
    of meaningful words present in a mail:

```
This shows the spam messages(136.15) are usually shorter than ham messages(174.39).
```
- Plotting a pyramid plot for length of emails:


### -

- From the plot, we can see a few lengthy emails in HAM. Very rarely are spam mails
    extremely lengthy. This is quite possible because of a lot of emails are replied to /
    forwarded in HAM, the previous reply history is also present in such mails.

## Extracting features for training:

- Using TfidfVectorizer() to make normalized document term matrix.
- Save the vectorizer to file for use by test set.
- Generate labels for training the model. Using 0 for ham and 1 for spam
- Save the fitted vectorizer and labels to be used with Training set using pickle.
- We’re using “protocol = 4” in pickle dump to dump objects more that 4GB in size, as we
    can see the file size of the feature matrices are 4.6Gb and 9.5Gb respectively

## Training:

- Load the feature matrix and the labels


- Make a dictionary of models so we can easily compare accuracy among different models
- In LogisticRegression, we’re choosing “ovr” as multi_class because it’s optimized for
    binary classification(spam and ham). And so is “liblinear” solver
- Train and calculate the accuracy of different models in the dictionary. We’ll be using K-
    Fold cross-validation with k as 15 to cover as much data as possible. We’re using cross-
    validation to cover as much data as possible which a set validation would have left off.

```
◦ Naive-Bayes: 98.67971094992707%
◦ LinearSVC: 99.15090797779823%
◦ LogisticRegression: 98.59050654038633%
◦ Model with max accuracy: LinearSVC
```
- Compare the classification_report and accuracy of different models to pick the best
    model.
- This snippet will automatically choose the model with the best accuracy, train it and save
    it to a file for testing.


- Gauge chart ranging from 0.9-1.0 for comparing accuracy
- All accuracy calculations done till now is on the training set by cross-validation

## Testing the model:

- The test data-frame which was previously saved is loaded and feature matrix is prepared
    just like the training set
- The feature matrix is prepared on the previous Vectorizer that was saved, as the shape
    will change if a new one is prepared
- Open the saved model file and predict the test set


- Calculating the out of sample error for the model:
- we get out of sample error (RMS) and accuracy:

```
Out of sample Error: 1.0051%
Accuracy: 98.99%
```
- Using sklearn.metrics.classification_report to get more information about the test.
- ROC Curve for the classifier:


- AUC values and confusion matrix are calculated as follows:
- By looking at the confusion matrix, we get the following deductions:

```
Number of:
Ham messages classified correctly = 4821
Spam messages classified correctly = 5172
Ham messages misclassified = 78
Spam messages misclassified = 24
```
- For use in production, the error rate should be a maximum of 2.5%
    The current model might be biased towards the Enron organization, as many company
    specific terms are involved. So, for this particular organization this model can be used in
    production.
- However for a generic one or for other companies, such a bias must be removed and/or
    trained with a very huge dataset from multiple organizations which cater to a variety of
    use cases.


## Libraries/Tools used for the assignment:

- Pandas - Used for storing mails and their classification in a data-frame
- Numpy - Numpy arrays and sum operations
- sklearn - For classifiers, model selection, splitting, training, metrics
- pygal - plotting
- matplotlib - plotting ROC graph
- nltk - For getting stopwords
- Pickle - Saving and loading files
- VSCode - IDE and debugging

## APPENDIX – Running the python script

Initial steps:

Place all the pre-processed ham files in dataset/allham/ directory and spam files in
dataset/allspam/ directory.
There are two approaches:

1. If you have VSCode installed:
    It’s fairly simple as there is an attached _launch.json_ file attached. Open the directory in
    vscode and start debugging. Choose the step you want to execute. Previous steps must be
    executed in order to get the required files for processing the latter.


2. If not:
    You need to pass command line arguments(which is the step) to the python file.

```
Eg:,
```
```
NOTE: This might not work on Mac OSX because of it not being able to resolve relative
paths.
```

