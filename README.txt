This program takes a Naïve Bayes approach to parsing sentiment analysis. Where this program diverges from most other
sentiment analysis programs is that it not only classifies a datum's emotional polarity (i.e. how positive or negative
the emotion is), but it also classifies its emotional intensity (i.e. how volatile or passive the emotion is).

These can be considered as on two separate axes: Polarity is the y-axis, and intensity is the x-axis. This creates four
quadrants into which a datum can be classified; hence, the output will state 'POS|VOL sentences' which indicates
high polarity and high intensity. The matrix is shown below:

                    |
                    |
                    |
      POS|PAS       |      POS|VOL
                    |
                    |
                    |
____________________|_______________________
                    |
                    |
                    |
      NEG|PAS       |      NEG|VOL
                    |
                    |
                    |
                    |

Explanation of functions:

polarity_training: this function trains the model on the polarity axis (that is, it learns the probability that each word
      belongs to a certain class and uses the Bayes formula to calculate this value and assign it to a dictionary)

intensity_training: this function does the same as the above, only this time working with the x-axis values

load_testing: this function puts the testing input into an easily traversable list

load_test_val: this function loads the optional file of testing values, which are the actual categories the test data
      should be classified as. This can be used to evaluate precision, recall, and f-score for each axises values

classify: this function is the one that compares the probabilities of the test strings according to the model. The higher
      probability classes are the ones into which the sentence is categorized, first by polarity, then by intensity

evaluate: this function counts false positives, false negatives, and true positives to calculate precision, recall, and
      f-score for polarity and for intensity

This function utilizes the testing data from: http://www.cs.cornell.edu/people/pabo/movie-review-data/
Polarity values were pre-classified, but I classified intensity values by hand (likely contributing to some margin of error)
