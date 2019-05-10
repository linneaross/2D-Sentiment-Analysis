# Linnea Ross's Honors Addendum Project for LINGUIST392B at UMass Amherst (Spring 2019)

# This program runs by taking command line arguments.
# The order of such a command should be <python sentiment_analysis.py pos_train neg_train vol_train pas_train test_file> (optional)<test_values>

# import necessary resources
import sys
import re

# TRAIN MODEL ON POLARITY DATA
def polarity_training(pos, neg):
    # this function goes through the two respective polarity rating files and calculates the
    # probability per argmax(class) P(data|class)P(class) [Naive Bayes]
    polarity_words = {'POS': {}, 'NEG': {}}
    pos_freq = {}
    neg_freq = {}
    # initialize values that will increment with each loop to determine the number of words in each file
    # as well as the total number of words in the polarity files
    pos_doc = 0
    neg_doc = 0
    total_doc = 0

    # looping over every sentence in each training file and incrementing local and total frequencies
    for line in pos:
        #removing punctuation
        line = re.sub(r'[^\w\s]','',line)
        pos_doc +=1
        total_doc += 1
        words = line.split()
        for word in words:
            if word not in pos_freq:
                pos_freq[word] = 0
            pos_freq[word] += 1

    for line in neg:
        #removing punctuation
        line = re.sub(r'[^\w\s]','',line)
        neg_doc +=1
        total_doc +=1
        words = line.split()
        for word in words:
            if word not in neg_freq:
                neg_freq[word] = 0
            neg_freq[word] += 1

    # calculate the probabilities that any random sentence comes from each class (in essence, P(class))
    pos_prob = float(pos_doc)/float(total_doc)
    neg_prob = float(neg_doc)/float(total_doc)

    # construct a complete vocabulary of all words across both input files
    vocab = []
    for pos_key in pos_freq:
        vocab.append(pos_key)
    for neg_key in neg_freq:
        if neg_key not in vocab:
            vocab.append(neg_key)

    pos_summation = 0
    for key in pos_freq:
        #laplace-smoothed denominator for the likelihood calculation (i.e. P(doc|class))
        pos_summation = pos_summation + pos_freq[key] + 1

    neg_summation = 0
    for key in neg_freq:
        #laplace-smoothed denominator for the likelihood calculation (i.e. P(doc|class))
        neg_summation = neg_summation + neg_freq[key] + 1

    for word in vocab:
        if word not in pos_freq:
            pos_count = 0
        else:
            pos_count = pos_freq[word]

        # calculate with laplace smoothing and previously smoothed summation values
        likelihood = float(pos_count + 1)/float(pos_summation)

        # P(doc|class)P(class)
        end_prob = likelihood * pos_prob

        polarity_words['POS'][word] = end_prob

    for word in vocab:
        if word not in neg_freq:
            neg_count = 0
        else:
            neg_count = neg_freq[word]

        #calculate with laplace smoothing
        likelihood = float(neg_count + 1)/float(neg_summation)

        # P(doc|class)P(class)
        end_prob = likelihood * neg_prob

        polarity_words['NEG'][word] = end_prob

    return polarity_words

# TRAIN MODEL ON INTENSITY DATA
def intensity_training(vol, pas):
    # this function performs the exact same operation as the previous function, just with the different axis values
    intensity_words = {'VOL': {}, 'PAS': {}}
    vol_freq = {}
    pas_freq = {}
    # initialize values that will increment with each loop to determine the number of words in each file
    # as well as the total number of words in the polarity files
    vol_doc = 0
    pas_doc = 0
    total_doc = 0

    # looping over every sentence in each training file
    for line in vol:
        # removing punctuation
        line = re.sub(r'[^\w\s]','',line)
        vol_doc +=1
        total_doc += 1
        words = line.split()
        for word in words:
            if word not in vol_freq:
                vol_freq[word] = 0
            vol_freq[word] += 1

    for line in pas:
        # removing punctuation
        line = re.sub(r'[^\w\s]','',line)
        pas_doc +=1
        total_doc +=1
        words = line.split()
        for word in words:
            if word not in pas_freq:
                pas_freq[word] = 0
            pas_freq[word] += 1

    # calculate the probabilities that any random sentence comes from each class (in essence, P(class))
    vol_prob = float(vol_doc)/float(total_doc)
    pas_prob = float(pas_doc)/float(total_doc)

    # construct a complete vocabulary of all words across both input files
    vocab = []
    for vol_key in vol_freq:
        vocab.append(vol_key)
    for pas_key in pas_freq:
        if pas_key not in vocab:
            vocab.append(pas_key)

    vol_summation = 0
    for key in vol_freq:
        #laplace-smoothed denominator for the likelihood calculation (i.e. P(doc|class))
        vol_summation = vol_summation + vol_freq[key] + 1

    pas_summation = 0
    for key in pas_freq:
        #laplace-smoothed denominator for the likelihood calculation (i.e. P(doc|class))
        pas_summation = pas_summation + pas_freq[key] + 1

    for word in vocab:
        if word not in vol_freq:
            vol_count = 0
        else:
            vol_count = vol_freq[word]

        # calculate with laplace smoothing and previously smoothed summation values
        likelihood = float(vol_count + 1)/float(vol_summation)

        # P(doc|class)P(class)
        end_prob = likelihood * vol_prob

        intensity_words['VOL'][word] = end_prob

    for word in vocab:
        if word not in pas_freq:
            pas_count = 0
        else:
            pas_count = pas_freq[word]

        # calculate with laplace smoothing and previously smoothed summation values
        likelihood = float(pas_count + 1)/float(pas_summation)

        # P(doc|class)P(class)
        end_prob = likelihood * pas_prob

        intensity_words['PAS'][word] = end_prob

    return intensity_words

# LOAD TEST DATA
def load_testing(test_file):
    # this file is comprised of raw data
    # this function returns a list of the cleaned data
    data = []
    for line in test_file:
        line = re.sub(r'[^\w\s]','',line)
        data.append(line)
    return data

def load_test_val(test, values_file):
    # same indexes as the last function, but used to check precision, recall, and f-score
    data = {}
    vals = []
    i = 0
    for line in values_file:
        vals.append(line)
    for item in range(len(test)):
        data[test[item]] = vals[i]
        i += 1
    return data

# NAIVE BAYES CLASSIFICATION ALGORITHM
def classify(polarity, intensity, test):
    temp_dict = {}
    quadrants = {}
    # these are the only 4 keys in the final result dictionary
    # their values are lists of the data classified into those quadrants
    quadrants['POS|VOL'] = []
    quadrants['POS|PAS'] = []
    quadrants['NEG|VOL'] = []
    quadrants['NEG|PAS'] = []

    # instantiate sub-dictionaries
    for line in test:
        temp_dict[line] = {'POL': None, 'INTENS': None}

    # find best polarity class for each line
    for line in test:
        string_prob = 1
        words = line.split()
        best_prob = 0
        best_class = None
        for key in polarity:
            line_prob = 1
            for word in words:
                line_prob = line_prob * polarity[key][word]
            if line_prob > best_prob:
                best_prob = line_prob
                best_class = key
        temp_dict[line]['POL'] = best_class

    # find best intensity class for each line
    for line in test:
        string_prob = 1
        words = line.split()
        best_prob = 0
        best_class = None
        for key in intensity:
            line_prob = 1
            for word in words:
                line_prob = line_prob * intensity[key][word]
            if line_prob > best_prob:
                best_prob = line_prob
                best_class = key
        temp_dict[line]['INTENS'] = best_class

    # add lines to appropriate dictionaries
    for key in temp_dict:
        quadrants['' + temp_dict[key]['POL'] + '|' + temp_dict[key]['INTENS'] + ''].append(key)

    # print output
    for quadrant in quadrants:
        print quadrant + ' sentences:'
        for item in quadrants[quadrant]:
            print item

    return quadrants

def evaluate(quadrants, test_values):
    # calculates and prints precision, recall, and f-score for polarity and intensity
    polar_fp = 0.0
    polar_tp = 0.0
    polar_fn = 0.0
    for quadrant in quadrants:
        for item in quadrants[quadrant]:
            if quadrant[0:3] != test_values[item][0:3]:
                if test_values[item][0:3] == 'POS':
                    polar_fp += 1.0
                if test_values[item][0:3] == 'NEG':
                    polar_fn += 1.0
            if quadrant[0:3] == test_values[item][0:3]:
                polar_tp += 1.0
    intens_fp = 0.0
    intens_tp = 0.0
    intens_fn = 0.0
    for quadrant in quadrants:
        for item in quadrants[quadrant]:
            if quadrant[4:7] != test_values[item][4:7]:
                if test_values[item][4:7] == 'VOL':
                    intens_fp += 1.0
                if test_values[item][4:7] == 'PAS':
                    intens_fn += 1.0
            if quadrant[4:7] == test_values[item][4:7]:
                intens_tp += 1.0
    print 'polarity precision: ' + str(polar_tp/(polar_tp+polar_fp)) + ', recall: ' + str(polar_tp/(polar_tp+polar_fn)) + ', f-score: ' + str((2*(polar_tp/(polar_tp+polar_fp))*(polar_tp/(polar_tp+polar_fn)))/(polar_tp/(polar_tp+polar_fp)+polar_tp/(polar_tp+polar_fn)))
    print 'intensity precision: ' + str(intens_tp/(intens_tp+intens_fp)) + ', recall: ' + str(intens_tp/(intens_tp+intens_fn)) + ', f-score: ' + str((2*(intens_tp/(intens_tp+intens_fp))*(intens_tp/(intens_tp+intens_fn)))/(intens_tp/(intens_tp+intens_fp)+intens_tp/(intens_tp+intens_fn)))

def main():
    # collect and open files
    pos = open(sys.argv[1])
    neg = open(sys.argv[2])
    vol = open(sys.argv[3])
    pas = open(sys.argv[4])
    test = load_testing(open(sys.argv[5]))
    # train the model on polarity
    polarity = polarity_training(pos, neg)
    # train the model on intensity
    intensity = intensity_training(vol, pas)
    # classify each test sentence into a quadrant
    classify(polarity, intensity, test)
    # if sixth argument, evaluate precision, recall, f-score
    if len(sys.argv) == 7:
        val = open(sys.argv[6])
        values = load_test_val(test, val)
        evaluate(classify(polarity, intensity, test), values)
    return

if __name__ == '__main__':
    main()
