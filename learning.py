import pandas as pd
import math
import random
#input: takes in dataframe object (list of lists)
#output: dataframe object with last column converted to ints
#output: dictionary storing the deleted names of columns with mapping order
def simplify_csv(data): 
    counter = 0
    mapping = {}
    for i in range(len(data)):
        t = data.iat[i, -1]
        if isinstance(t, str):
            if t not in mapping:
                mapping[t] = counter
                counter += 1
            data.iat[i, -1] = mapping[t]
    return data, mapping

#input: takes in dataframe object (list of lists)
#output: dictionary of dataframe objects
#dictionary is of size 3, with each element being a dataframe that stores data of a specific class
def separate_csv(data):
    organized = {}
    for i in range(len(data)):
        t = data.iat[i, -1]
        if t not in organized:
            organized[t] = []
        organized[t].append(data.iloc[i, :].tolist())
    return organized

def mean(column):
    return sum(column) / len(column)

def stdev(column):
    avg = mean(column)
    variance = sum((x - avg) ** 2 for x in column) / (len(column) - 1)
    return math.sqrt(variance)

#input: dataframe (list of lists)
#output: list of lists
#each element of summaries represents a column
#and contains list, which includes, mean, std dev, and total elements in df
def get_columnval(data):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*data)]
    del summaries[-1]
    return summaries

def get_formulas(organized): #return mean, stdev, number of rows in class for each class
    summaries = {}
    for key in organized:
        summaries[key] = get_columnval(organized[key])
    return summaries #3-element dictionary with each element being list of lists

#Assume data x is drawn from Gaussian distribution
#Calculate its probability given mean and std dev
def calculate_probability(x, mean, stdev):
    exponent = math.exp(-((x-mean)**2 / (2 * stdev**2 )))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

#Input: 3-element dictionary with each element being list of lists
#Input: row from dataframe
#Output: 3-element dictionary containing probability row belongs to each class
#Bayes-formula: P(Class|Data) = P(Data|Class) * P(Class) / P(Data)
#Assume all columns or data inputs are independent
def calculate_row(overall, row):
    num_of_rows = 0
    for i in overall:
        num_of_rows += overall[i][0][2]
    probabilities = {}
    for i in range(len(overall)):
        class_total_rows = overall[i][0][2]
        probabilities[i] = float(class_total_rows)/num_of_rows #P(class)
        for j in range(len(overall[i])):
            mean, stdev, count = overall[i][j]
            #independent inputs, probability of each input is calculated seperately and multiplied
            probabilities[i] *= calculate_probability(row.iloc[j], mean, stdev)
    return probabilities

def test_accuracy(actual, predicted):
    counter = 0
    for i in range(len(actual)):
        #print(actual[i])
        #print(predicted[i])
        if(actual[i] == predicted[i]):
            counter = counter + 1
    return counter/float(len(actual)) * 100.0

def evaluate_algorithm(data):
    folds = k_fold(data, 5)
    scores = list()
    for i in range(len(folds)):
        train = pd.DataFrame()
        for j in range(len(folds)):
            if(j != i):
                train = pd.concat([train, folds[j]])
        test = folds[i]
        predicted = naive_bayes(train, test)
        actual = folds[i]['class']
        accuracy = test_accuracy(actual, predicted)
        #print(accuracy)
       # actual = [row[-1] for row in i]
        #accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

def k_fold(data, folds):
    temp = data.copy()  # Create a copy to avoid modifying the original data
    result = list()
    size = int(len(data) / folds)
    for i in range(folds):
        c = pd.DataFrame()
        while len(c) < size:
            index = random.randrange(len(temp))
            row = temp.iloc[index, :]
            c = pd.concat([c, row.to_frame().T], ignore_index=True)
            temp = temp.drop(index).reset_index(drop=True)  # Drop and reset index
        result.append(c)
    return result

def naive_bayes(train, test):
    temp = separate_csv(train)
    summarize = get_formulas(temp)
    result = list()
    for i in range(len(test)):
        temp = calculate_row(summarize, test.iloc[i])
        best_label, best_prob = None, -1
        for class_value, probability in temp.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        result.append(best_label)
    return result
def predict_row(data, index):
    row_by_label = data.iloc[index]
    df = pd.DataFrame([row_by_label])
    result = naive_bayes(data, df)
    print(result)
    return
data = pd.read_csv("iris1.csv")
data, mapping = simplify_csv(data)
organized = separate_csv(data)
overall = get_formulas(organized)
scores = evaluate_algorithm(data)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
predict_row(data, 0)








