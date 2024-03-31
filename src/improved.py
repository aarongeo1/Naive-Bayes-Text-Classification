import argparse
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
import nltk

# Download stopwords from nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}
        self.word_probs = {}
        self.vocab = set()

    def train(self, data):
        self.classes = data['relation'].unique()
        total_samples = len(data)

        for class_label in self.classes:
            class_samples = data[data['relation'] == class_label]
            self.class_probs[class_label] = len(class_samples) / total_samples

            word_counts = defaultdict(int)
            total_words_in_class = 0

            for text in class_samples['tokens']:
                words = self.preprocess(text)
                for word in words:
                    word_counts[word] += 1
                    self.vocab.add(word)
                total_words_in_class += len(words)

            self.word_probs[class_label] = {word: (word_counts[word] + 1) / (total_words_in_class + len(self.vocab))
                                            for word in self.vocab}

    def preprocess(self, text):
        words = str(text).lower().split()
        words = [stemmer.stem(word) for word in words if word not in stop_words]
        return words

    def predict(self, text):
        best_class = None
        max_posterior = float('-inf')

        for class_label in self.classes:
            class_prob = np.log(self.class_probs[class_label])
            word_prob = 0

            for word in self.preprocess(text):
                word_prob += np.log(self.word_probs[class_label].get(word, 1 / (len(self.vocab) + 1)))

            posterior = class_prob + word_prob
            if posterior > max_posterior:
                max_posterior = posterior
                best_class = class_label

        return best_class

def cross_validate(data, n_splits=3):
    fold_size = len(data) // n_splits
    accuracies = []

    for fold in range(n_splits):
        validation_start = fold * fold_size
        validation_end = (fold + 1) * fold_size

        if fold == n_splits - 1:
            validation_end = len(data)

        train = pd.concat([data[:validation_start], data[validation_end:]])
        test = data[validation_start:validation_end]

        nb_classifier = NaiveBayesClassifier()
        nb_classifier.train(train)

        correct_predictions = 0
        for _, row in test.iterrows():
            prediction = nb_classifier.predict(row['tokens'])
            if prediction == row['relation']:
                correct_predictions += 1

        accuracies.append(correct_predictions / len(test))

    return np.mean(accuracies)

def calculate_confusion_matrix(true_labels, predictions, classes):
    cm = pd.DataFrame(np.zeros((len(classes), len(classes)), dtype=int),
                      index=classes, columns=classes)

    for true, pred in zip(true_labels, predictions):
        cm.loc[true, pred] += 1

    return cm


def calculate_precision_recall(cm):
    precision = {}
    recall = {}
    for label in cm.columns:
        tp = cm.loc[label, label]
        fp = cm.sum(axis=0)[label] - tp
        fn = cm.sum(axis=1)[label] - tp
        precision[label] = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall[label] = tp / (tp + fn) if (tp + fn) != 0 else 0
    return precision, recall

def calculate_micro_macro_averages(cm):
    all_tp = sum([cm.loc[label, label] for label in cm.columns])
    all_fp = sum([cm.sum(axis=0)[label] - cm.loc[label, label] for label in cm.columns])
    all_fn = sum([cm.sum(axis=1)[label] - cm.loc[label, label] for label in cm.columns])

    micro_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) != 0 else 0
    micro_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) != 0 else 0
    macro_precision = np.mean([precision for precision in calculate_precision_recall(cm)[0].values()])
    macro_recall = np.mean([recall for recall in calculate_precision_recall(cm)[1].values()])

    return micro_precision, micro_recall, macro_precision, macro_recall

def main():
    parser = argparse.ArgumentParser(description='Train and test a Naive Bayes classifier.')
    parser.add_argument('--train', help='Path to the training file')
    parser.add_argument('--test', help='Path to the test file')
    parser.add_argument('--output', help='Path to the output file')

    args = parser.parse_args()

    try:
        train_data = pd.read_csv(args.train)
        test_data = pd.read_csv(args.test)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Cross Validation for training accuracy
    training_accuracy = cross_validate(train_data)
    print(f"Training accuracy mean (3-fold cross-validation): {training_accuracy:.2f}")
    testing_accuracy = cross_validate(test_data)
    print(f"Testing accuracy mean (3-fold cross-validation): {testing_accuracy:.2f}")

 # Training and predicting
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.train(train_data)

    predictions = test_data['tokens'].apply(nb_classifier.predict)
    test_data['PredictedClass'] = predictions.values
    
    # Directly using predictions as a list for comparison
    test_relations = test_data['relation'].tolist()
    predictions_list = predictions.tolist()

    # Evaluating the model
    classes = train_data['relation'].unique()
    cm = calculate_confusion_matrix(test_relations, predictions_list, classes)
    precision, recall = calculate_precision_recall(cm)
    micro_precision, micro_recall, macro_precision, macro_recall = calculate_micro_macro_averages(cm)

    print("Confusion Matrix:\n", cm)
    print("Precision per class:", precision)
    print("Recall per class:", recall)
    print("Micro-Averaged Precision, Recall:", (micro_precision, micro_recall))
    print("Macro-Averaged Precision, Recall:", (macro_precision, macro_recall))

    # Output file
    output_df = test_data[['row_id', 'relation', 'PredictedClass']]
    output_df.columns = ['row_id', 'original_label', 'output_label']
    output_df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")

if __name__ == '__main__':
    main()