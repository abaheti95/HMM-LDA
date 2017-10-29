import os
from Evaluator import evaluator

MAX_DOCS = 1000000000

# Read the data and output of the model
DATA_FOLDER = os.path.join("Data", "20newsgroup")
RESULTS_FOLDER = os.path.join("Results", "20news", "2500_results")
# RESULTS_FOLDER = os.path.join("Results", "20news-topics", "500_results")

vocab_filename = os.path.join(DATA_FOLDER, "20news_vocab.txt")
data_filename = os.path.join(DATA_FOLDER, "20news_train_encoded.txt")
class_assignments_file = os.path.join(RESULTS_FOLDER, "class_assignments.txt")
topic_assignments_file = os.path.join(RESULTS_FOLDER, "topic_assignments.txt")
classes_statistics_file = os.path.join(RESULTS_FOLDER, "class_statistics.tsv")
topics_statistics_file = os.path.join(RESULTS_FOLDER, "topic_statistics.tsv")
print("Loading Vocabulary")
evaluator.load_vocabulary(vocab_filename)
print("Loading Data")
evaluator.load_documents(data_filename, MAX_DOCS)
print("Loading Class Assignments")
evaluator.load_classes(class_assignments_file, MAX_DOCS)
print("Loading Topic Assignments")
evaluator.load_topics(topic_assignments_file, MAX_DOCS)
print("Saving Classes Statistics")
evaluator.save_classes_to_tsv(classes_statistics_file)
print("Saving Topics Statistics")
evaluator.save_topics_to_tsv(topics_statistics_file)
