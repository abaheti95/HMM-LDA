# This file will read the files from the results folder and input data and emit the topic and class distribution csv.
import operator
import os
import numpy as np

vocab = dict()
vocab_lookup = list()
def load_vocabulary(vocab_filename):
	global vocab, vocab_lookup
	with open(vocab_filename, "r") as vocab_reader:
		i = 0
		for line in vocab_reader:
			if not line:
				continue
			line = line.split()
			vocab[line[0]] = i
			vocab_lookup.append(line[0])
			i += 1

documents = list()
def load_documents(data_filename, MAX_DOCS = 1000000000):
	global documents
	with open(data_filename, "r") as train_data_reader:
		num_documents = 0
		for idx, line in enumerate(train_data_reader):
			if not line:
				continue
			# each line is a conversation triplet which is also equivalent to a document in our case.
			line = line.replace("\t", " ")
			document = np.fromstring(line, dtype=np.int64, sep=' ')
			documents.append(document)
			num_documents += 1
			if num_documents == MAX_DOCS:
				break
			if num_documents % 1000000 == 0:
				print num_documents
	documents = np.array(documents)


# List of dictionaries
# Each dictionary will store all the words in that topic along with thier total counts and percentages as tuple values
topics = list()
classes = list()
class_assignments = list()
classes_size = list()

def load_classes(class_assignments_file, MAX_DOCS = 1000000000):
	global classes, class_assignments, classes_size
	with open(class_assignments_file, "r") as class_assignments_reader:
		first_line = next(class_assignments_reader)
		# first number is the number of documents and second number is number of classes
		num_classes = int(first_line.split()[1])
		for i in range(num_classes):
			classes.append(dict())
		classes_size = num_classes * [0]
		doc_idx = 0
		for line in class_assignments_reader:
			if not line:
				continue
			doc_class_assignments = np.fromstring(line.strip(), dtype=np.int32, sep=' ')
			class_assignments.append(list())
			for word_idx, assignment in enumerate(doc_class_assignments):
				class_assignments[doc_idx].append(assignment)
				word = vocab_lookup[documents[doc_idx][word_idx]]
				classes[assignment].setdefault(word, 0)
				classes[assignment][word] += 1
				classes_size[assignment] += 1
			doc_idx += 1
			if doc_idx == MAX_DOCS:
				break;
			if doc_idx % 1000000 == 0:
				print(doc_idx)

def load_topics(topic_assignments_file, MAX_DOCS = 1000000000):
	global topics, topics_size
	with open(topic_assignments_file, "r") as topic_assignments_reader:
		first_line = next(topic_assignments_reader)
		# first number is the number of documents and second number is number of topics
		num_topics = int(first_line.split()[1])
		for i in range(num_topics):
			topics.append(dict())
		topics_size = num_topics * [0]
		doc_idx = 0
		for line in topic_assignments_reader:
			if not line:
				continue
			doc_topic_assignments = np.fromstring(line.strip(), dtype=np.int32, sep=' ')
			for word_idx, assignment in enumerate(doc_topic_assignments):
				if class_assignments[doc_idx][word_idx] != 0:
					continue
				word = vocab_lookup[documents[doc_idx][word_idx]]
				topics[assignment].setdefault(word, 0)
				topics[assignment][word] += 1
				topics_size[assignment] += 1
			doc_idx += 1
			if doc_idx == MAX_DOCS:
				break;
			if doc_idx % 1000000 == 0:
				print(doc_idx)

def save_classes_to_tsv(classes_statistics_file):
	with open(classes_statistics_file, "w") as class_stats_writer:
		for class_idx, class_dict in enumerate(classes):
			# Write the percentage of the words
			classes_dict_sorted = reversed(sorted(class_dict.items(), key=operator.itemgetter(1)))
			class_stats_writer.write("C{}\t".format(class_idx))
			for word, count in classes_dict_sorted:
				if classes_size[class_idx] != 0:
					class_stats_writer.write(str(float(count)/classes_size[class_idx]*100) + "%\t")
				else:
					class_stats_writer.write("0\t")
			class_stats_writer.write("\n")
			# Write the actual counts of the words
			classes_dict_sorted = reversed(sorted(class_dict.items(), key=operator.itemgetter(1)))
			# class_stats_writer.write("C{}\t".format(class_idx))
			class_stats_writer.write("{}\t".format(classes_size[class_idx]))
			for word, count in classes_dict_sorted:
				class_stats_writer.write(str(count) + "\t")
			class_stats_writer.write("\n")
			# Write the class words in the third line
			classes_dict_sorted = reversed(sorted(class_dict.items(), key=operator.itemgetter(1)))
			class_stats_writer.write("C{}\t".format(class_idx))
			for word, count in classes_dict_sorted:
				class_stats_writer.write(word + "\t")
			class_stats_writer.write("\n")

def save_topics_to_tsv(topics_statistics_file):
	with open(topics_statistics_file, "w") as topic_stats_writer:
		for topic_idx, topic_dict in enumerate(topics):
			# Write the percentage of the words
			topics_dict_sorted = reversed(sorted(topic_dict.items(), key=operator.itemgetter(1)))
			topic_stats_writer.write("T{}\t".format(topic_idx))
			for word, count in topics_dict_sorted:
				if topics_size[topic_idx] != 0:
					topic_stats_writer.write(str(float(count)/topics_size[topic_idx]*100) + "%\t")
				else:
					topic_stats_writer.write("0\t")
			topic_stats_writer.write("\n")
			# Write the actual counts of the words
			topics_dict_sorted = reversed(sorted(topic_dict.items(), key=operator.itemgetter(1)))
			# topic_stats_writer.write("T{}\t".format(topic_idx))
			topic_stats_writer.write("{}\t".format(topics_size[topic_idx]))
			for word, count in topics_dict_sorted:
				topic_stats_writer.write(str(count) + "\t")
			topic_stats_writer.write("\n")
			# Write the topic words in the third line
			topics_dict_sorted = reversed(sorted(topic_dict.items(), key=operator.itemgetter(1)))
			topic_stats_writer.write("T{}\t".format(topic_idx))
			for word, count in topics_dict_sorted:
				topic_stats_writer.write(word + "\t")
			topic_stats_writer.write("\n")



