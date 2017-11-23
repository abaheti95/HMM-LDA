#include "Models/cpp_hmm_lda/model.h"
#include <map>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <sstream>
#include <string>

#include <ctime>

using namespace std;

void load_documents_to_model(HMMLDA &model, string &documents_filename) {
	ifstream file(documents_filename, ios::in);
	string str;
	int counter = 0;
	while(getline(file, str)) {
		stringstream ss(str);
		int num;
		vector<int> document;
		while(ss >> num) {
			document.push_back(num);
		}
		model.add_document(document);
		counter ++;
		// if(counter == 10000) break;
		if(counter%1000000 == 0) {
			cout << counter << endl;
		}
	}
	file.close();
}

void generate_labeling_for_input(HMMLDA &model, string &input_file, string &output_file, int num_topics) {
	ifstream infile(input_file, ios::in);
	ofstream outfile(output_file, ios::out);
	string str;
	int iterations = 10;
	// Pre-initialized vectors for relabeling
	vector<int> document, topic_assignment, class_assignment, num_words_in_document_assigned_to_topic;
	document.resize(1000);
	topic_assignment.resize(1000);
	class_assignment.resize(1000);
	num_words_in_document_assigned_to_topic.resize(num_topics);
	while(getline(infile, str)) {
		str = model.relabel_document(str, document, topic_assignment, class_assignment, num_words_in_document_assigned_to_topic, iterations);
		outfile << str << endl;
	}
	// Cleanup
	document.clear();
	topic_assignment.clear();
	class_assignment.clear();
	num_words_in_document_assigned_to_topic.clear();
	infile.close();
	outfile.close();
}

int main(int argc, char const *argv[]) {
	std::clock_t start;
	start = std::clock();
	// model parameter initializations
	int num_topics = 50;
	int num_classes = 17;
	// int num_classes = 3;			// topics only setting
	int start_word_id = 0;
	int end_word_id = 1;
	// double alpha = 50.0/num_topics;
	double alpha = 0.5;
	double beta = 0.3;			// topic words prior
	double gamma = 1;			// transition probabilities prior
	double delta = 0.1;		// Class words prior

	// your test
	string vocab_filename = "Data/20newsgroup/20news_vocab.txt";
	vector<string> vocab;
	HMMLDA model(vocab_filename, num_topics, num_classes, 
		start_word_id, end_word_id, alpha, beta, gamma, delta, "20news");
		// start_word_id, end_word_id, alpha, beta, gamma, delta, "20news-topics");
	string documents_filename = "Data/20newsgroup/20news_train_encoded.txt";
	load_documents_to_model(model, documents_filename);
	string class_assignments_file = "Results/20news-topics/500_results/class_assignments.txt";
	string topic_assignments_file = "Results/20news-topics/500_results/topic_assignments.txt";
	model.load_class_assignments(class_assignments_file);
	model.load_topic_assignments(topic_assignments_file);
	model.run_counts();

	string input_file = "Data/unlabelled/unlabelled_input.txt";
	string output_file = "Data/unlabelled/labelled_output.txt";
	generate_labeling_for_input(model, input_file, output_file, num_topics);
	std::cout << "Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << " s" << std::endl;
	return 0;
}
