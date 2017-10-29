#include "Models/cpp_hmm_lda/model.h"
#include <map>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <sstream>
#include <string>

#include <ctime>

using namespace std;
// https://stackoverflow.com/a/236803/4535284
template<typename Out>
void split(const std::string &s, char delim, Out result) {
	std::stringstream ss;
	ss.str(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		*(result++) = item;
	}
}

std::vector<std::string> split(const std::string &s, char delim) {
	std::vector<std::string> elems;
	split(s, delim, std::back_inserter(elems));
	return elems;
}


// Read the vocabulary from the file into an unordered_map
int read_vocabulary(string &vocab_filename, vector<string> &vocab) {
	ifstream file(vocab_filename, ios::in);
	string str;
	while(getline(file, str)) {
		vector<string> str_spl = split(str, ' ');
		vocab.push_back(str_spl[0]);
	}
	return vocab.size();
}

void load_documents_to_model(HiddenMarkovModelLatentDirichletAllocation &model, string &documents_filename) {
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
	int vocab_size = read_vocabulary(vocab_filename, vocab);
	HiddenMarkovModelLatentDirichletAllocation model(vocab_size, num_topics, num_classes, 
		start_word_id, end_word_id, alpha, beta, gamma, delta, "20news");
		// start_word_id, end_word_id, alpha, beta, gamma, delta, "20news-topics");
	string documents_filename = "Data/20newsgroup/20news_train_encoded.txt";
	load_documents_to_model(model, documents_filename);
	string topic_assignments_file = "Results/20news-topics/500_results/topic_assignments.txt";
	model.load_topic_assignments(topic_assignments_file);
	model.run_counts();
	cout << "Training Started" << endl;
	model.train(2500,100);
	std::cout << "Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << " s" << std::endl;
	return 0;
}
