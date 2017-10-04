#include "Models/cpp_hmm_lda/model.h"
// #include "Models/cpp_hmm_lda/model.cpp"
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
		// cout << document.size() << endl;
		model.add_document(document);
		counter ++;
		if(counter == 10000) break;
		if(counter%1000000 == 0) {
			cout << counter << endl;
		}
	}
}

int main(int argc, char const *argv[]) {
	std::clock_t start;
	start = std::clock();
	// your test
	string vocab_filename = "Data/4M.es.dict";
	vector<string> vocab;
	int vocab_size = read_vocabulary(vocab_filename, vocab);
	HiddenMarkovModelLatentDirichletAllocation model(vocab_size, 20, 5, 0.1, 0.1, 0.1, 0.1);
	string documents_filename = "Data/4M.es.train";
	load_documents_to_model(model, documents_filename);
	model.run_counts();
	cout << "Training Started" << endl;
	model.train(1000,50);
	std::cout << "Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << " s" << std::endl;
	return 0;
}
