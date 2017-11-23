cpp_hmm_lda_trainer: cpp_hmm_lda_trainer.cpp model.o
	g++ -std=c++11 -g -Wall Models/cpp_hmm_lda/model.o cpp_hmm_lda_trainer.cpp -o cpp_hmm_lda_trainer.out
cpp_hmm_lda_labeler: cpp_hmm_lda_labeler.cpp model.o
	g++ -std=c++11 -g -Wall Models/cpp_hmm_lda/model.o cpp_hmm_lda_labeler.cpp -o cpp_hmm_lda_labeler.out
model.o:
	g++ -std=c++11 -g -Wall -c Models/cpp_hmm_lda/model.cpp -o Models/cpp_hmm_lda/model.o
clean:
	rm cpp_hmm_lda_trainer.out cpp_hmm_lda_labeler.out