#include <boost/python.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <string>
#include <unordered_map> 
#include "src/node.h"
#include "src/vpylm.h"
#include "src/vocab.h"
using namespace boost;

void split_word_by(const wstring &str, wchar_t delim, vector<wstring> &elems){
	elems.clear();
	wstring item;
	for(wchar_t ch: str){
		if (ch == delim){
			if (!item.empty()){
				elems.push_back(item);
			}
			item.clear();
		}
		else{
			item += ch;
		}
	}
	if (!item.empty()){
		elems.push_back(item);
	}
}

template<class T>
python::list list_from_vector(vector<T> &vec){  
	 python::list list;
	 typename vector<T>::const_iterator it;

	 for(it = vec.begin(); it != vec.end(); ++it)   {
		  list.append(*it);
	 }
	 return list;
}

class PyVPYLM{
public:
	VPYLM* _vpylm;
	Vocab* _vocab;
	vector<vector<id>> _dataset_train;
	vector<vector<id>> _dataset_test;
	vector<vector<int>> _prev_depths_for_data;
	vector<int> _rand_indices;
	// 統計
	unordered_map<id, int> _word_count;
	int _sum_word_count;
	bool _gibbs_first_addition;
	bool _vpylm_loaded;
	bool _is_ready;
	PyVPYLM(){
		setlocale(LC_CTYPE, "ja_JP.UTF-8");
		ios_base::sync_with_stdio(false);
		locale default_loc("ja_JP.UTF-8");
		locale::global(default_loc);
		locale ctype_default(locale::classic(), default_loc, locale::ctype); //※
		wcout.imbue(ctype_default);
		wcin.imbue(ctype_default);
		
		_vpylm = new VPYLM();
		_vocab = new Vocab();
		_gibbs_first_addition = true;
		_sum_word_count = 0;
		_vpylm_loaded = false;
		_is_ready = false;
	}
	~PyVPYLM(){
		delete  _vpylm;
		delete  _vocab;
	}
	bool load_textfile(string filename, double train_split_ratio){
		wifstream ifs(filename.c_str());
		wstring sentence;
		if (ifs.fail()){
			return false;
		}
		vector<wstring> lines;
		while (getline(ifs, sentence) && !sentence.empty()){
			if (PyErr_CheckSignals() != 0) {		// ctrl+cが押されたかチェック
				return false;
			}
			lines.push_back(sentence);
		}
		vector<int> rand_indices;
		for(int i = 0;i < lines.size();i++){
			rand_indices.push_back(i);
		}
		int train_split = lines.size() * train_split_ratio;
		shuffle(rand_indices.begin(), rand_indices.end(), sampler::mt);	// データをシャッフル
		for(int i = 0;i < rand_indices.size();i++){
			wstring &sentence = lines[rand_indices[i]];
			if(i < train_split){
				add_train_data(sentence);
			}else{
				add_test_data(sentence);
			}
		}
		return true;
	}
	void add_train_data(wstring sentence){
		_add_data_to(sentence, _dataset_train);
	}
	void add_test_data(wstring sentence){
		_add_data_to(sentence, _dataset_test);
	}
	void _add_data_to(wstring &sentence, vector<vector<id>> &dataset){
		vector<wstring> words;
		split_word_by(sentence, L' ', words);	// スペースで分割
		if(words.size() > 0){
			vector<id> token_ids;
			token_ids.push_back(ID_BOS);
			for(auto word: words){
				if(word.size() == 0){
					continue;
				}
				id token_id = _vocab->add_string(word);
				token_ids.push_back(token_id);
				_word_count[token_id] += 1;
				_sum_word_count += 1;
			}
			token_ids.push_back(ID_EOS);
			dataset.push_back(token_ids);
		}
	}
	void compile(){
		for(int data_index = 0;data_index < _dataset_train.size();data_index++){
			vector<id> &token_ids = _dataset_train[data_index];
			vector<int> prev_depths(token_ids.size(), -1);
			_prev_depths_for_data.push_back(prev_depths);
		}
		_is_ready = true;
	}
	void set_g0(double g0){
		_vpylm->_g0 = g0;
	}
	void load(string dirname){
		_vocab->load(dirname + "/vpylm.vocab");
		if(_vpylm->load(dirname + "/vpylm.model")){
			_gibbs_first_addition = false;
			_vpylm_loaded = true;
		}
	}
	void save(string dirname){
		_vocab->save(dirname + "/vpylm.vocab");
		_vpylm->save(dirname + "/vpylm.model");
	}
	void perform_gibbs_sampling(){
		assert(_is_ready == true);
		assert(_vpylm_loaded == false);	// 再学習は現在非対応
		if(_rand_indices.size() != _dataset_train.size()){
			_rand_indices.clear();
			for(int data_index = 0;data_index < _dataset_train.size();data_index++){
				_rand_indices.push_back(data_index);
			}
		}
		shuffle(_rand_indices.begin(), _rand_indices.end(), sampler::mt);	// データをシャッフル
		for(int n = 0;n < _dataset_train.size();n++){
			if (PyErr_CheckSignals() != 0) {		// ctrl+cが押されたかチェック
				return;
			}
			int data_index = _rand_indices[n];
			vector<id> &token_ids = _dataset_train[data_index];
			vector<int> &prev_depths = _prev_depths_for_data[data_index];
			for(int token_t_index = 1;token_t_index < token_ids.size();token_t_index++){
				if(_gibbs_first_addition == false){
					int prev_depth = prev_depths[token_t_index];
					assert(prev_depth >= 0);
					_vpylm->remove_customer_at_timestep(token_ids, token_t_index, prev_depth);
				}
				int new_depth = _vpylm->sample_depth_at_timestep(token_ids, token_t_index);
				// 性能を向上させるならコメントを外す
				// if(token_t_index == 1){
				// 	new_depth = 1;
				// }
				_vpylm->add_customer_at_timestep(token_ids, token_t_index, new_depth);
				prev_depths[token_t_index] = new_depth;
			}
		}
		_gibbs_first_addition = false;
	}
	void remove_all_data(){
		for(int data_index = 0;data_index < _dataset_train.size();data_index++){
			if (PyErr_CheckSignals() != 0) {		// ctrl+cが押されたかチェック
				return;
			}
			vector<id> &token_ids = _dataset_train[data_index];
			vector<int> &prev_depths = _prev_depths_for_data[data_index];
			for(int token_t_index = 1;token_t_index < token_ids.size();token_t_index++){
				int prev_depth = prev_depths[token_t_index];
				assert(prev_depth >= 0);
				_vpylm->remove_customer_at_timestep(token_ids, token_t_index, prev_depth);
			}
		}
	}
	int get_num_train_data(){
		return _dataset_train.size();
	}
	int get_num_test_data(){
		return _dataset_test.size();
	}
	int get_num_nodes(){
		return _vpylm->get_num_nodes();
	}
	int get_num_customers(){
		return _vpylm->get_num_customers();
	}
	int get_num_types_of_words(){
		return _word_count.size();
	}
	int get_num_words(){
		return _sum_word_count;
	}
	int get_vpylm_depth(){
		return _vpylm->get_depth();
	}
	id get_bos_id(){
		return ID_BOS;
	}
	id get_eos_id(){
		return ID_EOS;
	}
	python::list count_tokens_of_each_depth(){
		unordered_map<int, int> counts_by_depth;
		_vpylm->count_tokens_of_each_depth(counts_by_depth);

		// ソート
		std::map<int, int> sorted_counts_by_depth(counts_by_depth.begin(), counts_by_depth.end());

		std::vector<int> counts;
		for(auto it = sorted_counts_by_depth.begin(); it != sorted_counts_by_depth.end(); ++it){
			counts.push_back(it->second);
		}
		return list_from_vector(counts);
	}
	python::list get_discount_parameters(){
		return list_from_vector(_vpylm->_d_m);
	}
	python::list get_strength_parameters(){
		return list_from_vector(_vpylm->_theta_m);
	}
	void sample_hyperparameters(){
		_vpylm->sample_hyperparams();
	}
	// データセット全体の対数尤度を計算
	double compute_log_Pdataset_train(){
		return _compute_log_Pdataset(_dataset_train);
	}
	double compute_log_Pdataset_test(){
		return _compute_log_Pdataset(_dataset_test);
	}
	double _compute_log_Pdataset(vector<vector<id>> &dataset){
		double log_Pdataset = 0;
		for(int data_index = 0;data_index < dataset.size();data_index++){
			if (PyErr_CheckSignals() != 0) {		// ctrl+cが押されたかチェック
				return 0;
			}
			vector<id> &token_ids = dataset[data_index];
			log_Pdataset += _vpylm->compute_log_Pw(token_ids);;
		}
		return log_Pdataset;
	}
	double compute_perplexity_train(){
		return _compute_perplexity(_dataset_train);
	}
	double compute_perplexity_test(){
		return _compute_perplexity(_dataset_test);
	}
	double _compute_perplexity(vector<vector<id>> &dataset){
		double log_Pdataset = 0;
		for(int data_index = 0;data_index < dataset.size();data_index++){
			if (PyErr_CheckSignals() != 0) {		// ctrl+cが押されたかチェック
				return 0;
			}
			vector<id> &token_ids = dataset[data_index];
			log_Pdataset += _vpylm->compute_log2_Pw(token_ids) / (token_ids.size() - 1);
		}
		return pow(2.0, -log_Pdataset / (double)dataset.size());
	}
	wstring generate_sentence(){
		std::vector<id> context_token_ids;
		context_token_ids.push_back(ID_BOS);
		for(int n = 0;n < 1000;n++){
			id next_id = _vpylm->sample_next_token(context_token_ids, _vocab->get_all_token_ids());
			if(next_id == ID_EOS){
				vector<id> token_ids(context_token_ids.begin() + 1, context_token_ids.end());
				return _vocab->token_ids_to_sentence(token_ids);
			}
			context_token_ids.push_back(next_id);
		}
		return _vocab->token_ids_to_sentence(context_token_ids);
	}
};

BOOST_PYTHON_MODULE(model){
	python::class_<PyVPYLM>("vpylm", python::init<>())
	.def("set_g0", &PyVPYLM::set_g0)
	.def("load_textfile", &PyVPYLM::load_textfile)
	.def("compile", &PyVPYLM::compile)
	.def("perform_gibbs_sampling", &PyVPYLM::perform_gibbs_sampling)
	.def("get_num_nodes", &PyVPYLM::get_num_nodes)
	.def("get_num_customers", &PyVPYLM::get_num_customers)
	.def("get_discount_parameters", &PyVPYLM::get_discount_parameters)
	.def("get_strength_parameters", &PyVPYLM::get_strength_parameters)
	.def("get_num_train_data", &PyVPYLM::get_num_train_data)
	.def("get_num_test_data", &PyVPYLM::get_num_test_data)
	.def("get_num_types_of_words", &PyVPYLM::get_num_types_of_words)
	.def("get_num_words", &PyVPYLM::get_num_words)
	.def("get_vpylm_depth", &PyVPYLM::get_vpylm_depth)
	.def("get_bos_id", &PyVPYLM::get_bos_id)
	.def("get_eos_id", &PyVPYLM::get_eos_id)
	.def("sample_hyperparameters", &PyVPYLM::sample_hyperparameters)
	.def("count_tokens_of_each_depth", &PyVPYLM::count_tokens_of_each_depth)
	.def("compute_log_Pdataset_train", &PyVPYLM::compute_log_Pdataset_train)
	.def("compute_log_Pdataset_test", &PyVPYLM::compute_log_Pdataset_test)
	.def("compute_perplexity_train", &PyVPYLM::compute_perplexity_train)
	.def("compute_perplexity_test", &PyVPYLM::compute_perplexity_test)
	.def("generate_sentence", &PyVPYLM::generate_sentence)
	.def("save", &PyVPYLM::save)
	.def("load", &PyVPYLM::load);
}