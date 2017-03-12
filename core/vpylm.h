#ifndef _vpylm_
#define _vpylm_
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <vector>
#include <cassert>
#include <unordered_map> 
#include <fstream>
#include "sampler.h"
#include "node.h"

class VPYLM{
private:
	friend class boost::serialization::access;
	template <class Archive>
	void serialize(Archive& archive, unsigned int version)
	{
		static_cast<void>(version); // No use
		archive & _root;
		archive & _g0;
		archive & _beta_stop;
		archive & _beta_pass;
		archive & _d_m;
		archive & _theta_m;
		archive & _a_m;
		archive & _b_m;
		archive & _alpha_m;
		archive & _beta_m;
	}
public:
	Node* _root;				// 文脈木のルートノード
	int _depth;					// 最大の深さ
	double _g0;					// ゼログラム確率

	// 深さmのノードに関するパラメータ
	vector<double> _d_m;		// Pitman-Yor過程のディスカウント係数
	vector<double> _theta_m;	// Pitman-Yor過程の集中度

	// "A Bayesian Interpretation of Interpolated Kneser-Ney" Appendix C参照
	// http://www.gatsby.ucl.ac.uk/~ywteh/research/compling/hpylm.pdf
	vector<double> _a_m;		// ベータ分布のパラメータ	dの推定用
	vector<double> _b_m;		// ベータ分布のパラメータ	dの推定用
	vector<double> _alpha_m;	// ガンマ分布のパラメータ	θの推定用
	vector<double> _beta_m;		// ガンマ分布のパラメータ	θの推定用
	double _beta_stop;		// 停止確率q_iのベータ分布の初期パラメータ
	double _beta_pass;		// 停止確率q_iのベータ分布の初期パラメータ

	VPYLM(){
		_root = new Node(0);
		_root->_depth = 0;	// ルートは深さ0
		// http://www.ism.ac.jp/~daichi/paper/ipsj07vpylm.pdfによると初期値は(4, 1)
		// しかしVPYLMは初期値にあまり依存しないらしい
		_beta_stop = VPYLM_BETA_STOP;
		_beta_pass = VPYLM_BETA_PASS;
	}
	~VPYLM(){
		_delete_node(_root);
	}
	void _delete_node(Node* node){
		for(auto &elem: node->_children){
			Node* child = elem.second;
			_delete_node(child);
		}
		delete node;
	}
	bool add_customer_at_timestep(vector<id> &token_ids, int token_t_index, int depth_t){
		assert(0 <= depth_t && depth_t <= token_t_index);
		Node* node = find_node_by_tracing_back_context(token_ids, token_t_index, depth_t, true);
		assert(node != NULL);
		id token_t = token_ids[token_t_index];
		return node->add_customer(token_t, _g0, _d_m, _theta_m);
	}
	bool remove_customer_at_timestep(vector<id> &token_ids, int token_t_index, int depth_t){
		assert(0 <= depth_t && depth_t <= token_t_index);
		Node* node = find_node_by_tracing_back_context(token_ids, token_t_index, depth_t, true);
		assert(node != NULL);
		id token_t = token_ids[token_t_index];
		node->remove_customer(token_t);
		// 客が一人もいなくなったらノードを削除する
		if(node->need_to_remove_from_parent()){
			node->remove_from_parent();
		}
		return true;
	}
	int sample_depth_at_timestep(vector<id> &context_token_ids, int token_t_index){
		if(token_t_index == 0){
			return 0;
		}
		id token_t = context_token_ids[token_t_index];

		// この値を下回れば打ち切り
		double eps = 1e-6;
		
		vector<double> probs;
		double sum = 0;
		double p_pass = 0;
		double Pw = 0;
		Node* node = _root;
		for(int n = 0;n <= token_t_index;n++){
			if(node){
				Pw = node->compute_Pw(token_t, _g0, _d_m, _theta_m);
				double p_stop = node->stop_probability(_beta_stop, _beta_pass);
				p_pass = node->pass_probability(_beta_stop, _beta_pass);
				double p = Pw * p_stop;
				probs.push_back(p);
				sum += p;
				if(p < eps){
					break;
				}
				if(n < token_t_index){
					id context_token_id = context_token_ids[token_t_index - n - 1];
					node = node->find_child_node(context_token_id);
				}
			}else{
				double p_stop = p_pass * _beta_stop / (_beta_stop + _beta_pass);
				double p = Pw * p_stop;
				probs.push_back(p);
				sum += p;
				p_pass *= _beta_pass / (_beta_stop + _beta_pass);
				if(p < eps){
					break;
				}
			}
		}
		double normalizer = 1.0 / sum;
		uniform_real_distribution<double> rand(0, 1);
		double bernoulli = rand(Sampler::mt);
		double stack = 0;
		for(int n = 0;n < probs.size();n++){
			stack += probs[n] * normalizer;
			if(bernoulli < stack){
				return n;
			}
		}
		return probs.size() - 1;
	}
	double compute_Pw_given_h(vector<id> &word_ids, vector<id> context_token_ids){
		double p = 1;
		for(int n = 0;n < word_ids.size();n++){
			p *= compute_Pw_given_h(word_ids[n], context_token_ids);
			context_token_ids.push_back(word_ids[n]);
		}
		return p;
	}
	double compute_Pw_given_h(id token_id, vector<id> &context_token_ids){
		Node* node = _root;
		int depth = 0;
		double p = 0;
		for(;depth < context_token_ids.size();depth++){
			id context_token_id = context_token_ids[context_token_ids.size() - depth - 1];
			if(node == NULL){
				break;
			}
			Node* child = node->find_child_node(context_token_id);
			if(child == NULL){
				break;
			}
			double p_w = node->compute_Pw(token_id, _g0, _d_m, _theta_m);
			double p_stop = node->stop_probability(_beta_stop, _beta_pass);
			p += p_w * p_stop;
			node = child;
		}
		double p_w = node->compute_Pw(token_id, _g0, _d_m, _theta_m);
		double p_stop = node->stop_probability(_beta_stop, _beta_pass);
		p += p_w * p_stop;
		return p;
	}
	// old version
	double _compute_Pw_given_h(id token_id, vector<id> &context_token_ids){
		Node* node = _root;
		int depth = 0;
		for(;depth < context_token_ids.size();depth++){
			id context_token_id = context_token_ids[context_token_ids.size() - depth - 1];
			if(node == NULL){
				break;
			}
			Node* child = node->find_child_node(context_token_id);
			if(child == NULL){
				break;
			}
			node = child;
		}
		double p = 0;
		for(int n = 0;n <= depth;n++){
			double a = compute_Pw_given_hn(token_id, context_token_ids, n);
			double b = compute_Pn_given_h(n, context_token_ids);
			p += a * b;
		}
		return p;
	}
	double compute_Pw_given_hn(id token_id, vector<id> &context_token_ids, int n){
		assert(n <= context_token_ids.size());
		Node* node = _root;
		int depth = 0;
		for(;depth < n;depth++){
			id context_token_id = context_token_ids[context_token_ids.size() - depth - 1];
			if(node == NULL){
				break;
			}
			Node* child = node->find_child_node(context_token_id);
			if(child == NULL){
				break;
			}
			node = child;
		}
		assert(depth == n);
		double p = node->compute_Pw(token_id, _g0, _d_m, _theta_m);
		return p;
	}
	double compute_Pn_given_h(int n, vector<id> &context_token_ids){
		assert(n <= context_token_ids.size());
		Node* node = _root;
		int depth = 0;
		for(;depth < n;depth++){
			id context_token_id = context_token_ids[context_token_ids.size() - depth - 1];
			if(node == NULL){
				break;
			}
			Node* child = node->find_child_node(context_token_id);
			if(child == NULL){
				break;
			}
			node = child;
		}
		assert(depth == n);
		return node->stop_probability(_beta_stop, _beta_pass);
	}
	double compute_Pw(vector<id> &word_ids){
		if(word_ids.size() == 0){
			return 0;
		}
		id word = word_ids[0];
		double mult_pw = _root->compute_Pw(word, _g0, _d_m, _theta_m) * _root->stop_probability(_beta_stop, _beta_pass);
		vector<id> context_token_ids(word_ids.begin(), word_ids.begin() + 1);
		for(int depth = 1;depth < word_ids.size();depth++){
			id word = word_ids[depth];
			double pw_h = compute_Pw_given_h(word, context_token_ids);
			assert(pw_h > 0);
			mult_pw *= pw_h;
			context_token_ids.push_back(word_ids[depth]);
		}
		return mult_pw;
	}
	id sample_next_token(vector<id> &context_token_ids){
		int token_t_index = context_token_ids.size() - 1;
		Node* node = _root;
		vector<double> probs;
		vector<Node*> nodes;
		double pstop = _root->stop_probability(_beta_stop, _beta_pass);
		probs.push_back(pstop);
		nodes.push_back(node);
		double sum = 0;

		for(int n = 0;n <= token_t_index;n++){
			if(node){
				id context_token_id = context_token_ids[token_t_index - n];
				node = node->find_child_node(context_token_id);
				if(node == NULL){
					break;
				}
				double pstop = node->stop_probability(_beta_stop, _beta_pass);
				probs.push_back(pstop);
				nodes.push_back(node);
				sum += pstop;
			}
		}
		if(sum == 0){
			return ID_EOS;
		}
		double normalizer = 1.0 / sum;
		double bernoulli = Sampler::uniform(0, 1);
		double stack = 0;
		int depth = probs.size();
		for(int n = 0;n < probs.size();n++){
			stack += probs[n] * normalizer;
			if(stack >= bernoulli){
				depth = n;
			}
		}
		node = nodes[depth];

		vector<id> word_ids;
		probs.clear();
		sum = 0;
		for(auto elem: node->_arrangement){
			id token_id = elem.first;
			double pw_h = compute_Pw_given_h(token_id, context_token_ids);
			if(pw_h > 0){
				word_ids.push_back(token_id);
				probs.push_back(pw_h);
				sum += pw_h;
			}
		}
		if(word_ids.size() == 0){
			return ID_EOS;
		}
		if(sum == 0){
			return ID_EOS;
		}
		normalizer = 1.0 / sum;
		bernoulli = Sampler::uniform(0, 1);
		stack = 0;
		for(int i = 0;i < word_ids.size();i++){
			stack += probs[i] * normalizer;
			if(stack >= bernoulli){
				return word_ids[i];
			}
		}
		return word_ids.back();;
	}

	// token列の位置tからorderだけ遡る
	// token_ids:        [0, 1, 2, 3, 4, 5]
	// token_t_index:4          ^     ^
	// order_t: 2               |<- <-|
	Node* find_node_by_tracing_back_context(vector<id> &token_ids, int token_t_index, int order_t, bool generate_node_if_needed = false, bool return_middle_node = false){
		if(token_t_index - order_t < 0){
			return NULL;
		}
		Node* node = _root;
		for(int depth = 1;depth <= order_t;depth++){
			id context_token_id = token_ids[token_t_index - depth];
			Node* child = node->find_child_node(context_token_id, generate_node_if_needed);
			if(child == NULL){
				if(return_middle_node){
					return node;
				}
				return NULL;
			}
			node = child;
		}
		return node;
	}
	double compute_log_Pw(vector<id> &token_ids){
		assert(token_ids.size() > 0);
		double sum_pw_h = 0;
		vector<id> context_token_ids;
		for(int depth = 0;depth < token_ids.size();depth++){
			id token_id = token_ids[depth];
			double pw_h = compute_Pw_given_h(token_id, context_token_ids);
			assert(pw_h > 0);
			sum_pw_h += log(pw_h);
			context_token_ids.push_back(token_id);
		}
		return sum_pw_h;
	}
	double compute_log2_Pw(vector<id> &token_ids){
		assert(token_ids.size() > 0);
		double sum_pw_h = 0;
		vector<id> context_token_ids;
		for(int depth = 0;depth < token_ids.size();depth++){
			id token_id = token_ids[depth];
			double pw_h = compute_Pw_given_h(token_id, context_token_ids);
			assert(pw_h > 0);
			sum_pw_h += log2(pw_h);
			context_token_ids.push_back(token_id);
		}
		return sum_pw_h;
	}
	void init_hyperparameters_at_depth_if_needed(int depth){
		if(depth >= _d_m.size()){
			while(_d_m.size() <= depth){
				_d_m.push_back(HPYLM_INITIAL_D);
			}
		}
		if(depth >= _theta_m.size()){
			while(_theta_m.size() <= depth){
				_theta_m.push_back(HPYLM_INITIAL_THETA);
			}
		}
		if(depth >= _a_m.size()){
			while(_a_m.size() <= depth){
				_a_m.push_back(HPYLM_INITIAL_A);
			}
		}
		if(depth >= _b_m.size()){
			while(_b_m.size() <= depth){
				_b_m.push_back(HPYLM_INITIAL_B);
			}
		}
		if(depth >= _alpha_m.size()){
			while(_alpha_m.size() <= depth){
				_alpha_m.push_back(HPYLM_INITIAL_ALPHA);
			}
		}
		if(depth >= _beta_m.size()){
			while(_beta_m.size() <= depth){
				_beta_m.push_back(HPYLM_INITIAL_BETA);
			}
		}
	}
	// "A Bayesian Interpretation of Interpolated Kneser-Ney" Appendix C参照
	// http://www.gatsby.ucl.ac.uk/~ywteh/research/compling/hpylm.pdf
	void sum_auxiliary_variables_recursively(Node* node, vector<double> &sum_log_x_u_m, vector<double> &sum_y_ui_m, vector<double> &sum_1_y_ui_m, vector<double> &sum_1_z_uwkj_m, int &bottom){
		for(auto elem: node->_children){
			Node* child = elem.second;
			int depth = child->_depth;

			if(depth > bottom){
				bottom = depth;
			}
			init_hyperparameters_at_depth_if_needed(depth);

			double d = _d_m[depth];
			double theta = _theta_m[depth];
			sum_log_x_u_m[depth] += child->auxiliary_log_x_u(theta);	// log(x_u)
			sum_y_ui_m[depth] += child->auxiliary_y_ui(d, theta);		// y_ui
			sum_1_y_ui_m[depth] += child->auxiliary_1_y_ui(d, theta);	// 1 - y_ui
			sum_1_z_uwkj_m[depth] += child->auxiliary_1_z_uwkj(d);		// 1 - z_uwkj

			sum_auxiliary_variables_recursively(child, sum_log_x_u_m, sum_y_ui_m, sum_1_y_ui_m, sum_1_z_uwkj_m, bottom);
		}
	}
	// dとθの推定
	void sample_hyperparams(){
		int max_depth = _d_m.size() - 1;

		// 親ノードの深さが0であることに注意
		vector<double> sum_log_x_u_m(max_depth + 1, 0.0);
		vector<double> sum_y_ui_m(max_depth + 1, 0.0);
		vector<double> sum_1_y_ui_m(max_depth + 1, 0.0);
		vector<double> sum_1_z_uwkj_m(max_depth + 1, 0.0);

		// _root
		sum_log_x_u_m[0] = _root->auxiliary_log_x_u(_theta_m[0]);			// log(x_u)
		sum_y_ui_m[0] = _root->auxiliary_y_ui(_d_m[0], _theta_m[0]);		// y_ui
		sum_1_y_ui_m[0] = _root->auxiliary_1_y_ui(_d_m[0], _theta_m[0]);	// 1 - y_ui
		sum_1_z_uwkj_m[0] = _root->auxiliary_1_z_uwkj(_d_m[0]);				// 1 - z_uwkj

		// それ以外
		_depth = 0;
		// _max_depthは以下を実行すると更新される
		// HPYLMでは無意味だがVPYLMで最大深さを求める時に使う
		sum_auxiliary_variables_recursively(_root, sum_log_x_u_m, sum_y_ui_m, sum_1_y_ui_m, sum_1_z_uwkj_m, _depth);
		init_hyperparameters_at_depth_if_needed(_depth);

		for(int u = 0;u <= _depth;u++){
			_d_m[u] = Sampler::beta(_a_m[u] + sum_1_y_ui_m[u], _b_m[u] + sum_1_z_uwkj_m[u]);
			_theta_m[u] = Sampler::gamma(_alpha_m[u] + sum_y_ui_m[u], _beta_m[u] - sum_log_x_u_m[u]);
		}
		// 不要な深さのハイパーパラメータを削除
		int num_remove = _d_m.size() - _depth - 1;
		for(int n = 0;n < num_remove;n++){
			_d_m.pop_back();
			_theta_m.pop_back();
			_a_m.pop_back();
			_b_m.pop_back();
			_alpha_m.pop_back();
			_beta_m.pop_back();
		}
	}
	int get_num_nodes(){
		return _root->get_num_nodes();
	}
	int get_num_customers(){
		return _root->get_num_customers();
	}
	int get_num_tables(){
		return _root->get_num_tables();
	}
	int get_sum_stop_counts(){
		return _root->sum_stop_counts();
	}
	int get_sum_pass_counts(){
		return _root->sum_pass_counts();
	}
	void set_active_tokens(unordered_map<id, bool> &flags){
		_root->set_active_tokens(flags);
	}
	void count_tokens_of_each_depth(unordered_map<int, int> &map){
		_root->count_tokens_of_each_depth(map);
	}
	void enumerate_phrases_at_depth(int depth, vector<vector<id>> &phrases){
		assert(depth <= _depth);
		// 指定の深さのノードを探索
		vector<Node*> nodes;
		_root->enumerate_nodes_at_depth(depth, nodes);
		for(auto &node: nodes){
			vector<id> phrase;
			while(node->_parent){
				phrase.push_back(node->_token_id);
				node = node->_parent;
			}
			phrases.push_back(phrase);
		}
	}
	bool save(string filename = "hpylm.model"){
		std::ofstream ofs(filename);
		boost::archive::binary_oarchive oarchive(ofs);
		oarchive << static_cast<const VPYLM&>(*this);
		return true;
	}
	bool load(string filename = "hpylm.model"){
		std::ifstream ifs(filename);
		if(ifs.good() == false){
			return false;
		}
		boost::archive::binary_iarchive iarchive(ifs);
		iarchive >> *this;
		return true;
	}
};

#endif