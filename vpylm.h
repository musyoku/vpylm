#ifndef _vpylm_
#define _vpylm_
#include <vector>
#include <random>
#include <cmath>
#include <unordered_map> 
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include "sampler.h"
#include "hpylm.h"
#include "node.h"

class VPYLM: public HPYLM{
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
	double _beta_stop;		// 停止確率q_iのベータ分布の初期パラメータ
	double _beta_pass;		// 停止確率q_iのベータ分布の初期パラメータ

	VPYLM(){
		_root = new Node(0);
		_root->_depth = 0;	// ルートは深さ0

		// http://www.ism.ac.jp/~daichi/paper/ipsj07vpylm.pdfによると初期値は(4, 1)
		// しかしVPYLMは初期値にあまり依存しないらしい
		_beta_stop = 4;
		_beta_pass = 1;
	}
	// n_tはw_tから見た深さ
	bool add(vector<id> &context, int w_t_i, int n_t){
		if(n_t > w_t_i){
			// cout << "VPYLM::add invalid order." << endl;
			return false;
		}

		id w_t = context[w_t_i];

		// cout << "VPYLM::add called with context: ";
  //       for(int i = 0;i < context.size();i++){
  //           cout << context[i] << ",";
  //       }
		// cout <<" w_t: " << w_t << " n_t: " << n_t << endl;

		Node* node = _root;
		for(int depth = 1;depth <= n_t;depth++){
			if(w_t_i - depth < 0){
				return false;
			}
			id u_t = context[w_t_i - depth];
			Node* child = node->generateChildIfNeeded(u_t);
			if(child == NULL){
				// cout << "internal error occurred." << endl;
				return false;
			}
			node = child;
		}
		double parent_p_w = _g0;
		if(node->_parent){
			parent_p_w = node->_parent->Pw(w_t, _g0, _d_m, _theta_m);
		}
		node->addCustomer(w_t, parent_p_w, _d_m, _theta_m);
		// cout << endl;
		return true;
	}

	bool remove(vector<id> &context, int w_t_i, int n_t){
		id w_t = context[w_t_i];

		// cout << "VPYLM::remove called with context: ";
  //       for(int i = 0;i < context.size();i++){
  //           cout << context[i] << ",";
  //       }
		// cout <<" w_t: " << w_t << " n_t: " << n_t << endl;


		Node* node = _root;
		for(int depth = 1;depth <= n_t;depth++){
			id u_t = context[w_t_i - depth];
			// cout << "u_t: " << u_t << endl;
			Node* child = node->findChildWithId(u_t);
			if(child == NULL){
				return false;
			}
			node = child;
		}

		bool should_remove_from_parent = false;
		node->removeCustomer(w_t, should_remove_from_parent);
		if(should_remove_from_parent && node->_parent != NULL){
			node->_parent->deleteChildWithId(node->_id);
		}
		return true;
	}

	int sampleOrder(vector<id> &context_ids, int w_t_i){
		// cout << "VPYLM::sampleOrder called with context_ids: ";
  //       for(int i = 0;i < context_ids.size();i++){
  //           cout << context_ids[i] << ",";
  //       }
		// cout << endl;
		if(w_t_i == 0){
			return 0;
		}
		// int max_n = w_t_i;
		// if(max_order != -1){
		// 	max_n = w_t_i > max_order ? max_order : w_t_i;
		// }
		id w_t = context_ids[w_t_i];
		// cout << "w_t: " << w_t << " w_t_i: " << w_t_i << endl;
		// cout << "_g0: " << _g0 << endl;
		vector<double> probs;
		double sum_p_stpp = 0;
		// この値を下回れば打ち切り
		double eps = 1e-6;
		
		double rand_max = 0;
		double p_pass = 0;
		double Pw = 0;
		Node* node = _root;
		for(int n = 0;n <= w_t_i;n++){
			// cout << "n = " << n << endl;
			// cout << "w_t: " << w_t << endl;
			if(node){
				// cout << "node: exists. depth: " << node->_depth << " word_id: " << node->_id << endl;
				Pw = node->Pw(w_t, _g0, _d_m, _theta_m);
				// cout << "Pw: " << Pw << endl;
				double p_stop = node->p_stop(_beta_stop, _beta_pass);
				// cout << "p_stop: " << p_stop << endl;
				p_pass = node->p_pass(_beta_stop, _beta_pass);
				double p = Pw * p_stop;
				// cout << "pw:  " << Pw << ", p_stop: " << p_stop << endl;
				probs.push_back(p);
				sum_p_stpp += p_stop;
				rand_max += probs[n];

				if(p < eps){
					break;
				}
				if(n < w_t_i){
					id u_t = context_ids[w_t_i - n - 1];
					// cout << "u_t: " << u_t << endl;
					node = node->findChildWithId(u_t);
				}
			}else{
				// cout << "node: none" << endl;
				// cout << "Pw: " << Pw << endl;
				double p_stop = p_pass * _beta_stop / (_beta_stop + _beta_pass);
				// cout << "p_stop: " << p_stop << endl;
				double p = Pw * p_stop;
				// cout << "p: " << p << endl;
				probs.push_back(p);
				sum_p_stpp += p_stop;
				rand_max += probs[n];
				p_pass *= _beta_pass / (_beta_stop + _beta_pass);
				// cout << "p_pass: " << p_pass << endl;

				if(p < eps){
					break;
				}
			}
		}

		// cout << "sum_p_stpp: " << sum_p_stpp << endl;
		// cout << "rand_max: " << rand_max << endl;
		// cout << "probs" << endl;
		// for(int n = 0;n <= max_n;n++){
		// 	cout << probs[n] << endl;
		// }
		double ratio = 1.0 / rand_max;
		uniform_real_distribution<double> rand(0, 1);
		double r = rand(Sampler::mt);
		// cout << "rand: " << r << endl;
		double sum = 0;
		for(int n = 0;n < probs.size();n++){
			sum += probs[n] * ratio;
			if(r < sum){
				// cout << "return " << n << endl;
				return n;
			}
		}
		// cout << "return " << probs.size() << endl;
		return probs.size() - 1;
	}

	// contextからwordが生成された確率
	// word = {うさぎですか}	context_ids = {ご注文は}
	// Pw_h(word, context_ids) = P(うさぎですか|ご注文は)
	double Pw_h(vector<id> &word_ids, vector<id> context_ids, bool fixed_depth = false){
		// cout << "p_w_h2 called with context_ids: ";
		// for(int i = 0;i < context_ids.size();i++){
			// cout << context_ids[i] << ",";
		// }
		// cout << " word_ids: ";
		// for(int i = 0;i < word_ids.size();i++){
			// cout << word_ids[i] << ",";
		// }
		// cout << endl;

		double p = 1;
		for(int n = 0;n < word_ids.size();n++){
			p *= Pw_h(word_ids[n], context_ids, fixed_depth);
			context_ids.push_back(word_ids[n]);
		}
		return p;
	}

	// P(w|h) = sum_n P(w|h,n)P(n|h)
	double Pw_h(id word_id, vector<id> &context_ids, bool fixed_depth = false){
		// cout << "Pw_h called with context_ids: ";
  //       for(int i = 0;i < context_ids.size();i++){
  //           cout << context_ids[i] << ",";
  //       }
  //       cout << " word_id: " << word_id << endl;

		// どの深さまでノードが存在するかを調べる
		Node* node = _root;
		int depth = 0;
		for(;depth < context_ids.size();depth++){
			id u_t = context_ids[context_ids.size() - depth - 1];
			// cout << "u_t: " << u_t << endl;
			if(node == NULL){
				break;
			}
			Node* child = node->findChildWithId(u_t);
			if(child == NULL){
				break;
			}
			node = child;
		}
		// cout << "depth: " << depth << endl;
		if(fixed_depth && depth != context_ids.size()){
			return 0;
		}
		double p = 0;
		for(int n = 0;n <= depth;n++){
			double a = Pw_hn(word_id, context_ids, n);
			double b = Pn_h(n, context_ids);
			// cout << "p += " << a << " * " << b << endl;
			p += a * b;
		}
		// cout << "depth available" << endl;
		return p;
	}

	// P(w|h,n)
	// 可能な深さ全てのノードが存在している必要がある
	double Pw_hn(id word_id, vector<id> &context_ids, int n){
		// cout << "Pw_hn called with context_ids: ";
  //       for(int i = 0;i < context_ids.size();i++){
  //           cout << context_ids[i] << ",";
  //       }
  //       cout << " word_id: " << word_id << " n: " << n << endl;

		if(n > context_ids.size()){
			// ノードが無い場合は0を返す
			printf("\x1b[41;97m");
			printf("WARNING");
			printf("\x1b[49;39m");
			printf(" n > context_ids.size() at VPYLM::Pw_hn\n");
			return 0;
		}

		// どの深さまでノードが存在するかを調べる
		Node* node = _root;
		int depth = 0;
		for(;depth < n;depth++){
			id u_t = context_ids[context_ids.size() - depth - 1];
			// cout << "u_t: " << u_t << endl;
			if(node == NULL){
				break;
			}
			Node* child = node->findChildWithId(u_t);
			if(child == NULL){
				break;
			}
			node = child;
		}
		// cout << "depth: " << depth << endl;

		if(depth != n){
			// ノードが無い場合は0を返す
			printf("\x1b[41;97m");
			printf("WARNING");
			printf("\x1b[49;39m");
			printf(" depth != n at VPYLM::Pw_hn\n");
			return 0;
		}

		double p = node->Pw(word_id, _g0, _d_m, _theta_m);
		// cout << "return: " << p << endl;
		return p;
	}

	// P(n|h)
	// 可能な深さ全てのノードが存在している必要がある
	double Pn_h(int n, vector<id> &context_ids){
		// cout << "Pn_h called with context_ids: ";
		// for(int i = 0;i < context_ids.size();i++){
		//     cout << context_ids[i] << ",";
		// }
		// cout << " n: " << n << endl;

		if(n > context_ids.size()){
			// ノードが無い場合は0を返す
			printf("\x1b[41;97m");
			printf("WARNING");
			printf("\x1b[49;39m");
			printf(" n > context_ids.size() at VPYLM::Pw_h\n");
			return 0;
		}

		Node* node = _root;
		int depth = 0;
		for(;depth < n;depth++){
			id u_t = context_ids[context_ids.size() - depth - 1];
			// cout << "u_t: " << u_t << endl;
			if(node == NULL){
				break;
			}
			Node* child = node->findChildWithId(u_t);
			if(child == NULL){
				break;
			}
			node = child;
		}
		// cout << depth << endl;
		if(depth != n){
			// ノードが無い場合は0を返す
			printf("\x1b[41;97m");
			printf("WARNING");
			printf("\x1b[49;39m");
			printf(" depth != n at VPYLM::Pn_h\n");
			return 0;
		}
		// cout << *node << endl;

		// 親ノードまで遡って計算される
		return node->p_stop(_beta_stop, _beta_pass);
	}

	// P(word)
	// s = abcならP(abc) = P(c|ab)P(b|a)P(a)
	double Pw(vector<id> &word_ids){
		if(word_ids.size() == 0){
			return 0;
		}
		// cout << "VPYLM::Pw called with word_ids: ";
  //       for(int i = 0;i < word_ids.size();i++){
  //           cout << word_ids[i] << ",";
  //       }
  //       cout << endl;
		id w_0 = word_ids[0];
		// cout << "w_0: " << w_0 << endl;
		double p0 = _root->Pw(w_0, _g0, _d_m, _theta_m) * _root->p_stop(_beta_stop, _beta_pass);
		// cout << "p0: " << p0 << endl;
		double p = p0;
		vector<id> h(word_ids.begin(), word_ids.begin() + 1);
		for(int depth = 1;depth < word_ids.size();depth++){
			vector<id> w = {word_ids[depth]};
			// cout << "w: " << word_ids[depth] << endl;
			// cout << "h: ";
			// for(int j = 0;j < h.size();j++){
				// cout << h[j] << ",";
			// }
			// cout << endl;
			double _p = Pw_h(w, h);
			// cout << "p *= " << _p << endl;
			p *= _p;
			h.push_back(word_ids[depth]);
		}
		// cout << "return: " << p << endl;
		return p;
	}

	double log_Pw(vector<id> &word_ids){
		if(word_ids.size() == 0){
			return 0;
		}
		// cout << "VPYLM::Pw called with word_ids: ";
  //       for(int i = 0;i < word_ids.size();i++){
  //           cout << word_ids[i] << ",";
  //       }
  //       cout << endl;
		id w_0 = word_ids[0];
		// cout << "w_0: " << w_0 << endl;
		double p0 = _root->Pw(w_0, _g0, _d_m, _theta_m) * _root->p_stop(_beta_stop, _beta_pass);
		// cout << "p0: " << p0 << endl;
		double p = log2(p0 + 1e-10);
		vector<id> context_ids(word_ids.begin(), word_ids.begin() + 1);
		for(int depth = 1;depth < word_ids.size();depth++){
			id word = word_ids[depth];
			// cout << "word: " << word_ids[depth] << endl;
			// cout << "context_ids: ";
			// for(int j = 0;j < context_ids.size();j++){
				// cout << context_ids[j] << ",";
			// }
			// cout << endl;
			double _p = Pw_h(word, context_ids);
			// cout << "p *= " << _p << endl;
			p += log2(_p + 1e-10);
			context_ids.push_back(word_ids[depth]);
		}
		// cout << "return: " << p << endl;
		return p;
	}

	id sampleNextWord(vector<id> &context_ids, id eos_id){
		// どの深さまでノードが存在するかを調べる
		// int max_n = w_t_i;
		// if(max_order != -1){
		// 	max_n = w_t_i > max_order ? max_order : w_t_i;
		// }
		// for(int i = 0;i < context_ids.size();i++){
		// 	cout << context_ids[i] << ",";
		// }
		// cout << endl;
		int w_t_i = context_ids.size() - 1;
		// cout << "_g0: " << _g0 << endl;
		Node* node = _root;
		vector<double> probs;
		vector<Node*> nodes;
		double p = _root->p_stop(_beta_stop, _beta_pass);
		probs.push_back(p);
		nodes.push_back(node);
		double sum = 0;

		for(int n = 0;n <= w_t_i;n++){
			if(node){
				id u_t = context_ids[w_t_i - n];
				node = node->findChildWithId(u_t);
				if(node == NULL){
					break;
				}
				double p = node->p_stop(_beta_stop, _beta_pass);
				probs.push_back(p);
				nodes.push_back(node);
				sum += p;
			}
		}
		if(sum == 0){
			return eos_id;
		}
		double ratio = 1.0 / sum;
		uniform_real_distribution<double> rand(0, 1);
		double r = rand(Sampler::mt);
		// cout << "rand: " << r << endl;
		sum = 0;
		int depth = probs.size();
		for(int n = 0;n < probs.size();n++){
			sum += probs[n] * ratio;
			if(r < sum){
				// cout << "return " << n << endl;
				depth = n;
			}
		}
		// cout << "depth: " << depth << endl;
		node = nodes[depth];
		// cout << *node << endl;

		vector<id> word_ids;
		probs.clear();
		sum = 0;
		for(auto elem: node->_arrangement){
			id word_id = elem.first;
			double p = Pw_h(word_id, context_ids);
			if(p > 0){
				word_ids.push_back(word_id);
				probs.push_back(p);
				sum += p;
			}
		}
		if(word_ids.size() == 0){
			return eos_id;
		}
		if(sum == 0){
			return eos_id;
		}
		ratio = 1.0 / sum;
		r = Sampler::uniform(0, 1);
		sum = 0;
		id sampled_word_id = word_ids.back();
		for(int i = 0;i < word_ids.size();i++){
			sum += probs[i] * ratio;
			if(sum > r){
				sampled_word_id = word_ids[i];
				break;
			}
		}
		return sampled_word_id;
	}

	int maxDepth(){
		return _d_m.size() - 1;
	}

	int numChildNodes(){
		return _root->numChildNodes();
	}

	int numCustomers(){
		return _root->numCustomers();
	}

	void save(string dir = "model/"){
		string filename = dir + "vpylm.model";
		std::ofstream ofs(filename);
		boost::archive::binary_oarchive oarchive(ofs);
		oarchive << static_cast<const VPYLM&>(*this);
		cout << "saved to " << filename << endl;
		cout << "	num_customers: " << numCustomers() << endl;
		cout << "	num_nodes: " << numChildNodes() << endl;
		cout << "	max_depth: " << maxDepth() << endl;
	}

	void load(string dir = "model/"){
		string filename = dir + "vpylm.model";
		std::ifstream ifs(filename);
		if(ifs.good()){
			cout << "loading " << filename << endl;
			boost::archive::binary_iarchive iarchive(ifs);
			iarchive >> *this;
			cout << "	num_customers: " << numCustomers() << endl;
			cout << "	num_nodes: " << numChildNodes() << endl;
			cout << "	max_depth: " << maxDepth() << endl;
		}
	}
};

#endif