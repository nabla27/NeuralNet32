#ifndef FILING_IOXML_H
#define FILING_IOXML_H


/* boost���C�u�������p�X�ɑ��݂��邩�̃}�N����` */
#ifndef HAS_BOOST_HEADER
#define HAS_BOOST_HEADER 0
#endif





#ifdef __has_include //c++17 or later





/* �eboost���C�u���������݂��邩�m�F */
#if __has_include (<boost/property_tree/xml_parser.hpp>)
#if __has_include (<boost/property_tree/ptree.hpp>)
#if __has_include (<boost/foreach.hpp>)
#if __has_include (<boost/lexical_cast.hpp>)

#undef HAS_BOOST_HEADER
#define HAS_BOOST_HEADER 1

#endif
#endif
#endif
#endif





#if HAS_BOOST_HEADER

#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>

#include <string>
#include <vector>
#include "nn/layerset.h"
#include "vec/function.h"





namespace io {




	class Xmlout {
	private:
		nn::LayerSet read_layer;
	public:
		void xml_reader(const std::string file_name);

		inline nn::LayerSet get_layerset() const { return read_layer; }
	};







	void Xmlout::xml_reader(const std::string file_name)
	{
		using namespace boost::property_tree;

		ptree pt;

		read_xml(file_name, pt);

		//�m�[�h��
		BOOST_FOREACH(const ptree::value_type & child, pt.get_child("root.Node")) {
			const int val = boost::lexical_cast<int>(child.second.data());
			read_layer.node.push_back(val);
		}

		//�d�݂ƃo�C�A�X�̃T�C�Y���m��
		std::vector<size_t> index;
		size_t data_size, label_size;
		if (boost::optional<size_t> val = pt.get_optional<size_t>("root.DataSize")) {
			data_size = val.get();
		}
		if (boost::optional<size_t> val = pt.get_optional<size_t>("root.LabelSize")) {
			label_size = val.get();
		}

		index.push_back(data_size);
		for (size_t i = 0; i < read_layer.node.size(); ++i) { index.push_back(read_layer.node[i]); }
		index.push_back(label_size);

		size_t layer_size = index.size() - 1;
		for (size_t i = 0; i < layer_size; ++i) {
			read_layer.weights.push_back(vec::vector2d(index[i], vec::vector1d(index[i + 1])));
			read_layer.bias.push_back(vec::vector1d(index[i + 1], 0));
		}

		//�d�݂̓ǂݎ��
		size_t i = 0, j = 0, k = 0;
		BOOST_FOREACH(const ptree::value_type& child, pt.get_child("root.Weights")) {
			const double val = boost::lexical_cast<double>(child.second.data());
			read_layer.weights[i][j][k] = val;
			k++;
			if (k == read_layer.weights[i][j].size()) { k = 0; j++; }
			if (j == read_layer.weights[i].size()) { j = 0; k = 0; i++; }
		}

		//�o�C�A�X�̓ǂݎ��
		i = 0; j = 0;
		BOOST_FOREACH(const ptree::value_type& child, pt.get_child("root.bias")) {
			const double val = boost::lexical_cast<double>(child.second.data());
			read_layer.bias[i][j] = val;
			j++;
			if (j == read_layer.bias[i].size()) { j = 0; i++; }
		}
	}












	void xml_writer(
		const nn::LayerSet& layerset,
		const std::string file_name
	)
	{
		using namespace boost::property_tree;

		ptree pt;

		//�P���f�[�^�̃T�C�Y
		pt.add("root.DataSize", layerset.weights[0].size());

		//���x���̃T�C�Y
		pt.add("root.LabelSize", layerset.weights[layerset.weights.size() - 1][0].size());

		//�m�[�h��
		ptree& child_node = pt.add("root.Node", "");
		size_t node_size = layerset.node.size();
		for (size_t i = 0; i < node_size; ++i) {
			child_node.add("value", layerset.node[i]);
		}

		//�d��
		ptree& child_weights = pt.add("root.Weights", "");
		size_t nn_size = layerset.weights.size();
		for (size_t i = 0; i < nn_size; ++i) {
			size_t row = layerset.weights[i].size();
			for (size_t j = 0; j < row; ++j) {
				size_t col = layerset.weights[i][j].size();
				for (size_t k = 0; k < col; ++k) {
					child_weights.add("w", layerset.weights[i][j][k]);
				}
			}
		}

		//�o�C�A�X
		ptree& child_bias = pt.add("root.bias", "");
		for (size_t i = 0; i < nn_size; ++i) {
			size_t size_b = layerset.bias[i].size();
			for (size_t j = 0; j < size_b; ++j) {
				child_bias.add("b", layerset.bias[i][j]);
			}
		}

		const int indent = 4; //�o�̓t�@�C���ł̃C���f���g��
		write_xml(file_name, pt, std::locale(), xml_writer_make_settings<std::string>(' ', indent, "utf-8"));
	}








}












#endif //HAS_BOOST_HEADER

#endif //__has_include

#endif //FILING_IOXML_H