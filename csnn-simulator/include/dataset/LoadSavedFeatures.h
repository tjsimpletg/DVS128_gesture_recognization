#ifndef _LOAD_SAVED_FEATURES_H
#define _LOAD_SAVED_FEATURES_H

#include <filesystem>
#include <cassert>
#include <limits>
#include <tuple>

#include "Input.h"
#include "tool/Operations.h"

namespace dataset
{

	/**
	 * @brief This class monitors introducing the dataset into the program, 
	 * It is responsible for loading and counting the number of samples.
	 * 
	 * @param folder_path the path to the saved featuremaps
	 */
	class LoadSavedFeatures : public Input
	{

	public:
		LoadSavedFeatures(const std::string &folder_path, size_t max_read = std::numeric_limits<size_t>::max());

		virtual bool has_next() const;
		virtual std::pair<std::string, Tensor<InputType>> next();
		virtual void reset();
		virtual void close();

		size_t size() const;
		virtual std::string to_string() const;

		virtual const Shape &shape() const;

	private:
		uint32_t swap(uint32_t v);

		std::string _folder_path;

		std::vector<std::pair<std::string, Tensor<float>>> _features;
		
		std::vector<std::string> _data_list;

		uint32_t _size;
		uint32_t _cursor;

		Shape _shape;

		uint32_t _max_read;
	};

}

#endif
