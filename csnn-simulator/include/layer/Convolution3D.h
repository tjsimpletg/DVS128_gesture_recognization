#ifndef _CONVOLTUION3D_H
#define _CONVOLTUION3D_H

#include "Stdp.h"
#include "Layer.h"
#include "Stdp.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include "tool/Operations.h"
#include "plot/Threshold.h"
#include "plot/Evolution.h"
#include <thread> // std::this_thread::sleep_for
#include <chrono>

namespace layer
{
	/**
	 * @brief Convolution is a type of filtering applied on an input image, extracts certain features depending on a number of filters.
	 * the features can be low level (curve, color, etc...) or more sophisticated features(eyes, nose, etc...)
	 */
	class Convolution3D;

	namespace _priv
	{

#ifdef SMID_AVX256
		class Convolution3DImpl
		{

		public:
			Convolution3DImpl(Convolution3D &model);

			void resize();
			void train(const std::vector<Spike> &input_spike, const Tensor<Time> &input_time, std::vector<Spike> &output_spike);
			void train(const std::string &label, const std::vector<Spike> &input_spike, const Tensor<Time> &input_time, std::vector<Spike> &output_spike);
			void test(const std::vector<Spike> &input_spike, const Tensor<Time> &, std::vector<Spike> &output_spike);

		private:
			Convolution3D &_model;
			std::string _label; // the label of the cuttent sample
			Tensor<float> _a;
			Tensor<float> _inh;
			Tensor<bool> _wta;
		};

#else

		class Convolution3DImpl
		{

		public:
			Convolution3DImpl(Convolution3D &model);

			void resize();
			void train(const std::vector<Spike> &input_spike, const Tensor<Time> &input_time, std::vector<Spike> &output_spike);
			void train(const std::string &label, const std::vector<Spike> &input_spike, const Tensor<Time> &input_time, std::vector<Spike> &output_spike);
			void test(const std::vector<Spike> &input_spike, const Tensor<Time> &, std::vector<Spike> &output_spike);

		private:
			Convolution3D &_model;
			std::string _label; // the label of the cuttent sample
			Tensor<float> _a;	// Activations. A tensor of the activations of all the neurons in the layer.
			Tensor<bool> _inh;	// Inhibitions. A tensor of the inhibition values of all the neurons in the layer.
			Tensor<bool> _wta;	// Winner takes all.
			uint32_t epoch_number;
		};
#endif
	} // namespace _priv

	/**
	 * @brief  Convolution is a type of filtering applied on an input image, extracts certain features depending on a number of filters.
	 * the features can be low level (curve, color, etc...) or more sophisticated features(eyes, nose, etc...)
	 *
	 * @param filter_width the width of the convolutional kernel
	 * @param filter_height the height of the convolutional kernel
	 * @param filter_depth the depth of the convolutional kernel in case of 3D convolution
	 * @param filter_number the number of the convolutional kernels, how many filters per layer
	 * This information is saved in the build file.
	 * @param model_path the path of a pre-trained model, in order not to train weights from scratch. If it's an empty string the weights are re-trained from random values.
	 * @param stride_x The step of the convolutional filter in the x diresction
	 * @param stride_y The step of the convolutional filter in the y diresction
	 * @param stride_k The step of the convolutional filter in the z diresction
	 * @param padding_x added padding to the filter in the x direction
	 * @param padding_y added padding to the filter in the y direction
	 * @param padding_k added padding to the filter in the z direction
	 */
	class Convolution3D : public Layer4D
	{

		friend class _priv::Convolution3DImpl;

	public:
		/**
		 * @brief Construct a new Convolution 3 D object
		 *
		 */
		Convolution3D();

		/**
		 * @brief Construct a new Convolution3D object, this one allows passing a trained mdoel weights by giving the exp name.
		 *
		 */
		Convolution3D(size_t filter_width, size_t filter_height, size_t filter_depth = 1, size_t filter_number = 6, std::string model_path = "", size_t stride_x = 1,
					  size_t stride_y = 1, size_t stride_k = 1, size_t padding_x = 0, size_t padding_y = 0, size_t padding_k = 0);

		Convolution3D(const Convolution3D &that) = delete;
		Convolution3D &operator=(const Convolution3D &that) = delete;

		virtual Shape compute_shape(const Shape &previous_shape);

		virtual size_t train_pass_number() const;
		virtual void process_train_sample(const std::string &label, Tensor<float> &sample, size_t current_pass, size_t current_index, size_t number);
		virtual void process_test_sample(const std::string &label, Tensor<float> &sample, size_t current_index, size_t number);

		virtual void train(const std::string &label, const std::vector<Spike> &input_spike, const Tensor<Time> &input_time, std::vector<Spike> &output_spike); //, size_t layer_index, size_t epoch_index );
		virtual void test(const std::string &label, const std::vector<Spike> &input_spike, const Tensor<Time> &input_time, std::vector<Spike> &output_spike);
		virtual void on_epoch_end();

		virtual Tensor<float> reconstruct(const Tensor<float> &t) const;
		virtual Tensor<float> construct_features(const Tensor<float> &t) const;

		void plot_threshold(bool only_in_train);
		void plot_evolution(bool only_in_train);

	private:
		uint32_t _epoch_number;
		uint32_t _current_epoch_number;
		uint32_t _last_epoch_number;
		uint32_t _sample_number;
		uint32_t _spike_count;

		// progress bar counter
		uint32_t _sample_count;
		uint32_t _drawn_weights;
		uint32_t _saved_weights;
		uint32_t _saved_random_start;
		uint32_t _logged_spiking_neuron;
		std::string _file_path;
		// this variable is used later in adapting other variables,(the learning rates (i.e. η w and η th ) are decreased by a factor α)
		float _annealing;
		// minimum threashould
		float _min_th;
		float _t_obj;
		float _lr_th;
		bool _draw;							// A flag to indicate if the user wants to print the weights as a JSOn file + images to see them
		bool _patch_coo_collection; // A flag that if true gets the patches from a set of coordinated and not randomly
		bool _log_spiking_neuron;			// A flag to log a json file of the activations (which neuron fired for which feature)
		bool _save_weights;
		bool _save_random_start;

		bool _inhibition;
		std::string _model_path;

		// synaptic weights of of the network
		Tensor<float> _w;
		// threashoulds of the neurons of the network.
		Tensor<float> _th;
		// spike time dependent plasticity - the learning rule used
		STDP *_stdp;
		// input_depth example RGB images have a depth of 3 while greyscale have a depth of 1.
		size_t _input_depth;
		// In case of 3D data, this indocates the time dimention.
		size_t _input_conv_depth;

		bool _wta_infer;

		_priv::Convolution3DImpl _impl;
	};

} // namespace layer
#endif
