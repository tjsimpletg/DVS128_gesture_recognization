#ifndef OPERATIONS_H
#define OPERATIONS_H
#include <fstream>
#include <string>
#include <vector>
#include <Tensor.h>
#include <chrono>
#include <thread>
#include "SparseTensor.h"

/**
 * @brief This function re-loads the descriptors that are previously saved using the SavePairVector.
 *
 * IMPORTANT NOTE: The language of the system can create an error with the parsing string to float functionstd::stof.
 * With the english system the delimiter is a coma ',' while in the french system the delimiter is a period '.'
 *
 * @param fileName The location of the .json file that contains the descriptors.
 * @param output The output descriptor vector that is used as an input to the SVM.
 */
void LoadPairVector(std::string fileName, std::vector<std::pair<std::string, Tensor<float>>> &output);

/**
 * @brief Loads the labels from a JSON file.
 *
 * @param in
 * @return std::vector<double>
 */
std::vector<double> LoadLabels(std::vector<std::pair<std::string, Tensor<float>>> &in);

/**
 * @brief a trial for using the load function found in the Tensor class.
 *
 */
void LoadPairVector2(std::string fileName, std::vector<std::pair<std::string, Tensor<float>>> output);

/**
 * @brief This function re-loads the descriptors that are previously saved using the SavePairVector.
 *
 * IMPORTANT NOTE: The language of the system can create an error with the parsing string to float functionstd::stof.
 * With the english system the delimiter is a coma ',' while in the french system the delimiter is a period '.'
 *
 * @param fileName The location of the .json file that contains the descriptors.
 * @param output The output descriptor vector that is used as an input to the SVM.
 */
void LoadWeights(std::string fileName, std::string label, Tensor<float> &output);

/**
 * @brief This function allows saving the weight matrecies, in order to reload them and continue the training later.
 *
 * @param fileName the save location of the JSON file that contains the weights.
 * @param label the label of the clss responsible for these weights
 * @param output the tensor of weights.
 */
void SaveWeights(std::string fileName, std::string label, Tensor<float> output);

/**
 * @brief loads labels from a JSON file.
 *
 * @param fileName
 * @return std::vector<std::string>
 */
std::vector<std::string> Load_json_labels(std::string fileName);

/**
 * @file SaveString.h
 * @author your name (you@domain.com)
 * @brief This output stream saves data information into a text file. (example usage: in the SVM class,
 * this string can be used to save the predictions into a seperate text file)
 * @version 0.1
 * @date 2021-04-05
 *
 * @copyright Copyright (c) 2021
 *
 */
void SaveString(std::string fileName, std::string stringToSave);

/**
 * @brief Saves pairs of (label and tensor) that represent the extracted features normalized between 0 & 1 that go directly into the SVM.
 *
 */
void SaveInputPairVector(std::string fileName, std::vector<std::pair<std::string, SparseTensor<float>>> output);

/**
 * @brief Saves pairs of (label and tensor) that represent the extracted features that go directly into the SVM.
 *
 * @param fileName
 * @param output
 */
void SavePairVector(std::string fileName, std::vector<std::pair<std::string, SparseTensor<float>>> output);

/**
 * @brief Saves a tensor of time.
 *
 * @param fileName
 * @param output
 */
void SaveTimeTensor(std::string fileName, Tensor<Time> output);

/**
 * @brief Saves a Feature.
 *
 * @param fileName
 * @param output
 */
void SaveFeature(std::string fileName, std::string label, Tensor<float> time_output, size_t sample_index, size_t total_sample_nbr);

/**
 * @brief Edits json file.
 *
 * @param fileName
 */
void JSONstringEdits(std::string fileName);

/**
 * @brief This function saves the log of which neuron fired at which sample.
 *
 * @param fileName the save location of the JSON file that contains the weights.
 * @param label the label of the clss responsible for these weights
 * @param neuron the neuron that fired.
 */
void LogSpikingNeuron(std::string fileName, std::string label, size_t neuron);

/**
 * @brief draws the features.
 *
 * @param fileName
 * @param output
 */
void DrawSparseFeatures(std::string draw_folder_path, std::vector<std::pair<std::string, SparseTensor<float>>> sparseTensorVector);

/**
 * @brief draws the features.
 *
 * @param fileName
 * @param output
 */
void DrawFeatures(std::string draw_folder_path, std::vector<std::pair<std::string, Tensor<float>>> TensorVector);

/**
 * @brief Concatination fusion.
 */
void FuseStreamsConcat1(std::vector<std::pair<std::string, Tensor<float>>> &_space, std::vector<std::pair<std::string, Tensor<float>>> &_time,
                        std::vector<std::pair<std::string, Tensor<float>>> &_fuse, std::string _draw_fused_path);

/**
 * @brief Concatination fusion.
 */
void FuseStreamsConcat2(std::vector<std::pair<std::string, Tensor<float>>> &_space, std::vector<std::pair<std::string, Tensor<float>>> &_time,
                        std::vector<std::pair<std::string, Tensor<float>>> &_fuse, std::string _draw_fused_path);
/**
 * @brief Max value fusion.
 */
void FuseStreamsConcat3(std::vector<std::pair<std::string, Tensor<float>>> &_space, std::vector<std::pair<std::string, Tensor<float>>> &_time,
                        std::vector<std::pair<std::string, Tensor<float>>> &_fuse, std::string _draw_fused_path);
/**
 * @brief Fuse by multiplication.
 */
void FuseStreamsConcat4(std::vector<std::pair<std::string, Tensor<float>>> &_space, std::vector<std::pair<std::string, Tensor<float>>> &_time,
                        std::vector<std::pair<std::string, Tensor<float>>> &_fuse, std::string _draw_fused_path);
/**
 * @brief Concatination fusion.
 */
void FuseStreamsConcat5(std::vector<std::pair<std::string, Tensor<float>>> &_space, std::vector<std::pair<std::string, Tensor<float>>> &_time,
                        std::vector<std::pair<std::string, Tensor<float>>> &_fuse, std::string _draw_fused_path);
/**
 * @brief Concatination fusion.
 */
void FuseStreamsConcat6(std::vector<std::pair<std::string, Tensor<float>>> &_space, std::vector<std::pair<std::string, Tensor<float>>> &_time,
                        std::vector<std::pair<std::string, Tensor<float>>> &_fuse, std::string _draw_fused_path);

void RegularToSparseStream(std::vector<std::pair<std::string, Tensor<float>>> &_regular_stream, std::vector<std::pair<std::string, SparseTensor<float>>> &_sparse_stream);

std::vector<std::pair<std::string, Tensor<float>>> temporal_pooling(std::vector<std::pair<std::string, Tensor<float>>> &_space, size_t _new_val);

/**
 * @brief Draws a progress bar in the output terminal.
 */
void draw_progress(int sample_count, int total_count);

void set_sample_count(int number, int stype);

int get_train_count();

int get_test_count();

void reset_train_count();

void reset_test_count();

void set_spike_coordinates(std::tuple<size_t, size_t, size_t, size_t> spike_coordinate);

std::vector<std::tuple<size_t, size_t, size_t, size_t>> get_spike_coordinates();

#endif
