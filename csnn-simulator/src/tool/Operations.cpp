#include "tool/Operations.h"

/**
 * @brief This function re-loads the descriptors that are previously saved using the SavePairVector.
 *
 * IMPORTANT NOTE: The language of the system can create an error with the parsing string to float functionstd::stof.
 * With the english system the delimiter is a coma ',' while in the french system the delimiter is a period '.'
 *
 * @param fileName The location of the .json file that contains the descriptors.
 * @param output The output descriptor vector that is used as an input to the SVM.
 */
// variables used to count the samples and draw the progress bar
static int _train_count;
static int _test_count;
// a variable used to collect the list of coordinates where a spike was fired.
static std::vector<std::tuple<size_t, size_t, size_t, size_t>> spike_coordinates;

void LoadPairVector(std::string fileName, std::vector<std::pair<std::string, Tensor<float>>> &output)
{
    // std::ifstream _jsonTextFile;
    // _jsonTextFile.open(fileName, std::ifstream::in);
    // std::string _jsonText;
    // std::getline(_jsonTextFile, _jsonText);
    // _jsonTextFile.close();

    std::ifstream _jsonTextFile(fileName);
    std::stringstream buffer;
    buffer << _jsonTextFile.rdbuf();
    std::string _jsonText;
    _jsonText = buffer.str();

    DynamicJsonDocument doc(JSON_ARRAY_SIZE(_jsonText.length()));
    DeserializationError error = deserializeJson(doc, _jsonText.c_str());
    // Test if parsing succeeds.
    if (error)
    {
        ASSERT_DEBUG("JSON PARSE FAILED");
    }
    for (std::size_t _s = 0; _s < doc.size(); ++_s)
    {
        std::vector<float> _fusedSampleVector;
        std::string _data = doc[_s]["data"];
        std::string _label = doc[_s]["label"];
        std::replace(_data.begin(), _data.end(), ',', ' ');
        std::remove(_data.begin(), _data.end(), '[');
        std::remove(_data.begin(), _data.end(), ']');

        std::stringstream iss(_data);

        float number;

        while (iss >> number)
            _fusedSampleVector.push_back(number);

        std::vector<size_t> temp_dims = {};
        for (size_t i = 0; i < 4; i++)
            temp_dims.push_back(doc[_s]["dim_" + std::to_string(i)]);

        Tensor<float> _fusedSample = Tensor<float>(Shape(temp_dims));

        for (size_t i = 0; i < _fusedSampleVector.size(); i++)
            _fusedSample.at_index(i) = _fusedSampleVector[i];

        output.push_back(std::make_pair(_label, _fusedSample));
    }
}

// void LoadPairVector(std::string fileName, std::vector<std::pair<std::string, Tensor<float>>> &output)
// {
//     std::ifstream _jsonTextFile;
//     _jsonTextFile.open(fileName, std::ifstream::in);
//     std::string _jsonText;
//     std::getline(_jsonTextFile, _jsonText);
//     _jsonTextFile.close();

//     DynamicJsonDocument doc(JSON_ARRAY_SIZE(_jsonText.length()));
//     DeserializationError error = deserializeJson(doc, _jsonText.c_str());
//     // Test if parsing succeeds.
//     if (error)
//     {
//         ASSERT_DEBUG("JSON PARSE FAILED");
//     }
//     for (std::size_t _s = 0; _s < doc.size(); ++_s)
//     {

//         std::vector<float> _fusedSampleVector;
//         std::string _data = doc[_s]["data"];
//         std::string _label = doc[_s]["label"];
//         std::replace(_data.begin(), _data.end(), '[', ' ');
//         std::replace(_data.begin(), _data.end(), ']', ' ');
//         std::string delimiter = ",";

//         size_t pos = 0;
//         std::string token;
//         while ((pos = _data.find(delimiter)) != std::string::npos)
//         {
//             token = _data.substr(0, pos);
//             _fusedSampleVector.push_back(std::stof(token, nullptr));
//             _data.erase(0, pos + delimiter.length());
//         }

//         std::vector<size_t> temp_dims = {};
//         for (size_t i = 0; i < 4; i++)
//             temp_dims.push_back(doc[_s]["dim_" + std::to_string(i)]);

//         Tensor<float> _fusedSample = Tensor<float>(Shape(temp_dims));

//         for (size_t i = 0; i < _fusedSampleVector.size(); i++)
//             _fusedSample.at_index(i) = _fusedSampleVector[i];

//         output.push_back(std::make_pair(_label, _fusedSample));
//     }
// }

/**
 * @brief Loads the labels from a JSON file.
 *
 * @param in
 * @return std::vector<double>
 */
std::vector<double> LoadLabels(std::vector<std::pair<std::string, Tensor<float>>> &in)
{
    std::vector<double> _out;
    for (std::size_t _s = 0; _s < in.size(); ++_s)
    {
        double _label = std::stod(in[_s].first);
        if (std::find(_out.begin(), _out.end(), _label) == _out.end())
            _out.push_back(_label);
    }
    return _out;
}

/**
 * @brief a trial for using the load function found in the Tensor class.
 *
 */
void LoadPairVector2(std::string fileName, std::vector<std::pair<std::string, Tensor<float>>> output)
{
    std::ifstream _jsonTextFile;
    _jsonTextFile.open(fileName, std::ifstream::in);

    for (std::size_t i = 0; i < output.size(); ++i)
    {
        output[i].second.load(_jsonTextFile);
    }
}

/**
 * @brief This function re-loads the descriptors that are previously saved using the SavePairVector.
 *
 * IMPORTANT NOTE: The language of the system can create an error with the parsing string to float functionstd::stof.
 * With the english system the delimiter is a coma ',' while in the french system the delimiter is a period '.'
 *
 * @param fileName The location of the .json file that contains the descriptors.
 * @param output The output descriptor vector that is used as an input to the SVM.
 */
void LoadWeights(std::string fileName, std::string label, Tensor<float> &in)
{
    std::ifstream _jsonTextFile;
    _jsonTextFile.open(fileName, std::ifstream::in);
    std::string _jsonText;
    std::getline(_jsonTextFile, _jsonText);
    _jsonTextFile.close();
    _jsonText.pop_back();
    _jsonText = "[" + _jsonText + "]";
    DynamicJsonDocument doc(JSON_ARRAY_SIZE(_jsonText.length()));
    DeserializationError error = deserializeJson(doc, _jsonText.c_str());
    // Test if parsing succeeds.
    if (error)
    {
        ASSERT_DEBUG("JSON PARSE FAILED");
    }
    for (std::size_t _s = 0; _s < doc.size(); ++_s)
    {
        std::string _label = doc[_s]["label"];
        if (label == _label)
        {
            std::vector<float> _fusedSampleVector;
            std::string _data = doc[_s]["data"];
            std::replace(_data.begin(), _data.end(), '[', ' ');
            std::replace(_data.begin(), _data.end(), ']', ' ');
            std::string delimiter = ",";

            size_t pos = 0;
            std::string token;
            while ((pos = _data.find(delimiter)) != std::string::npos)
            {
                token = _data.substr(0, pos);
                _fusedSampleVector.push_back(std::stof(token, nullptr));
                _data.erase(0, pos + delimiter.length());
            }

            for (size_t i = 0; i < _fusedSampleVector.size(); i++)
                in.at_index(i) = _fusedSampleVector[i];
        }
    }
}

/**
 * @brief This function allows saving the weight matrecies, in order to reload them and continue the training later.
 *
 * @param fileName the save location of the JSON file that contains the weights.
 * @param label the label of the clss responsible for these weights
 * @param output the tensor of weights.
 */
void SaveWeights(std::string fileName, std::string label, Tensor<float> output)
{
    std::ofstream _jsonTextFile;
    _jsonTextFile.open(fileName, std::ios_base::app);
    std::string JSON_output;
    std::string type_Name;
    size_t size = output.shape().product();

    DynamicJsonDocument doc(JSON_ARRAY_SIZE(3 + output.shape().number() + size));

    doc["label"] = label;

    uint8_t dim_number = output.shape().number();
    for (size_t _j = 0; _j < dim_number; _j++)
    {
        type_Name = "dim_" + std::to_string(_j);
        doc[type_Name] = output.shape().dim(_j);
    }

    for (size_t _j = 0; _j < size; _j++)
    {
        doc["data"][_j] = output.at_index(_j);
    }

    serializeJson(doc, JSON_output);

    _jsonTextFile << JSON_output;

    _jsonTextFile << ",";
    JSON_output = "";

    _jsonTextFile.close();
}

// /**
//  * @brief This function allows saving the weight matrecies, in order to reload them and continue the training later.
//  *
//  * @param fileName the save location of the JSON file that contains the weights.
//  * @param label the label of the clss responsible for these weights
//  * @param output the tensor of weights.
//  */
// void SaveWeights(std::string fileName, std::string label, Tensor<float> output)
// {
//     std::ofstream _jsonTextFile;
//     _jsonTextFile.open(fileName, std::ios_base::app);
//     std::string JSON_output;
//     std::string type_Name;

//     DynamicJsonDocument doc(output.shape().product() + output.shape().number() + 3);

//     doc["label"] = label;

//     size_t size = output.shape().product();

//     uint8_t dim_number = output.shape().number();
//     for (size_t _j = 0; _j < dim_number; _j++)
//     {
//         type_Name = "dim_" + std::to_string(_j);
//         doc[type_Name] = output.shape().dim(_j);
//     }

//     for (size_t _j = 0; _j < size; _j++)
//     {
//         doc["data"][_j] = output.at_index(_j);
//     }
//     serializeJson(doc, JSON_output);

//     _jsonTextFile << JSON_output;

//     _jsonTextFile << ",";
//     JSON_output = "";

//     _jsonTextFile.close();
// }

/**
 * @brief This function re-loads the descriptors that are previously saved using the SavePairVector.
 *
 * IMPORTANT NOTE: The language of the system can create an error with the parsing string to float functionstd::stof.
 * With the english system the delimiter is a coma ',' while in the french system the delimiter is a period '.'
 *
 * @param fileName The location of the .json file that contains the descriptors.
 * @param output The output descriptor vector that is used as an input to the SVM.
 */
std::vector<std::string> Load_json_labels(std::string fileName)
{
    std::vector<std::string> _labels;
    std::ifstream _jsonTextFile;
    _jsonTextFile.open(fileName, std::ifstream::in);
    std::string _jsonText;
    std::getline(_jsonTextFile, _jsonText);
    _jsonTextFile.close();

    DynamicJsonDocument doc(JSON_ARRAY_SIZE(_jsonText.length()));
    DeserializationError error = deserializeJson(doc, _jsonText.c_str());
    // Test if parsing succeeds.
    if (error)
    {
        ASSERT_DEBUG("JSON PARSE FAILED");
    }
    for (std::size_t _s = 0; _s < doc.size(); ++_s)
    {
        std::string _label = doc[_s]["label"];

        if (std::find(_labels.begin(), _labels.end(), _label) == _labels.end())
            _labels.push_back(_label);
    }
}

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
void SaveString(std::string fileName, std::string stringToSave)
{
    std::ofstream _textFile;
    _textFile.open(fileName, std::ios_base::app);
    _textFile << stringToSave << std::endl;
    _textFile.close();
}

/**
 * @brief Saves pairs of lavel and tensor that represent the training descriptors that go directly into the SVM.
 *
 * @param fileName
 * @param output
 */
void SavePairVector(std::string fileName, std::vector<std::pair<std::string, SparseTensor<float>>> sparseOutput)
{
    std::ofstream _jsonTextFile;
    _jsonTextFile.open(fileName, std::ios_base::app);
    std::string JSON_output;
    std::string type_Name;
    _jsonTextFile << "[";

    for (std::size_t i = 0; i < sparseOutput.size(); ++i)
    {
        Tensor<float> output = from_sparse_tensor(sparseOutput[i].second);
        DynamicJsonDocument doc(JSON_ARRAY_SIZE(sparseOutput[i].second.shape().product() + sparseOutput[i].second.shape().number() + 3));

        doc["label"] = sparseOutput[i].first;

        size_t size = sparseOutput[i].second.shape().product();

        uint8_t dim_number = sparseOutput[i].second.shape().number();
        for (size_t _j = 0; _j < dim_number; _j++)
        {
            type_Name = "dim_" + std::to_string(_j);
            doc[type_Name] = sparseOutput[i].second.shape().dim(_j);
        }

        for (size_t _j = 0; _j < size; _j++)
        {
            doc["data"][_j] = output.at_index(_j);
        }
        serializeJson(doc, JSON_output);

        _jsonTextFile << JSON_output;
        if (i != sparseOutput.size() - 1)
            _jsonTextFile << ",";
        JSON_output = "";
    }
    _jsonTextFile << "]";
    _jsonTextFile.close();
}

/**
 * @brief Saves pairs of lavel and tensor that represent the training descriptors that go directly into the SVM.
 *
 * @param fileName
 * @param output
 */
void SaveInputPairVector(std::string fileName, std::vector<std::pair<std::string, SparseTensor<float>>> sparseOutput)
{
    std::ofstream _jsonTextFile;
    _jsonTextFile.open(fileName, std::ios_base::app);
    std::string JSON_output;
    std::string type_Name;
    _jsonTextFile << "[";

    for (std::size_t i = 0; i < sparseOutput.size(); ++i)
    {
        Tensor<float> output = from_sparse_tensor(sparseOutput[i].second);
        Tensor<float>::normalize_tensor(output);
        DynamicJsonDocument doc(JSON_ARRAY_SIZE(sparseOutput[i].second.shape().product() + sparseOutput[i].second.shape().number() + 3));

        doc["label"] = sparseOutput[i].first;

        size_t size = sparseOutput[i].second.shape().product();

        uint8_t dim_number = sparseOutput[i].second.shape().number();
        for (size_t _j = 0; _j < dim_number; _j++)
        {
            type_Name = "dim_" + std::to_string(_j);
            doc[type_Name] = sparseOutput[i].second.shape().dim(_j);
        }

        for (size_t _j = 0; _j < size; _j++)
        {
            doc["data"][_j] = output.at_index(_j);
        }
        serializeJson(doc, JSON_output);

        _jsonTextFile << JSON_output;
        if (i != sparseOutput.size() - 1)
            _jsonTextFile << ",";
        JSON_output = "";
    }
    _jsonTextFile << "]";
    _jsonTextFile.close();
}

/**
 * @brief Saves a tensor of time.
 *
 * @param fileName
 * @param output
 */
void SaveTimeTensor(std::string fileName, Tensor<Time> time_output)
{
    std::ofstream _jsonTextFile;
    _jsonTextFile.open(fileName, std::ios_base::app);
    std::string JSON_output;
    std::string type_Name;

    DynamicJsonDocument doc(JSON_ARRAY_SIZE(time_output.shape().product() + time_output.shape().number() + 3));

    // doc["label"] = "";

    size_t size = time_output.shape().product();

    uint8_t dim_number = time_output.shape().number();
    for (size_t _j = 0; _j < dim_number; _j++)
    {
        type_Name = "dim_" + std::to_string(_j);
        doc[type_Name] = time_output.shape().dim(_j);
    }

    for (size_t _j = 0; _j < size; _j++)
    {
        doc["data"][_j] = time_output.at_index(_j);
    }
    serializeJson(doc, JSON_output);

    _jsonTextFile << JSON_output;
    _jsonTextFile << ",";

    _jsonTextFile.close();
}

/**
 * @brief Saves a Feature.
 *
 * @param fileName
 * @param output
 */
void SaveFeature(std::string fileName, std::string label, Tensor<float> time_output, size_t sample_index, size_t total_sample_nbr)
{
    std::ofstream _jsonTextFile;
    _jsonTextFile.open(fileName, std::ios_base::app);
    std::string JSON_output;
    std::string type_Name;

    if (sample_index == 1)
        _jsonTextFile << "[";

    DynamicJsonDocument doc(JSON_ARRAY_SIZE(time_output.shape().product() + time_output.shape().number() + 3));

    doc["label"] = label;

    size_t size = time_output.shape().product();

    uint8_t dim_number = time_output.shape().number();
    for (size_t _j = 0; _j < dim_number; _j++)
    {
        type_Name = "dim_" + std::to_string(_j);
        doc[type_Name] = time_output.shape().dim(_j);
    }

    for (size_t _j = 0; _j < size; _j++)
    {
        doc["data"][_j] = time_output.at_index(_j);
    }
    serializeJson(doc, JSON_output);

    _jsonTextFile << JSON_output;
    if (sample_index != total_sample_nbr)
        _jsonTextFile << ",";

    if (sample_index == total_sample_nbr)
        _jsonTextFile << "]";

    _jsonTextFile.close();
}

/**
 * @brief Edits json file.
 *
 * @param fileName
 */
void JSONstringEdits(std::string fileName)
{
    std::ifstream _jsonTextFile;
    _jsonTextFile.open(fileName, std::ios_base::in);
    std::string _jsonText;

    std::ofstream _jsonTextOutFile;
    _jsonTextOutFile.open(fileName, std::ios_base::app);

    _jsonTextFile >> _jsonText;
    _jsonText.insert(0, 1, '[');
    _jsonText.pop_back();
    _jsonText.insert(_jsonText.length(), 1, ']');

    _jsonTextOutFile << _jsonText;

    _jsonTextOutFile.close();
    _jsonTextFile.close();
}

/**
 * @brief This function saves the log of which neuron fired at which sample.
 *
 * @param fileName the save location of the JSON file that contains the weights.
 * @param label the label of the clss responsible for these weights
 * @param neuron the neuron that fired.
 */
void LogSpikingNeuron(std::string fileName, std::string label, size_t neuron)
{
    std::ofstream _File;
    _File.open(fileName + ".csv", std::ios_base::app);
    _File << std::to_string(neuron) + "," + label + "\n";
    //_File << "N: " << neuron << "  L: " << label << std::endl;
    _File.close();
}

void plotGraph(std::string directory, std::string fileName)
{
    std::vector<double> x = {1, 2, 3, 4};
    std::vector<double> y = {1, 4, 9, 16};

    // matplotlibcpp::plt::plot(x, y);
    // matplotlibcpp::plt::show();
}

void DrawSparseFeatures(std::string draw_folder_path, std::vector<std::pair<std::string, SparseTensor<float>>> sparseTensorVector)
{
    for (std::size_t i = 0; i < sparseTensorVector.size(); ++i)
    {
        Tensor<float> in = from_sparse_tensor(sparseTensorVector[i].second);

        Tensor<float>::draw_tensor(draw_folder_path + std::to_string(i) + "_label:" + sparseTensorVector[i].first, in);
    }
}

void DrawFeatures(std::string draw_folder_path, std::vector<std::pair<std::string, Tensor<float>>> TensorVector)
{
    for (std::size_t i = 0; i < TensorVector.size(); ++i)
    {
        Tensor<float> in = TensorVector[i].second;
        size_t _conv_depth = in.shape().dim(3);

        for (size_t conv = 0; conv < _conv_depth; conv++)
            Tensor<float>::draw_tensor(draw_folder_path + std::to_string(i) + "_" + std::to_string(conv) + "_label:" + TensorVector[i].first, in);
    }
}
//         Tensor<float> in = TensorVector[i].second;

//         size_t _height = in.shape().dim(0);
//         size_t _width = in.shape().dim(1);
//         size_t _depth = in.shape().dim(2);
//         size_t _conv_depth = in.shape().dim(3);

//         cv::Size _frame_size(500, 500);
//         // CONV_DEPTH by being incremented every frame.
//         for (size_t conv = 0; conv < _conv_depth; conv++)
//         {
//             cv::Mat frame(_height, _width, CV_32F);

//             for (int k = 0; k < _depth; k++)
//             {
//                 for (size_t _i = 0; _i < _height; _i++)
//                     for (size_t j = 0; j < _width; j++)
//                     {
//                         frame.at<float>(_i, j) = in.at(_i, j, k, conv); // * 255;
//                     }
//                 cv::resize(frame, frame, _frame_size);
//                 imwrite(draw_folder_path + std::to_string(i) + "_" + std::to_string(conv) + "_label:" + TensorVector[i].first + ".png", frame);
//             }
//         }
//     }
// }

/**
 * @brief Concatination in the width dimension.
 */
void FuseStreamsConcat1(std::vector<std::pair<std::string, Tensor<float>>> &_space,
                        std::vector<std::pair<std::string, Tensor<float>>> &_time,
                        std::vector<std::pair<std::string, Tensor<float>>> &_fuse, std::string _draw_fused_path)
{
    // size_t _height = _space[0].second.shape().dim(0);
    // size_t _width = _space[0].second.shape().dim(1);
    size_t _height = std::min(_space[0].second.shape().dim(0), _time[0].second.shape().dim(0));
    size_t _width = std::min(_space[0].second.shape().dim(1), _time[0].second.shape().dim(1));
    size_t _depth = std::min(_space[0].second.shape().dim(2), _time[0].second.shape().dim(2));
    size_t _conv_depth = _space[0].second.shape().dim(3);

    if (_space[0].second.shape().dim(3) > _time[0].second.shape().dim(3))
        temporal_pooling(_space, 1);

    _fuse.empty();
    size_t _size = std::min(_space.size(), _time.size());
    for (std::size_t _s = 0; _s < _size; ++_s)
    {
        std::pair<std::string, Tensor<float>> _fuse_sample(_space[_s].first, Shape(std::vector<size_t>({_height, _width * 2, _depth, _conv_depth})));

        // CONV_DEPTH by being incremented every frame.
        for (size_t conv = 0; conv < _conv_depth; conv++)
            for (size_t k = 0; k < _depth; k++)
                for (size_t i = 0; i < _height; i++)
                    for (size_t j = 0; j < _width; j++)
                    {
                        _fuse_sample.second.at(i, j, k, conv) = _time[_s].second.at(i, j, k, conv);
                        _fuse_sample.second.at(i, j + _width, k, conv) = _space[_s].second.at(i, j, k, conv);
                    }
        _fuse.push_back(_fuse_sample);
        if ((!_draw_fused_path.empty()) && _s < 100)
        {
            Tensor<float>::draw_tensor(_draw_fused_path + "/space/" + _fuse_sample.first + "_space_" + std::to_string(_s), _space[_s].second);
            Tensor<float>::draw_tensor(_draw_fused_path + "/time/" + _fuse_sample.first + "_time_" + std::to_string(_s), _time[_s].second);
            Tensor<float>::draw_tensor(_draw_fused_path + "/concat/" + _fuse_sample.first + "_concat_" + std::to_string(_s), _fuse_sample.second);
        }
    }
}

std::vector<std::pair<std::string, Tensor<float>>> temporal_pooling(std::vector<std::pair<std::string, Tensor<float>>> &_space, size_t _new_val)
{
    size_t _height = _space[0].second.shape().dim(0);
    size_t _width = _space[0].second.shape().dim(1);
    size_t _depth = _space[0].second.shape().dim(2);
    size_t _old_val = _space[0].second.shape().dim(3);

    std::vector<std::pair<std::string, Tensor<float>>> _space_out;
    Tensor<float> out(Shape({_width, _height, _depth, _new_val}));

    for (const std::pair<std::string, Tensor<float>> _sample : _space)
    {
        for (size_t x = 0; x < _width; x++)
            for (size_t y = 0; y < _height; y++)
                for (size_t z = 0; z < _depth; z++)
                    for (size_t k = 0; k < _new_val; k++)
                    {
                        float v = 0;
                        for (size_t fk = 0; fk < _old_val; fk++)
                        {
                            v += _sample.second.at(x, y, z, k * _old_val + fk);
                        }
                        out.at(x, y, z, k) = v;
                    }
        _space_out.push_back(std::make_pair(_sample.first, out));
    }

    return _space_out;
}

/**
 * @brief Concatination in the width dimension.
 */
void FuseStreamsConcat2(std::vector<std::pair<std::string, Tensor<float>>> &_space,
                        std::vector<std::pair<std::string, Tensor<float>>> &_time,
                        std::vector<std::pair<std::string, Tensor<float>>> &_fuse, std::string _draw_fused_path)
{
    size_t _height = _space[0].second.shape().dim(0);
    size_t _width = _space[0].second.shape().dim(1);
    size_t _depth = _space[0].second.shape().dim(2);
    size_t _conv_depth = _space[0].second.shape().dim(3);

    _fuse.empty();
    size_t _size = std::min(_space.size(), _time.size());
    for (std::size_t _i = 0; _i < _size; ++_i)
    {
        std::pair<std::string, Tensor<float>> _fuse_sample(_space[_i].first, Shape(std::vector<size_t>({_height, _width * 2, _depth, _conv_depth})));
        // CONV_DEPTH by being incremented every frame.
        for (size_t conv = 0; conv < _conv_depth; conv++)
            for (size_t k = 0; k < _depth; k++)
                for (size_t j = 0, j1 = 0; j < _width * 2; j += 2, j1++)
                    for (size_t i = 0; i < _height; i++)
                    {
                        _fuse_sample.second.at(i, j, k, conv) = _space[_i].second.at(i, j1, k, conv);
                        _fuse_sample.second.at(i, j + 1, k, conv) = _time[_i].second.at(i, j1, k, conv);
                    }
        _fuse.push_back(_fuse_sample);
        if ((!_draw_fused_path.empty()) && _i < 100)
        {
            Tensor<float>::draw_tensor(_draw_fused_path + "/space/" + _fuse_sample.first + "_space_" + std::to_string(_i), _space[_i].second);
            Tensor<float>::draw_tensor(_draw_fused_path + "/time/" + _fuse_sample.first + "_time_" + std::to_string(_i), _time[_i].second);
            Tensor<float>::draw_tensor(_draw_fused_path + "/concat/" + _fuse_sample.first + "_concat_" + std::to_string(_i), _fuse_sample.second);
        }
    }
}

/**
 * @brief Concatination in the width dimension.
 */
void FuseStreamsConcat5(std::vector<std::pair<std::string, Tensor<float>>> &_space,
                        std::vector<std::pair<std::string, Tensor<float>>> &_time,
                        std::vector<std::pair<std::string, Tensor<float>>> &_fuse, std::string _draw_fused_path)
{
    size_t _height = _space[0].second.shape().dim(0);
    size_t _width = _space[0].second.shape().dim(1);
    size_t _depth_sp = _space[0].second.shape().dim(2);
    size_t _depth_ti = _time[0].second.shape().dim(2);
    size_t _conv_depth_sp = _space[0].second.shape().dim(3);
    size_t _conv_depth_ti = _time[0].second.shape().dim(3);

    _fuse.empty();
    size_t _size = std::min(_space.size(), _time.size());
    for (std::size_t _i = 0; _i < _size; ++_i)
    {
        std::pair<std::string, Tensor<float>> _fuse_sample(_space[_i].first, Shape(std::vector<size_t>({_height, _width * 2, _depth_sp + _depth_ti, _conv_depth_sp + _conv_depth_ti})));
        // CONV_DEPTH by being incremented every frame.
        for (size_t conv_sp = 0; conv_sp < _conv_depth_sp; conv_sp++)
            for (size_t k_sp = 0; k_sp < _depth_sp; k_sp++)
                for (size_t conv = 0, conv_ti = conv_sp; conv_ti < conv_sp + _conv_depth_ti; conv++, conv_ti++)
                    for (size_t k = 0, k_ti = _depth_sp; k_ti < _depth_sp + _depth_ti; k++, k_ti++)
                        for (size_t j = 0, j1 = 0; j < _width * 2; j += 2, j1++)
                            for (size_t i = 0; i < _height; i++)
                            {
                                _fuse_sample.second.at(i, j, k_sp, conv_sp) = _space[_i].second.at(i, j1, k_sp, conv_sp);
                                _fuse_sample.second.at(i, j + 1, k, conv) = _time[_i].second.at(i, j1, k, conv);
                            }
        _fuse.push_back(_fuse_sample);
        if ((!_draw_fused_path.empty()) && _i < 100)
        {
            Tensor<float>::draw_tensor(_draw_fused_path + "/space/" + _fuse_sample.first + "_space_" + std::to_string(_i), _space[_i].second);
            Tensor<float>::draw_tensor(_draw_fused_path + "/time/" + _fuse_sample.first + "_time_" + std::to_string(_i), _time[_i].second);
            Tensor<float>::draw_tensor(_draw_fused_path + "/concat/" + _fuse_sample.first + "_concat_" + std::to_string(_i), _fuse_sample.second);
        }
    }
}

/**
 * @brief Concatination in the width dimension.
 */
void FuseStreamsConcat6(std::vector<std::pair<std::string, Tensor<float>>> &_space,
                        std::vector<std::pair<std::string, Tensor<float>>> &_time,
                        std::vector<std::pair<std::string, Tensor<float>>> &_fuse, std::string _draw_fused_path)
{
    size_t _height = std::min(_space[0].second.shape().dim(0), _time[0].second.shape().dim(0));
    size_t _width = std::min(_space[0].second.shape().dim(1), _time[0].second.shape().dim(1));
    size_t _depth = std::min(_space[0].second.shape().dim(2), _time[0].second.shape().dim(2));
    size_t _conv_depth_sp = _space[0].second.shape().dim(3);
    size_t _conv_depth_ti = _time[0].second.shape().dim(3);

    _fuse.empty();
    size_t _size = std::min(_space.size(), _time.size());
    for (std::size_t _i = 0; _i < _size; ++_i)
    {
        std::pair<std::string, Tensor<float>> _fuse_sample(_space[_i].first, Shape(std::vector<size_t>({_height, _width * 2, _depth, _conv_depth_sp + _conv_depth_ti})));
        // CONV_DEPTH by being incremented every frame.
        for (size_t conv_sp = 0; conv_sp < _conv_depth_sp; conv_sp++)
            for (size_t k = 0; k < _depth; k++)
                for (size_t conv = 0, conv_ti = conv_sp; conv_ti < conv_sp + _conv_depth_ti; conv++, conv_ti++)
                    for (size_t j = 0, j1 = 0; j < _width * 2; j += 2, j1++)
                        for (size_t i = 0; i < _height; i++)
                        {
                            _fuse_sample.second.at(i, j, k, conv_sp) = _space[_i].second.at(i, j1, k, conv_sp);
                            _fuse_sample.second.at(i, j + 1, k, conv) = _time[_i].second.at(i, j1, k, conv);
                        }
        _fuse.push_back(_fuse_sample);
        if ((!_draw_fused_path.empty()) && _i < 100)
        {
            Tensor<float>::draw_tensor(_draw_fused_path + "/space/" + _fuse_sample.first + "_space_" + std::to_string(_i), _space[_i].second);
            Tensor<float>::draw_tensor(_draw_fused_path + "/time/" + _fuse_sample.first + "_time_" + std::to_string(_i), _time[_i].second);
            Tensor<float>::draw_tensor(_draw_fused_path + "/concat/" + _fuse_sample.first + "_concat_" + std::to_string(_i), _fuse_sample.second);
        }
    }
}

/**
 * @brief Choosing the max, max fusion.
 */
void FuseStreamsConcat3(std::vector<std::pair<std::string, Tensor<float>>> &_space,
                        std::vector<std::pair<std::string, Tensor<float>>> &_time,
                        std::vector<std::pair<std::string, Tensor<float>>> &_fuse, std::string _draw_fused_path)
{
    size_t _height = _space[0].second.shape().dim(0);
    size_t _width = _space[0].second.shape().dim(1);
    size_t _depth = _space[0].second.shape().dim(2);
    size_t _conv_depth = _space[0].second.shape().dim(3);

    _fuse.empty();
    std::string _file_path = "";
    for (std::size_t _i = 0; _i < _space.size(); ++_i)
    {
        std::pair<std::string, Tensor<float>> _fuse_sample(_space[_i].first, Shape(std::vector<size_t>({_height, _width, _depth, _conv_depth})));
        // CONV_DEPTH by being incremented every frame.
        for (size_t conv = 0; conv < _conv_depth; conv++)
            for (int k = 0; k < _depth; k++)
                for (size_t i = 0; i < _height; i++)
                    for (size_t j = 0; j < _width; j++)
                    {
                        _fuse_sample.second.at(i, j, k, conv) = std::max(_space[_i].second.at(i, j, k, conv), _time[_i].second.at(i, j, k, conv));
                    }
        _fuse.push_back(_fuse_sample);
        if (!_draw_fused_path.empty())
            Tensor<float>::draw_resized_tensor(_draw_fused_path + _fuse_sample.first + "_" + std::to_string(_i), _fuse_sample.second);
    }
}

/***
 * Fuse by multiplication
 */
void FuseStreamsConcat4(std::vector<std::pair<std::string, Tensor<float>>> &_space,
                        std::vector<std::pair<std::string, Tensor<float>>> &_time,
                        std::vector<std::pair<std::string, Tensor<float>>> &_fuse, std::string _draw_fused_path)
{
    size_t _height = _space[0].second.shape().dim(0);
    size_t _width = _space[0].second.shape().dim(1);
    size_t _depth = _space[0].second.shape().dim(2);
    size_t _conv_depth = _space[0].second.shape().dim(3);

    _fuse.empty();
    std::string _file_path = "";
    for (std::size_t _i = 0; _i < _space.size(); ++_i)
    {
        std::pair<std::string, Tensor<float>> _fuse_sample(_space[_i].first, Shape(std::vector<size_t>({_height, _width, _depth, _conv_depth})));
        // CONV_DEPTH by being incremented every frame.
        for (size_t conv = 0; conv < _conv_depth; conv++)
            for (int k = 0; k < _depth; k++)
                for (size_t i = 0; i < _height; i++)
                    for (size_t j = 0; j < _width; j++)
                    {
                        _fuse_sample.second.at(i, j, k, conv) = _space[_i].second.at(i, j, k, conv) * _time[_i].second.at(i, j, k, conv);
                    }
        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
        _fuse.push_back(_fuse_sample);
        if (!_draw_fused_path.empty())
        {
            // Tensor<float>::draw_tensor(_draw_fused_path + "/space/" + _space[_i].first + "_" + std::to_string(_i), _space[_i].second);
            // Tensor<float>::draw_tensor(_draw_fused_path + "/time/" + _time[_i].first + "_" + std::to_string(_i), _time[_i].second);
            Tensor<float>::draw_resized_tensor(_draw_fused_path + "/fused/" + _fuse_sample.first + "_" + std::to_string(_i), _fuse_sample.second);
        }
    }
}

void RegularToSparseStream(std::vector<std::pair<std::string, Tensor<float>>> &_regular_stream, std::vector<std::pair<std::string, SparseTensor<float>>> &_sparse_stream)
{
    for (std::pair<std::string, Tensor<float>> &entry : _regular_stream)
    {
        _sparse_stream.emplace_back(entry.first, to_sparse_tensor(entry.second));
    }
}

void draw_progress(int sample_count, int total_count)
{
    // we erase the line with \r
    // we print [
    // we print _sample_count times |
    // we print _sample_number times " "
    // we print ]

    if (sample_count <= total_count)
    {
        size_t _progress = ((sample_count / (double)total_count) * 100);
        std::cout << "\r[";
        for (int _ = _progress / 2 - 1; _ >= 0; _--)
            std::cout << "|";
        for (int _ = 50 - _progress / 2 - 1; _ >= 0; _--)
            std::cout << " ";
        std::cout << "] _ " << _progress << "% (" << sample_count << "/" << total_count << ")";
        std::cout.flush();
    }
    if (sample_count == total_count)
        std::cout << std::endl;
}

void set_sample_count(int number, int stype)
{
    if (stype == 1)
        _train_count = number;
    else if (stype == 2)
        _test_count = number;
}

int get_train_count()
{
    return _train_count;
}

int get_test_count()
{
    return _test_count;
}

void reset_train_count()
{
    _train_count = 0;
}

void reset_test_count()
{
    _test_count = 0;
}

void set_spike_coordinates(std::tuple<size_t, size_t, size_t, size_t> spike_coordinate)
{
    if (find(spike_coordinates.begin(), spike_coordinates.end(), spike_coordinate) == spike_coordinates.end())
        spike_coordinates.push_back(spike_coordinate);
}

std::vector<std::tuple<size_t, size_t, size_t, size_t>> get_spike_coordinates()
{
    return spike_coordinates;
}