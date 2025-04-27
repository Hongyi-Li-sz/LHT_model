
#include <iostream>
#include <fstream>
#include <chrono>
#include <assert.h>
#include "LHT.h"
#include <vector>
#include <sstream>
#include <cassert>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <cmath>
#include <map>
#include <iomanip>






std::vector<std::vector<double>> train_features;
std::vector<uint8_t> train_labels;
std::vector<std::vector<double>> test_features;
std::vector<uint8_t> test_labels;
std::vector<double> experiment_accuracies;
size_t treeNo;
extern long leafCount;
extern bool treeFinal;
class cLF_tree;
long nSensors;
int numberLF_trees;
int minSamples2Split;
int constancyLimit;
double feature_train_size;
double feature_test_size;
double beta;
double forest_rate;
size_t feature_dim = 0;
size_t class_num;


void loadData(int randomSeed = static_cast<int>(time(0)));
double reinforcement(uint8_t action, uint8_t label);
double testOnTEST_SET(int numberLF_trees, cLF_tree** apLF_tree);
double runSingleExperiment(int numberLF_trees, int randomSeed, char* ProjectName);
void calculateConfidenceInterval(const std::vector<double>& accuracies);


void loadData(int randomSeed) {

    // Define paths to training and testing data files
    std::string train_path = "..\\Data\\satimage.scale.txt";
    std::string test_path = "..\\Data\\satimage_test.txt";

    // Create a label mapping to map original labels to new labels starting from 1
    std::unordered_map<uint8_t, uint8_t> label_mapping;
    uint8_t next_label = 1;

    // Track the maximum feature index
    size_t max_feature_index = 0;

    // Lambda function to load data from a file
    auto loadFromFile = [&](const std::string& file_path, std::vector<std::vector<double>>& features,
        std::vector<uint8_t>& labels) {
            // Open the file
            std::ifstream file(file_path);
            if (!file.is_open()) {
                std::cerr << "Error opening file: " << file_path << std::endl;
                return false;
            }

            // Read the file line by line
            std::string line;
            while (std::getline(file, line)) {
                std::istringstream ss(line);
                std::string token;
                uint8_t original_label;

                // Read the label (first column)
                ss >> token;
                original_label = static_cast<uint8_t>(std::stoi(token));

                // Map the original label to a new label if it's not already mapped
                if (label_mapping.find(original_label) == label_mapping.end()) {
                    label_mapping[original_label] = next_label++;
                }
                uint8_t label = label_mapping[original_label];

                // Create a feature vector (initialized as empty)
                std::map<size_t, double> feature_map;

                // Read features (format: index:value or index: value)
                while (ss >> token) {
                    size_t colon_pos = token.find(':');
                    if (colon_pos != std::string::npos) {
                        size_t index = std::stoul(token.substr(0, colon_pos)) - 1;

                        std::string value_str = token.substr(colon_pos + 1);
                        // Handle cases where there's a space before the value
                        value_str.erase(0, value_str.find_first_not_of(" \t"));

                        // Handle cases where the value is empty (e.g., "index: ")
                        double value;
                        if (value_str.empty() || value_str == " ") {
                            std::string next_value;
                            if (ss >> next_value) {
                                value = std::stod(next_value);
                            }
                            else {
                                std::cerr << "Error parsing feature value for index " << index + 1 << std::endl;
                                continue;
                            }
                        }
                        else {
                            // Handle cases where the value is in the format ".50" (no leading 0)
                            if (value_str[0] == '.') {
                                value_str = "0" + value_str;
                            }
                            value = std::stod(value_str);
                        }

                        feature_map[index] = value;

                        // Update the maximum feature index
                        max_feature_index = std::max(max_feature_index, index);
                    }
                }

                // Convert the feature map to a vector, filling in missing values with -1
                std::vector<double> feature_vector(max_feature_index + 1, -1.0);
                for (const auto& pair : feature_map) {
                    feature_vector[pair.first] = pair.second;
                }

                // Add the feature vector and label to the dataset
                features.push_back(feature_vector);
                labels.push_back(label);
            }

            // Close the file
            file.close();
            return true;
        };

    // Load the training data first to establish the label mapping
    if (!loadFromFile(train_path, train_features, train_labels)) {
        return;
    }

    // Load the testing data using the same label mapping
    if (!loadFromFile(test_path, test_features, test_labels)) {
        return;
    }

    // Update the feature dimension for all samples
    feature_dim = max_feature_index + 1;
    for (auto& feature_vector : train_features) {
        if (feature_vector.size() < feature_dim) {
            feature_vector.resize(feature_dim, -1.0);
        }
    }

    for (auto& feature_vector : test_features) {
        if (feature_vector.size() < feature_dim) {
            feature_vector.resize(feature_dim, -1.0);
        }
    }

    // Check if any features were found
    if (feature_dim == 0) {
        std::cerr << "Error: No features found in the dataset" << std::endl;
        return;
    }

    // Calculate the mean and standard deviation of each feature (based on the training data only)
    std::vector<double> feature_means(feature_dim, 0.0);
    std::vector<double> feature_stds(feature_dim, 0.0);
    std::vector<size_t> feature_counts(feature_dim, 0); // Count of non-missing values for each feature

    // Calculate the mean (ignoring missing values)
    for (const auto& features : train_features) {
        for (size_t i = 0; i < features.size(); ++i) {
            if (features[i] != -1.0) {
                feature_means[i] += features[i];
                feature_counts[i]++;
            }
        }
    }

    for (size_t i = 0; i < feature_means.size(); ++i) {
        if (feature_counts[i] > 0) {
            feature_means[i] /= feature_counts[i];
        }
    }

    // Calculate the standard deviation (ignoring missing values)
    for (const auto& features : train_features) {
        for (size_t i = 0; i < features.size(); ++i) {
            if (features[i] != -1.0) {
                feature_stds[i] += std::pow(features[i] - feature_means[i], 2);
            }
        }
    }

    for (size_t i = 0; i < feature_stds.size(); ++i) {
        if (feature_counts[i] > 0) {
            feature_stds[i] = std::sqrt(feature_stds[i] / feature_counts[i]);
        }
    }

    // Standardize the features for both the training and testing data (ignoring missing values)
    for (auto& feature_set : train_features) {
        for (size_t i = 0; i < feature_set.size(); ++i) {
            if (feature_set[i] != -1.0) {
                if (feature_stds[i] != 0) {
                    feature_set[i] = (feature_set[i] - feature_means[i]) / feature_stds[i];
                }
            }
        }
    }

    for (auto& feature_set : test_features) {
        for (size_t i = 0; i < feature_set.size(); ++i) {
            if (feature_set[i] != -1.0) {
                if (feature_stds[i] != 0) {
                    feature_set[i] = (feature_set[i] - feature_means[i]) / feature_stds[i];
                }
            }
        }
    }

    // Save the original testing data
    std::vector<std::vector<double>> original_test_features = test_features;
    std::vector<uint8_t> original_test_labels = test_labels;
    size_t original_test_size = test_features.size();

    // Initialize a random number generator
    std::mt19937 rng(randomSeed);
    std::uniform_int_distribution<size_t> dist(0, original_test_size - 1);

    // Clear the current testing data and resample
    test_features.clear();
    test_labels.clear();

    // Resample the same number of instances as the original testing data
    for (size_t i = 0; i < original_test_size; ++i) {
        size_t idx = dist(rng);
        test_features.push_back(original_test_features[idx]);
        test_labels.push_back(original_test_labels[idx]);
    }

    // Update output information
    feature_train_size = train_features.size();
    feature_test_size = test_features.size();
    class_num = label_mapping.size();
}


int main(int argc, char* argv[])
{
    std::cout << "Learning Hyperplane Trees\n\nInput parameters:\n" << std::endl;

    int argCount = argc;
    if (argc != 6)
    {
        std::cout << " Wrong number of input values! " << std::endl;
        return 1;
    }

    loadData();
    nSensors = feature_dim;
    std::cout << "\nProject name         " << argv[1] << "\nGamma, min_samples   " << argv[2]
        << "\nBeta                 " << argv[3] << "\nTree Num./ Class     " <<
        argv[4] << "\nForest Rate          " << argv[5] << std::endl << std::endl;

    char* projectName = argv[1];
    minSamples2Split = atoi(argv[2]);
    beta = atof(argv[3]);
    numberLF_trees = atoi(argv[4]) * class_num;
    forest_rate = atof(argv[5]);



    // Perform repeated experiments
    const int numExperiments = 10;
    experiment_accuracies.clear();

    std::cout << "\nStarting " << numExperiments << " repeated experiments...\n";

    for (int exp = 0; exp < numExperiments; exp++) {
        std::cout << "\n======== Experiment " << (exp + 1) << "/" << numExperiments << " ========\n";
        // Use different random seeds
        int randomSeed = static_cast<int>(time(0));
        double accuracy = runSingleExperiment(numberLF_trees, randomSeed, projectName);
        experiment_accuracies.push_back(accuracy);
        std::cout << "Experiment " << (exp + 1) << " completed, accuracy: " << accuracy * 100.0 << "%\n";
    }

    // Calculate and display confidence interval
    calculateConfidenceInterval(experiment_accuracies);
    std::cout << "Class number  " << class_num << " Feature dimension " << feature_dim << " Sample number " << feature_train_size + feature_test_size << "\n";
    std::cout << " Sample number " << feature_train_size + feature_test_size << " Training sample number " << feature_train_size << " Testing sample number " << feature_test_size << "\n";

    return 0;
}  // End of main routine and the program run

// Helper function to calculate confidence interval
void calculateConfidenceInterval(const std::vector<double>& accuracies) {
    double sum = 0.0;
    for (double acc : accuracies) {
        sum += acc;
    }
    double mean = sum / accuracies.size();

    double variance = 0.0;
    for (double acc : accuracies) {
        variance += (acc - mean) * (acc - mean);
    }
    variance /= accuracies.size();

    // Calculate standard error
    double std_error = std::sqrt(variance / accuracies.size());

    // Use 95% confidence interval (t=2.262 for 9 degrees of freedom)
    double t_value = 2.262;
    double margin_of_error = t_value * std_error;

    std::cout << "\n---------- Experiment result statistics ----------\n";
    std::cout << "Performed " << accuracies.size() << " repeated experiments\n";
    std::cout << "Average accuracy: " << mean * 100.0 << "%\n";
    std::cout << "Standard deviation: " << std::sqrt(variance) * 100.0 << "%\n";
    std::cout << "95% Confidence interval: [" << (mean - margin_of_error) * 100.0
        << "%, " << (mean + margin_of_error) * 100.0 << "%]\n";
    std::cout << "Accuracy of each experiment: ";
    for (size_t i = 0; i < accuracies.size(); ++i) {
        std::cout << accuracies[i] * 100.0 << "%";
        if (i < accuracies.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
}

// New function: perform a single experiment and return accuracy
double runSingleExperiment(int numberLF_trees, int randomSeed, char* projectName) {
    // Reload data using the specified random seed
    train_features.clear();
    train_labels.clear();
    test_features.clear();
    test_labels.clear();
    loadData(randomSeed);

    // Create and train trees
    initCentroidSampleNumbers(nSensors);

    std::cout << "Creating array of " << numberLF_trees << " Learning Hyperplane Trees.\n";
    auto apLF_tree = (cLF_tree**)malloc(numberLF_trees * sizeof(cLF_tree*));
    auto start_first_tree = std::chrono::steady_clock::now();
    double tparallel = 0;

    for (treeNo = 0; treeNo < numberLF_trees; ++treeNo) {
        if (treeNo < numberLF_trees) {
            apLF_tree[treeNo] = create_LF_tree();
            apLF_tree[treeNo]->setTreeNumber(treeNo);
            apLF_tree[treeNo]->setSensorNumber(nSensors);
            int SBcount = 0;
            const auto start_tree = std::chrono::steady_clock::now();
            treeFinal = false;
            apLF_tree[treeNo]->featureShuffl();
            apLF_tree[treeNo]->growTree();
            const auto finish_tree = std::chrono::steady_clock::now();
            const auto elapsed0 = std::chrono::duration_cast<std::chrono::milliseconds>(finish_tree - start_tree);
            if (elapsed0.count() > tparallel) tparallel = (double)elapsed0.count();
            auto elapsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(finish_tree - start_first_tree);
            std::cout << "Growing tree " << treeNo << " took " << elapsed0.count() << " msec. Elapsed " << elapsed1.count() <<
                " msec. To go (est.) " << ceil(elapsed1.count() * (numberLF_trees - treeNo - 1) / ((treeNo + 1) * 60.0)) << " msec." << std::endl;
        }
    }

    auto end_last_tree = std::chrono::steady_clock::now();
    auto trainTime = std::chrono::duration_cast<std::chrono::milliseconds>(end_last_tree - start_first_tree);
    std::cout << "\nResults of testing on TRAINING DATA --  perhaps useful for development \n";
    std::cout << "Mean time to grow a Learning Hyperplane Tree         " << trainTime.count() / (double)numberLF_trees << " msec." << std::endl;
    std::cout << "Estimated parallel training time                     " << tparallel << " msec. " << std::endl;


    // Perform testing and get accuracy
    double accuracy = testOnTEST_SET(numberLF_trees, apLF_tree);

    // Free memory
    for (int i = 0; i < numberLF_trees; i++) {
        if (apLF_tree[i]) {
            delete apLF_tree[i];
        }
    }
    free(apLF_tree);

    return accuracy;
}


double reinforcement(uint8_t action, uint8_t label)  // This training is done by the trainer who knows only the system's action and the label
{
    return (action == label) ? 1 : 0;
}

double testOnTEST_SET(int numberLF_trees, cLF_tree** apLF_tree)
{
    uint32_t numberoftestimages = feature_test_size;
    std::vector<std::vector<double>> pcurrentImage;
    uint8_t label;
    uint8_t chosenAction = 0;
    double maxP_right = 0;
    int lowSumWeightsCount = 0;
    std::vector<double> P_right;
    std::vector<size_t> badRecognitions{};
    P_right.assign(10, 0);
    long goodDecisions = 0;
    auto start_test_set = std::chrono::steady_clock::now();
    for (uint32_t imageNo = 0; imageNo < numberoftestimages; ++imageNo)
    {
        pcurrentImage = test_features;
        label = test_labels[imageNo];
        double P_rightTree = 0; // The probability that a given tree would get it right
        double P_rightAction = 0;
        double accuP_right = 0;
        double accu_weights = 0;
        double wt = 0;
        int count = 0;
        // We determine for each action the probability of being right and the weight
        maxP_right = 0.0;
        for (uint8_t action = 0; action < class_num + 1; ++action)
        {
            count = 0;
            accuP_right = 0;
            for (int treeNo = 0; treeNo < numberLF_trees; ++treeNo)
            {
                if (treeNo % class_num + 1 == action)
                {
                    P_rightTree = apLF_tree[treeNo]->evalBoundedWeightedSB(pcurrentImage[imageNo], wt);
                    accuP_right += P_rightTree;
                    accu_weights += wt;
                    ++count;
                }
            }
            // P_rightAction is the probability of being right on the image for the current
            // action using the weighted average of probabilities over several trees.
            P_rightAction = accuP_right;
            if (P_rightAction > maxP_right) // maximize over all actions
            {
                maxP_right = P_rightAction;
                chosenAction = action;
            }
        }
        if (accu_weights <= 0.0001f)
        {
            ++lowSumWeightsCount; // Skip this sample in the output
        }
        else
        {
            if (reinforcement(chosenAction, label) == 1)   // Has the system learned the behaviour that was positively reinforced?
            {
                ++goodDecisions;
            }
            else
            {
                badRecognitions.push_back(imageNo);
            }
        }
    } // Loop  over test images
    auto test_finished = std::chrono::steady_clock::now();
    double ProbabilityOfCorrect = (double)goodDecisions / feature_test_size;
    if (lowSumWeightsCount > 0) std::cout << "Low sum weights count " << lowSumWeightsCount << "." << std::endl;
    auto elapsed_test = std::chrono::duration_cast<std::chrono::milliseconds>(test_finished - start_test_set);
    // std::cout << "Mean time required to classify an image           " << elapsed_test.count() / feature_test_size << " milliseconds. " << std::endl;
    std::cout << "Probability of a correct decision                 " << ProbabilityOfCorrect * 100.0 << " percent " << std::endl;

    return ProbabilityOfCorrect;
} // End of testOnTEST_SET


