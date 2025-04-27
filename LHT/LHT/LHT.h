#include <memory>
#include <limits>
#include <vector>
#include <array>
#include <assert.h>
#include <iostream>
#include <random>

extern double forest_rate;
extern double beta;

extern int numberLF_trees;
extern int minSamples2Split;
extern int constancyLimit;
extern double feature_train_size;
extern size_t feature_dim;
extern size_t treeNo;
extern size_t class_num;
extern double reinforcement(uint8_t action, uint8_t label);
long leafCount{ 0 }; // Counts the number of leaf Sample Blocks (SB), i.e. nodes which compute probabilities and are not used for branching
bool treeFinal{ false }; // Signals the end of tree growth.
size_t non_ID{ 0x3777777777 };
extern std::vector<std::vector<double>> train_features;
extern std::vector<uint8_t> train_labels;
std::vector<double> initC{}; // Each tree has this starting centroid of all samples 
std::vector<size_t> image_numbers_setup{}; // Aids speed.
void initCentroidSampleNumbers(int nSensors);
// In what follows, S (D) means less (greater than or equal to) the features-computed value.
// One can think of it as to the left or right on the axis. Latin: Sinister = left, Dexter = right.

struct SF // Used for features
{
    size_t sensor;
    double factor;
};

struct SB  // The Sample Blocks are of two types: 1. feature nodes that partition sample blocks into two forming a tree
    // and 2. leaf blocks with linear functions as membership functions giving the probability of the target class.
    // When a leaf block is split in two by a feature, it becomes a feature block with the two resulting blocks as leaf children.
{
    bool   is_final{ false };                          /* Initially false; true when SB can't change/split   */
    std::vector<double> C{};                          /* Centroid vector, C[nSensors] is output centroid    */
    std::vector<double> W{};                          /* Weight vector for active domain components         */
    std::vector<size_t> active;                       /* list of pixels deemed non-constant on a block      */
    std::vector<size_t> image_numbers{};              /* a list of indices of images in train_images        */
    std::vector<SF>  features;                        /* A feature vector is at a node of the feature tree  */
    double FTvalue{};                                 /* Value used at a branch of a feature tree           */
    size_t FTS{ non_ID };                              /* Left SBid stored at a branching of a feature tree   */
    size_t FTD{ non_ID };                               /* Right SBid at a feature tree branching              */
    size_t FTparent{ non_ID };                        /* The parent of this block in the feature tree        */
};

class cLF_tree
{
    // Data members
    size_t m_treeNo{ 0 };                  // The number of the current tree. Its associated action is  m_treeNo % 10 when sed for MNIST classification;
    size_t m_nSensors{ feature_dim };              // Number of sensors (pixels) of MNIST data
    std::vector<SB> m_SB{};                // Vector to contain all SB structs

public: // Methods
    // We first create one leaf SB at index 0 of the m_SB vector which is to contain all SB nodes.
    // At the first split, it will become a feature node and two leaves will be its children in the Learning Feature Tree.

    void    setTreeNumber(size_t treeNo);
    void    setSensorNumber(uint32_t nSensors);
    size_t  createSB();
    void    growTree();
    void    createFeature(size_t SBid);
    void    splitSB(size_t SBid, int c);
    size_t  findBlock(std::vector<double> pX, double& weight);
    size_t  findBlock(std::vector<double> pX);
    double  evalBoundedWeightedSB(std::vector<double> pimage, double& weight);
    double  evalBoundedSB(std::vector<double> pimage);
    void    makeSBfinal(size_t SBid);
    void    makeFTnode(size_t SBid);
    void    featureShuffl();
};

cLF_tree* create_LF_tree()
{
    // A Learning Feature tree partitions the training samples in several stages so two classes, targets
    // and non-targets, tend to be concentrated in different blocks. Each partition of a block into two
    // blocks is based on a linear function of sample values. At the leaves of the Feature Tree are
    // linear functions which estimate the probability of the target class at each point of the block.
    //  Classification accuracy is improved by using several trees on the same problem.
    // Learning Feature Trees grow from a single leaf to hundreds or thousands of leaves.
    // They are independent from each other and can grow in parallel on large enough hardware.
    // Evaluation on a new sample is expected to be fast enough for real-time systems.
    auto pLF_tree = new cLF_tree;
    pLF_tree->createSB();
    uint32_t sensor = 0;
    return pLF_tree;
}

void    cLF_tree::setTreeNumber(size_t treeNo)
{
    cLF_tree::m_treeNo = treeNo;
}

void   cLF_tree::setSensorNumber(uint32_t nSensors)
{
    cLF_tree::m_nSensors = nSensors;
}



size_t  cLF_tree::createSB() // The splitting SB at pointer pSB leads to creation and initialization of a new SB
{
    uint32_t nSensors = m_nSensors;
    size_t SBid = m_SB.size();               // Get the unique identifier, or index, of a sample block (SB).
    m_SB.push_back(SB());                    // Add the new SB to the vector m_SB of all SBs (including those used as feature tree nodes)
    m_SB[SBid].is_final = false;             // A final SB does not change any more
    m_SB[SBid].C.assign(nSensors + 1, 0);    // The centroid ( an engram) of the images on the block, a vector including the output centroid at index nSensors
    m_SB[SBid].W.assign(nSensors + 1, 0);    // The weights on nSensors inputs of the linear function on a block. The W indices, like those of C, don't change due to inactivity of sensors
    m_SB[SBid].FTS = non_ID;                 // non_ID is a huge size_t value that indicates a not initialized left branch of a feature tree. (S = sinister in Latin = left ).
    m_SB[SBid].FTD = non_ID;                 // Indicates the above for the right branch. (D = dexter in Latin)
    m_SB[SBid].FTvalue = 0;                  // This is the split value at a node of the feature tree, the value of the SB's feature at the centroid vector or other placement.
    m_SB[SBid].active.clear();
    if (SBid == 0)
    {
        for (uint32_t sensor = 0; sensor < nSensors; ++sensor) // Sets the initially nSensors active sensors of SBid = 0 to 0,1,2 ...nSensors - 1 
            // Others get activity set during a split.
        {
            m_SB[SBid].active.push_back(sensor);
        }
    }
    ++leafCount; // Creates one additional leaf SB
    return SBid;
} // End of createSB

void initCentroidSampleNumbers(int nSensors) //sets up two vectors: initC and image_numbers_setup that initialize trees
{
    double accuC = 0;
    initC.assign(nSensors + 1, 0);
    for (size_t sensor = 0; sensor < nSensors; ++sensor)
    {
        accuC = 0;

        for (const auto& row : train_features) {
            accuC += row[sensor];
        }
        initC[sensor] = accuC / forest_rate * feature_train_size;
    }
    image_numbers_setup.clear();
    for (size_t s = 0; s < forest_rate * feature_train_size; ++s) image_numbers_setup.push_back(s);
    // initC and image_numbers_setup will speed up initialization of SampleBlock SBid = 0 of all trees
} // End of initCentroidSampleNumbers

void cLF_tree::featureShuffl() {

    std::vector<std::pair<std::vector<double>, uint8_t>> train_data;
    for (size_t i = 0; i < train_features.size(); ++i) {
        train_data.emplace_back(train_features[i], train_labels[i]);
    }
    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(train_data.begin(), train_data.end(), g);

    train_features.clear();
    train_labels.clear();
    for (const auto& data : train_data) {
        train_features.push_back(data.first);
        train_labels.push_back(data.second);
    }
}

void cLF_tree::growTree()
{
    if (m_SB.size() == 1)
    {
        treeFinal = false;
        m_SB[0].image_numbers = image_numbers_setup;
    }
    // This routine does four things with image and label data in collaboration with two other routines: createFeature and splitSB.
    // 1. It takes the precomputed centroid of domain dimensions for SB 0 and adds some shifts. The centroids of other SB are computed in splitSB.
    // 2. It determines the variances of the image samples' active component dimensions to support step 3.
    // 3. It removes some domain dimensions whose values vary little in the samples of the block and are "deemed constant" over an SB block, 
    // 4. It finds the weights of linear functions with domain SB to fit the data based on the remaining active dimensions.
    uint32_t nSensors = m_nSensors; // This is the number of domain coordinates (number of pixels in an image = size of retina)
    size_t sensor{ 0 };
    size_t sensorIndex{ 0 };
    size_t SBid{ 0 };
    size_t imageNo{ 0 }; // This is the index of the image and its label in the MNIST training data files of images and labels
    size_t local_imageNo{ 0 }; //This is the index of image numbers (imageNo values) which are stored with the SB.
    double  y{ 0 }; // y is the reinforcement for the given action and image label.
    std::vector<size_t> stayActive{}; // This temporarily records the remaining active sensors prior to elimination of some from the active list
    double accu = 0;
    for (SBid = 0; SBid < m_SB.size(); ++SBid)  // for ( SBid = 0; SBid < size_m_SB; ++SBid) uses iterations
    {
        if (m_SB[SBid].is_final == false && m_SB[SBid].image_numbers.size() > 0)
        {
            if (SBid == 0) // We compute a tree-shifted centroid for the domain coordinates of SB 0, others are done during splitSB
            {
                m_SB[0].features.clear();
                m_SB[0].C = initC; // presomputed to save time
            }
            // In general, an SB can't split if it has too few image samples
            // We have to take the following steps in all cases so the piece can be made final if necessary.
            // we store some values important for computing variance and doing least squares to get weights W
            double En = 0;
            double Ex = 0;
            double Ex2 = 0;
            double Exy = 0;
            double Ey = 0;
            // Compute the mean reinforcement for this SB using its local image numbers
            local_imageNo = 0;
            while (local_imageNo < m_SB[SBid].image_numbers.size())
            {
                imageNo = m_SB[SBid].image_numbers[local_imageNo];
                Ey += reinforcement((uint8_t)(m_treeNo % class_num + 1), train_labels[imageNo]); // ?? this may only be correct for classifying all digits
                ++local_imageNo;
            }
            En = (double)m_SB[SBid].image_numbers.size();
            stayActive.clear(); // Helps to change the activity vector
            long value{ 0 }; // sum of sensor values
            // We now compute the "E values" necessary for solving least squares by the normal equations
            sensorIndex = 0;
            while (sensorIndex < m_SB[SBid].active.size())
            {
                sensor = m_SB[SBid].active[sensorIndex];
                local_imageNo = 0;
                while (local_imageNo < m_SB[SBid].image_numbers.size())
                {
                    imageNo = m_SB[SBid].image_numbers[local_imageNo];
                    //pimage = train_images + imageNo;
                    y = reinforcement((m_treeNo % class_num + 1), train_labels[imageNo]);
                    {
                        //value = pimage->pixels[sensor];
                        value = train_features[imageNo][sensor];
                        // Ex += value; This divided by En just recalculates the centroid.
                        Ex2 += value * value;
                        Exy += value * y;
                    }
                    ++local_imageNo;
                }
                // Take the means
                Ex = m_SB[SBid].C[sensor];
                Ex2 /= En;
                Exy /= En;
                Ey /= En;
                m_SB[SBid].C[nSensors] = Ey; // This is the mean value of the output component of the centroid for this SB
                // Compute the mean value and variance of intensity for each active sensor. 
                // Low variance entails making the sensor inactive on the SB ( and hence on all results of splitting SB).
                // An inactive sensor doesn't participate in features of a block.
                // Determine which variables are deemed constant over this SB (by the constancy criterion).
                double Variance = Ex2 - Ex * Ex;

                stayActive.push_back(sensor); // Record the fact that this sensor is not considered constant
                // If this SB can't split, we need the weights of the active sensors
                m_SB[SBid].W[sensor] = (Exy - Ex * Ey) / Variance;
                ++sensorIndex;
            }
            m_SB[SBid].active.clear(); // Remove the constant sensors on block SBid starting by clearing the active vector
            size_t j = 0;
            while (j < stayActive.size())
            {
                m_SB[SBid].active.push_back(stayActive[j]);
                ++j;
            }
        } // End if SBid is_final = false && m_SB[SBid].image_numbers.size() > 0 )condition
        if (m_SB[SBid].image_numbers.size() < minSamples2Split)
        {
            if (m_SB[SBid].image_numbers.size() == 0) m_SB[SBid].C[nSensors] = 0;
            makeSBfinal(SBid);
        }
        else
        {
            createFeature(SBid);
        }
    } // End of for ( SBid = 0; SBid < size_m_SB; ++SBid)
} // End of growTree

void cLF_tree::createFeature(size_t SBid)
{
    // Precondition: m_SB[SBid].image_numbers.size() >= 2 * minSamples2Split && 9 * m_SB[SBid].image_numbers.size() >= 2 * m_SB[SBid].active.size()
    // This is a multi-level feature extraction part of the algorithm. By appropriate splitting, it tries to concentrate target-
    //  and non-target images on different children of split SB. This automatic feature construction needs much further work.
    // For example, we can choose the position of the split so that all targets are all on the D side, i.e. they have greater
    // values than some threshold. If this is the case, there are no targets on the S side and
    //  the S side is made final with function constant 0 and no active sensors if the number of imange samples is sufficient.
    // To start, we need a rough idea of the distribution of targets and non-targets on the SB on each sensor.
    // We find the upper and lower bounds of each set for active sensors.(Inactive sensors don't have enough range to discriminate.).
    int nSensors = m_nSensors;
    size_t activeSize = m_SB[SBid].active.size();
    size_t sensor{ 0 };
    size_t imageNo{ 0 };
    long countTargets = 0;
    long countNonTargets = 0;
    size_t local_imageNo = 0;
    while (local_imageNo < m_SB[SBid].image_numbers.size())
    {
        imageNo = m_SB[SBid].image_numbers[local_imageNo];
        if (reinforcement((m_treeNo % class_num + 1), train_labels[imageNo]) == 1) // target class
        {
            ++countTargets;
        }
        else
        {
            ++countNonTargets;
        }
        ++local_imageNo;
    }
    if (countNonTargets == 0)
    {
        makeSBfinal(SBid);
        m_SB[SBid].C[nSensors] = 1.0; // All images on SBid are targets
        return;
    }
    if (countTargets == 0)
    {
        makeSBfinal(SBid);
        m_SB[SBid].C[nSensors] = 0; // All images on SBid are non_targets
        return;
    }
    // Both are non-zero, so we have a mixture of targets and non-targets
    // We look for sensors that show a difference between distributions of targets and non-targets
    double fltCountT = countTargets;
    double fltCountN = countNonTargets;
    m_SB[SBid].C[nSensors] = fltCountT / (fltCountT + fltCountN); // This can be enhanced by weights calculated previously


    // We collect statistics using upper and lower bounds on both targets and non-targets
    std::vector<float> tSensor; // for computing centroids of intensity of targets for sensors
    tSensor.assign(nSensors, 0);
    std::vector<float> nontSensor; // same for non-targets
    nontSensor.assign(nSensors, 0);
    size_t sensorIndex = 0;
    while (sensorIndex < activeSize) // Only active sensors have enough variation to discriminate
    {

        sensor = m_SB[SBid].active[sensorIndex];
        local_imageNo = 0;
        while (local_imageNo < m_SB[SBid].image_numbers.size())
        {
            imageNo = m_SB[SBid].image_numbers[local_imageNo];
            if (reinforcement((m_treeNo % class_num + 1), train_labels[imageNo]) == 1) // targets
            {
                tSensor[sensor] += (float)train_features[imageNo][sensor];
            }
            else
            {
                if (rand() % 100 < 100) {
                    nontSensor[sensor] += (float)train_features[imageNo][sensor];
                }
            }
            ++local_imageNo;
        }
        // tSensor[sensor] is the total intensity of that sensor (pixel) for all the training *target* images on SB.
        // nontSensor[sensor] is for the non-targets..
        ++sensorIndex;

    }
    // There is at least one target and one non-target image as a result of the first step above.
    // We take the sensor of maximal difference of centroids for sure and some others of lesser difference.
    m_SB[SBid].features.clear();
    double sensorDiff = 0;
    double maxSD = -FLT_MAX;
    double minSD = FLT_MAX;
    double maxAbsSD;

    // get the maximum difference of centroids over all active semsors
    sensorIndex = 0;
    while (sensorIndex < activeSize)
    {
        sensor = m_SB[SBid].active[sensorIndex];
        // Examine the target and non-target centroids of image values on the active sensors
        //sensorDiff = tSensor[sensor] / (double)countTargets - nontSensor[sensor] / (double)countNonTargets;
        sensorDiff = tSensor[sensor] / (double)countTargets - nontSensor[sensor] / ((double)countNonTargets * 1);
        //sensorDiff = tSensor[sensor] / (double)countTargets;
        if (maxSD < sensorDiff) maxSD = sensorDiff;
        if (minSD > sensorDiff) minSD = sensorDiff;
        ++sensorIndex;
    }
    // Pick out features                                                                  
    int featureCount = 0;
    sensorIndex = 0;
    SF sf;
    while (sensorIndex < activeSize)
    {
        //std::cout << "rate" << rate << std::endl;
        sensor = m_SB[SBid].active[sensorIndex];
        //sensorDiff = tSensor[sensor] / (double)countTargets - nontSensor[sensor] / (double)countNonTargets;
        sensorDiff = tSensor[sensor] / (double)countTargets - nontSensor[sensor] / ((double)countNonTargets * 1);


        maxAbsSD = std::max(fabs(minSD), fabs(maxSD));
        if (fabs(sensorDiff) >= ((double)m_treeNo / (double)numberLF_trees) * beta * maxAbsSD) {
            //if (fabs(sensorDiff) >= beta * maxAbsSD) {
            sf.sensor = sensor; sf.factor = sensorDiff / maxAbsSD;
            m_SB[SBid].features.push_back(sf); // Form the features vector, which is a vector of sensors (as ints).
        }

        ++sensorIndex;
    }
    tSensor.clear();
    nontSensor.clear();
    // Normalize the feature vector any change of a single image pixel by one unit changes the 
    // feature by no more than one unit
    double maxfactor = 0;
    for (auto sf : m_SB[SBid].features) if (fabs(sf.factor) > maxfactor) maxfactor = fabs(sf.factor);
    for (auto sf : m_SB[SBid].features) sf.factor /= maxfactor;





    // Mow we have the features vector. Let's see how it works.
    double featureSum; // Sum of all the sensor values of an image for sensors in the feature vector.
    double minTFS = FLT_MAX; // minimum Target Feature Sum below all targets(on this SB)
    double maxTFS = -FLT_MAX; // maximum Target Feature Sum above all targets (on this SB)
    double minNFS = FLT_MAX; // minimum Non-Target Feature Sum below all non-targets(on this SB)
    double maxNFS = -FLT_MAX; // maximum Non-Target Feature Sum above all non-targets (on this SB)
    local_imageNo = 0;
    while (local_imageNo < m_SB[SBid].image_numbers.size())
    {
        imageNo = m_SB[SBid].image_numbers[local_imageNo];
        //pimage = train_images + imageNo;
        featureSum = 0;
        //for (auto sf : m_SB[SBid].features) featureSum += pimage->pixels[sf.sensor] * sf.factor;
        for (auto sf : m_SB[SBid].features) featureSum += train_features[imageNo][sf.sensor] * sf.factor;
        if (reinforcement((m_treeNo % class_num + 1), train_labels[imageNo]) == 1)
        {
            if (featureSum < minTFS) minTFS = featureSum;
            if (featureSum > maxTFS) maxTFS = featureSum;
        }
        else
        {
            if (featureSum < minNFS) minNFS = featureSum;
            if (featureSum > maxNFS) maxNFS = featureSum;
        }
        ++local_imageNo;
    }



    long tbminNFS = 0;
    long tamaxNFS = 0;
    long nbminTFS = 0;
    long namaxTFS = 0;
    double epsilon = 1e-9;
    local_imageNo = 0;

    while (local_imageNo < m_SB[SBid].image_numbers.size())
    {
        imageNo = m_SB[SBid].image_numbers[local_imageNo];
        //pimage = train_images + imageNo;
        featureSum = 0;
        //for (auto sf : m_SB[SBid].features) featureSum += pimage->pixels[sf.sensor] * sf.factor;
        for (auto sf : m_SB[SBid].features) featureSum += train_features[imageNo][sf.sensor] * sf.factor;
        {
            // We can calculate exactly for each SB
            // 1. The number tbminNFS of targets with feature value below minNFS
            // 2. The number tamaxNFS of targets with feature value above maxNFS
            // 3. The number nbminTFS of non-targets with feature value below minTFS
            // 4. The number namaxTFS of non_targets with feature value above maxTFS
            // We choose a threshold corresponding to the maximum number of images in the groups 1 to 4.
            if (reinforcement((m_treeNo % class_num + 1), train_labels[imageNo]) == 1)
            {
                // Targets
                if (featureSum < minNFS - epsilon)
                {
                    ++tbminNFS;
                }
                if (featureSum > maxNFS + epsilon)
                {
                    ++tamaxNFS;
                }
            }
            else
            {
                // Non-Targets        
                if (featureSum < minTFS - epsilon)
                {
                    ++nbminTFS;
                }
                if (featureSum > maxTFS + epsilon)
                {
                    ++namaxTFS;
                }
            }
        }
        ++local_imageNo;
    }
    // We need a struct to find which of the above cases has a maximal sized set of images above the minimum size.
    struct cv { int c; long v; };
    cv cv1{ 1, tbminNFS };  cv cv2{ 2, tamaxNFS }; cv cv3{ 3, nbminTFS }; cv cv4{ 4, namaxTFS };
    cv ans1 = (cv1.v > cv2.v) ? cv1 : cv2; cv ans2 = (cv3.v > cv4.v) ? cv3 : cv4;
    cv ans = (ans1.v > ans2.v) ? ans1 : ans2;

    if (ans.v < minSamples2Split) { ans.c = 5; } // Case 5 splits the piece in two at a mid-point of all four threshold values

    if (minTFS > maxNFS + epsilon) { ans.c = 6; }  //Not necessary
    if (minNFS > maxTFS + epsilon) { ans.c = 7; }   //Not necessary
    //std::cout << "ans.c: " << ans.c <<"tamaxNFS"<< tamaxNFS << std::endl;

    //std::cout <<"randomEpsilon"<< randomEpsilon <<"minNFS"<< minNFS<<"maxNFS"<< maxNFS <<"minTFS"<< minTFS <<"maxTFS"<< maxTFS<< std::endl;
    switch (ans.c)
    {
    case 1: m_SB[SBid].FTvalue = minNFS; break;
        //case 2: m_SB[SBid].FTvalue = maxNFS+ 0.0000000000001; break;
    case 2: m_SB[SBid].FTvalue = maxNFS + epsilon; break;
    case 3: m_SB[SBid].FTvalue = minTFS; break;
        //case 4: m_SB[SBid].FTvalue = maxTFS+ 0.0000000000001; break;
    case 4: m_SB[SBid].FTvalue = maxTFS + epsilon; break;
    case 5:/* featureSum = 0; // Another way of choosing FTvalue
            for (SF sf : m_SB[SBid].features)
            {
                featureSum += sf.factor * m_SB[SBid].C[sf.sensor];
            }
            m_SB[SBid].FTvalue = featureSum;*/

        m_SB[SBid].FTvalue = (minNFS + maxNFS + minTFS + maxTFS) / 4; break;
    case 6: m_SB[SBid].FTvalue = (minTFS + maxNFS) / 2; break;
    case 7: m_SB[SBid].FTvalue = (minNFS + maxTFS) / 2;
    }

    // std::cout << "Case " << ans.c << " threshold " << m_SB[SBid].FTvalue;
    // In case 1, the S part is made final and its C[nSensors]  = 1.0,
    // in case 2, the D part is made final and its C[nSensors] = 1.0,
    // in case 3, the S part is made final and its C[nSensors] = 0,
    // in case 4, the D part is made final and its C[nSensors] = 0.
    // in case 5, neither part is made final and its C[nSensors] is not set in splitSB.

    if (minNFS == maxTFS && maxTFS == minTFS && minTFS == maxNFS) {
        m_SB[SBid].is_final = true;
        std::cout << "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa: " << std::endl;
    }
    else {
        splitSB(SBid, ans.c);
    }
} // End of createFeature

void cLF_tree::splitSB(size_t SBid, int cxx)
{
    // Splitting adds two SBs,  but removes one leaf SB which now does branching
    int nSensors = m_nSensors;
    size_t SSBid = createSB(); // Create two child SBs N.B. Before this, we have to make sure they will each have enough images.
    size_t DSBid = createSB();
    --leafCount; // We created two leaf SBs, but changing the leaf parent to a branching node of the LF-tree removes one leaf SB
    // Set the children's activity like SBid's activity
    size_t sensorIndex = 0;
    size_t sensor = 0;
    // Copy the activity vector of SBid to the children
    while (sensorIndex < m_SB[SBid].active.size())
    {
        sensor = m_SB[SBid].active[sensorIndex];
        m_SB[SSBid].active.push_back(sensor);
        m_SB[DSBid].active.push_back(sensor);
        ++sensorIndex;
    }
    // Assign the image numbers to the S (left) or D (right) child
    size_t imageNo = 0;
    // Compute the feature values of the images whose numbers are on SBid for distribution of the numbers to the children
    m_SB[SSBid].C.assign(nSensors + 1, 0); // Make room for the output centroid
    m_SB[DSBid].C.assign(nSensors + 1, 0);
    int Sn = 0; int Dn = 0;
    size_t local_imageNo = 0;
    while (local_imageNo < m_SB[SBid].image_numbers.size())
    {
        imageNo = m_SB[SBid].image_numbers[local_imageNo];
        // compute the feature value of the image
        double featureSum = 0;
        for (auto sf : m_SB[SBid].features) featureSum += (double)train_features[imageNo][sf.sensor] * sf.factor;
        // Use the feature to compute the centroids of the children and to distribute the image numbers to the children
        sensor = 0;
        while (sensor < nSensors)
        {
            if (featureSum < m_SB[SBid].FTvalue)
            {
                ++Sn;
                m_SB[SSBid].C[sensor] += (double)train_features[imageNo][sensor];
            }
            else
            {
                ++Dn;
                m_SB[DSBid].C[sensor] += (double)train_features[imageNo][sensor];
            }
            ++sensor;
        }
        if (featureSum < m_SB[SBid].FTvalue)
        {
            // put the image index in train_images on the S-side images
            m_SB[SSBid].image_numbers.push_back(imageNo);
        }
        else
        {
            // put the image number on the D-side.
            m_SB[DSBid].image_numbers.push_back(imageNo);
        }
        ++local_imageNo;
    }
    sensor = 0;
    while (sensor < nSensors)
    {
        m_SB[SSBid].C[sensor] /= Sn; // S-side centroid
        m_SB[DSBid].C[sensor] /= Dn; // D-side centroid
        ++sensor;
    }
    m_SB[SBid].image_numbers.clear();
    makeFTnode(SBid); // Only the leaves of the feature tree will be processed
    m_SB[SSBid].is_final = false;
    m_SB[DSBid].is_final = false;
    // In case 1, the S part is made final and its C[nSensors]  = 1.0,
    // in case 2, the D part is made final and its C[nSensors] = 1.0,
    // in case 3, the S part is made final and its C[nSensors] = 0,
    // in case 4, the D part is made final and its C[nSensors] = 0.
    switch (cxx)
    {
    case 1: m_SB[SSBid].is_final = true; m_SB[SSBid].C[nSensors] = 1.0; break;
    case 2: m_SB[DSBid].is_final = true; m_SB[DSBid].C[nSensors] = 1.0; break;
    case 3: m_SB[SSBid].is_final = true; m_SB[SSBid].C[nSensors] = 0; break;
    case 4: m_SB[DSBid].is_final = true; m_SB[DSBid].C[nSensors] = 0; break;
    case 5:;
    }
    // Set the parent and child indices in the LF-tree for branching
    m_SB[SSBid].FTparent = m_SB[DSBid].FTparent = SBid;
    m_SB[SBid].FTS = SSBid;
    m_SB[SBid].FTD = DSBid;
}  // End of splitSB

size_t cLF_tree::findBlock(std::vector<double> pX, double& weight)
{
    //weight = 1.0;
    weight = 100000.0;
    size_t Blkid = 0;
    while (true)
    {
        if (m_SB[Blkid].FTS == non_ID)
        {
            return Blkid; // Blkid is the leaf sought
        }
        else
        {
            // Blkid is not the goal of the search
            // Compute the feature for the Blkid feature tree node
            double featureSum = 0;
            for (auto sf : m_SB[Blkid].features)
            {
                featureSum += pX[sf.sensor] * sf.factor;
            }
            double dist = featureSum - m_SB[Blkid].FTvalue;
            Blkid = (dist < 0) ? m_SB[Blkid].FTS : m_SB[Blkid].FTD;
            dist = fabs(dist);
            weight = dist < weight ? dist : weight;
        }
    }
} // End of findBlock(size_t BlockID, mnist_image_t* pX)

size_t cLF_tree::findBlock(std::vector<double> pX) // Faster version without weight
{
    size_t Blkid = 0;
    while (true)
    {
        if (m_SB[Blkid].FTS == non_ID)
        {
            return Blkid; // Blkid is the leaf sought
        }
        else
        {
            // Blkid is not the goal of the search
            // Compute the feature for the Blkid feature tree node
            double featureSum = 0;
            for (auto sf : m_SB[Blkid].features)
            {
                featureSum += pX[sf.sensor] * sf.factor;
            }
            double dist = featureSum - m_SB[Blkid].FTvalue;
            Blkid = (dist < 0) ? m_SB[Blkid].FTS : m_SB[Blkid].FTD;
        }
    }
}

double cLF_tree::evalBoundedWeightedSB(std::vector<double> pimage, double& weight)
{
    // Evaluates the bounded weighted function of an SB for the given image and returns the weight as a reference
    int nSensors = m_nSensors;
    size_t SBid = findBlock(pimage, weight);
    double value = m_SB[SBid].C[nSensors];
    size_t sensorIndex = 0;
    size_t sensor = 0;
    while (sensorIndex < m_SB[SBid].active.size())
    {
        sensor = m_SB[SBid].active[sensorIndex];
        value += m_SB[SBid].W[sensor] * ((double)(pimage[sensor]) - m_SB[SBid].C[sensor]);
        ++sensorIndex;
    }
    if (value < 0.0) return 0; // We could replace these three lines by a sigmoid function
    if (value >= 1.0) return weight;
    return value * weight;
} //End of evalBoundedWeightedSB

double cLF_tree::evalBoundedSB(std::vector<double> pimage) // For evaluating individual trees on the training set
{
    // evaluates the linear function of the block
    int nSensors = m_nSensors;
    size_t SBid = findBlock(pimage);
    double value = m_SB[SBid].C[nSensors];
    size_t sensorIndex = 0;
    size_t sensor = 0;
    while (sensorIndex < m_SB[SBid].active.size())
    {
        sensor = m_SB[SBid].active[sensorIndex];
        value += m_SB[SBid].W[sensor] * ((double)(pimage[sensor]) - m_SB[SBid].C[sensor]);
        ++sensorIndex;
    }
    if (value < 0.0) return 0; // We could replace these three lines by a sigmoid function
    if (value >= 1.0) return 1.0;
    return value;
} //End of evalBoundedSB

void cLF_tree::makeSBfinal(size_t SBid)
{
    m_SB[SBid].is_final = true;
    m_SB[SBid].features.clear();
    if (m_SB[SBid].image_numbers.size() < m_SB[SBid].active.size()) m_SB[SBid].active.clear();
    m_SB[SBid].image_numbers.clear();
}

void cLF_tree::makeFTnode(size_t SBid) // After splitting, a non-final leaf becomes a feature tree node. The feature vector and FTvalue are kept.
{
    m_SB[SBid].is_final = true;
    m_SB[SBid].image_numbers.clear();
}



