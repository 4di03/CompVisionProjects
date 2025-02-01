/**
 * 
 */
#include <opencv2/opencv.hpp>
#include <fstream>  

#define DEFAULT_RESNET_EMBEDDINGS_FILE_PATH "/Users/adithyapalle/work/CS5330/content_based_image_retrieval/ResNet18_olym.csv"




class FeatureExtractor {

public:
    // Extract features from an image (cv::Mat) into a feature vector (cv::Mat)
    virtual std::vector<cv::Mat> extractFeatures(const cv::Mat& image) = 0;

    // Extract features from the path to an image file (std::string) into a feature vector (cv::Mat)
    virtual std::vector<cv::Mat> extractFeaturesFromFile(const std::string& imagePath){
        cv::Mat image = cv::imread(imagePath);
        return extractFeatures(image);
    }

};

// ideally these two would be implemented with composition, but for times sake I will use inheritance



class SingleFeatureExtractor : public FeatureExtractor {
public:
    // Extract features from an image (cv::Mat) into a feature vector (cv::Mat)
    virtual cv::Mat _extractFeatures(const cv::Mat& image) = 0;
    // Extract features from an image (cv::Mat) into a feature vector (cv::Mat)
    std::vector<cv::Mat> extractFeatures(const cv::Mat& image){
        std::vector<cv::Mat> features;
        features.push_back(_extractFeatures(image));
        return features;
    }
};



class CenterSquareFeatureExtractor : public SingleFeatureExtractor{
    private:
        // The size of the center square
        int size;

    public:
        CenterSquareFeatureExtractor(int size) : size(size) {}


        /**
         * Extract the center square of the image with dimensions size x size
         */
        cv::Mat _extractFeatures(const cv::Mat& image) override
        {
            // Check if the image is empty
            if (image.empty())
            {
                throw std::invalid_argument("Image is empty");
            }

            // Get the center square of the image
            cv::Rect roi(image.cols / 2 - size / 2, image.rows / 2 - size / 2, size, size);
            cv::Mat centerSquare = image(roi);

            return centerSquare;
        }

};


class Histogram3D : public SingleFeatureExtractor{

    private:
        int numBins;

    public:
        Histogram3D(int numBins) : numBins(numBins) {}

        /**
         * Extract the 3D histogram of the image, using numBins bins for each channel (RGB).
         * The histogram is normalized so that the sum of all bins is 1.
         * @param image the input image (CV_8UC3)
         * @return the 3D histogram (CV_8U) with dimension (numBins, numBins, numBins)
         */
        cv::Mat _extractFeatures(const cv::Mat& image) override
        {

            // Check if the image is empty
            if (image.empty())
            {
                throw std::invalid_argument("Image is empty");
            }
            int histSize[] = {numBins, numBins, numBins};
            cv::Mat hist(3, histSize, CV_32F, cv::Scalar(0)); // Proper 3D histogram
            unsigned char binSize = 256 / numBins;
            // Loop over all pixels
            for (int i = 0; i < image.rows; i++)
            {
                const cv::Vec3b* ptr = image.ptr<cv::Vec3b>(i); // Pointer to row i
                for (int j = 0; j < image.cols; j++)
                {

                    // Get the RGB values
                    unsigned char b = ptr[j][0];
                    unsigned char g = ptr[j][1];
                    unsigned char r = ptr[j][2];

                    unsigned char bIndex = b/binSize;
                    unsigned char gIndex = g/binSize;
                    unsigned char rIndex = r/binSize;


                    // Increment the histogram
                    float prevVal = hist.at<float>(bIndex,gIndex,rIndex);
                    int idx[] = {bIndex, gIndex, rIndex};
                    hist.at<float>(idx)++;
                    float postVal = hist.at<float>(bIndex,gIndex,rIndex);



                }
            }
           
            hist = hist / (image.rows * image.cols);

            return hist;
        }
};


class Histogram2D : public SingleFeatureExtractor{

    private:
        int numBins = 8;
    public:
        Histogram2D(int numBins) : numBins(numBins) {}

        /**
         * Generate a 2D chromaticity histogram of the image (R and G channels).
         * @param image the input image (CV_8UC3)
         * @ return the 2D histogram (CV_8U) with dimension (numBins, numBins). Only the top left triangle is filled as the histogram values are normalized such that r+g+b=1
         */
        cv::Mat _extractFeatures(const cv::Mat& image) override{


            // The below code is a modified snippet from Bruce Maxwell's makeHist.cpp demo


            cv::Mat hist = cv::Mat::zeros( cv::Size( numBins, numBins ), CV_32FC1 );

            // loop over all pixels
            for( int i=0;i<image.rows;i++) {
                const cv::Vec3b *ptr = image.ptr<cv::Vec3b>(i); // pointer to row i
                for(int j=0;j<image.cols;j++) {

                // get the RGB values
                float B = ptr[j][0];
                float G = ptr[j][1];
                float R = ptr[j][2];

                // compute the r,g chromaticity
                float divisor = R + G + B;
                divisor = divisor > 0.0 ? divisor : 1.0; // check for all zeros
                float r = R / divisor;
                float g = G / divisor;

                // compute indexes, r, g are in [0, 1]
                int rindex = (int)( r * (numBins - 1) + 0.5 );
                int gindex = (int)( g * (numBins - 1) + 0.5 );

                // increment the histogram
                hist.at<float>(rindex, gindex)++;

                // keep track of the size of the largest bucket (just so we know what it is)
                float newvalue = hist.at<float>(rindex, gindex);
                }
            }

              hist /= (image.rows * image.cols); // normalize the histogram by the number of pixels

            return hist;
        }
};




/**
 * Applies a separable filter to the image.
 * 
 * if K is a seperable filter, then K = V * H where H and V are 1D filters.
 * This function then does V*(H*src)  which is equivalent to K*src by the associative and commutative properties of convolution.
 * 
 */
int applySeparableFilter(const cv::Mat& src, cv::Mat& dst, std::vector<float>& verticalFilter, std::vector<float>& horizontalFilter){
    if (horizontalFilter.size() %2 != 1 || verticalFilter.size() %2 != 1){
        // make sure the filters are odd in dimensions so that we can center them
        return -1;
    }
    if(src.empty()){
        return -1;
    }

    // copy src so that we dont have to read and write to the same image
    cv::Mat srcCopy;

    src.convertTo(srcCopy, CV_32FC3); // convert to float so that we can apply math with proper precision
    // make dst a copy of the data in source so that border pixels are not modified
    src.copyTo(dst);
    dst.convertTo(dst, CV_16SC3); // convert to 16 bit signed int so that we can store the results of the kernel operation

    
    cv::Mat tmp = cv::Mat::zeros(src.size(), CV_32FC3);

    int horizontalCutoff = horizontalFilter.size()/2;

    //std::cout << "Kernel vector: " << kernelVector[0] << " " << kernelVector[1] << " " << kernelVector[2] << " " << kernelVector[3] << " " << kernelVector[4] << std::endl;
    // horizontal filter pass
    for(int i = 0; i < srcCopy.rows; i++){
        cv::Vec3f* origRow = srcCopy.ptr<cv::Vec3f>(i);
        cv::Vec3f* row = tmp.ptr<cv::Vec3f>(i);
        for(int j = horizontalCutoff; j < srcCopy.cols-horizontalCutoff; j++){
            // modify the row in srcCopy as we need to store these results for the vertical pass
            for(int k = -horizontalCutoff; k <= horizontalCutoff; k++){
                row[j] += origRow[j+k] * horizontalFilter[horizontalCutoff + k];
            }
        }
    }

    int verticalCutoff = verticalFilter.size()/2;


    // vertical filter pass
    for (int j = horizontalCutoff; j < tmp.cols - horizontalCutoff; j++){
        for (int i = verticalCutoff; i < tmp.rows - verticalCutoff; i++){

            cv::Vec3f sum = cv::Vec3f(0, 0, 0); 

            for(int k = -verticalCutoff; k <= verticalCutoff; k++){
                sum += tmp.at<cv::Vec3f>(i+k, j) * verticalFilter[verticalCutoff + k];
            }

            // modify dst directly
            dst.at<cv::Vec3s>(i, j) = sum;

        }
    }
    return 0;

}
/**
 * Appies a 3x3 sobel X filter to the image to highlight vertical edges.
 * 
 */
int sobelX3x3( const cv::Mat &src,  cv::Mat &dst ){
    std::vector<float> vert = {0.25,0.5,0.25};
    std::vector<float> horiz = {0.5,0,-0.5};
    return applySeparableFilter(src, dst, vert,horiz);
}

/**
 * Appies a 3x3 sobel Y filter to the image to highlight horizontal edges.
 * 
 */
int sobelY3x3( const cv::Mat &src,  cv::Mat &dst ){
    std::vector<float> vert = {0.5,0,-0.5};
    std::vector<float> horiz = {0.25,0.5,0.25};

    return applySeparableFilter(src, dst, vert,horiz);
}


/**
 * Computes the gradient magnitude image from two 3 channel gradient images.
 * The output is remains a three channel image.
 * 
 * @param sx The gradient image in the x direction (3 - channel, signed short).
 * @param sy The gradient image in the y direction (3 - channel, signed short).
 * @param dst The destination image (3 - channel, unsigned int).
 * @returns 0 if the operation was successful, -1 otherwise.
 */
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst ){
    if (sx.empty() || sy.empty() || sx.size() != sy.size()){
        return -1;
    }

    // initalize dst as a uchar 3 channel image
    dst.create(sx.size(), CV_8UC3);

    for (int i = 0; i < sx.rows; i++){
        cv::Vec3s* rowX = sx.ptr<cv::Vec3s>(i);
        cv::Vec3s* rowY = sy.ptr<cv::Vec3s>(i);
        cv::Vec3b* rowDst = dst.ptr<cv::Vec3b>(i);

        for (int j = 0;  j < sx.cols; j++){
            cv::Vec3s pixelX = rowX[j];
            cv::Vec3s pixelY = rowY[j];

            // compute the magnitude of the gradient for all 3 channels
            cv::Vec3b magnitude = cv::Vec3s(sqrt(pixelX[0]*pixelX[0] + pixelY[0]*pixelY[0]),
                                            sqrt(pixelX[1]*pixelX[1] + pixelY[1]*pixelY[1]),
                                            sqrt(pixelX[2]*pixelX[2] + pixelY[2]*pixelY[2]));

            rowDst[j] = magnitude;
        }


    }

    return 0;

}




class TextureExtractor : public SingleFeatureExtractor{
    private:
        int numBins = 8;
    public:
        TextureExtractor(int numBins) : numBins(numBins) {}

        /** 
         * Generates a sobel gradient magnitude histogram (3D) from the input image.
         * @param src - the input image (CV_8UC3)
         * @param numBins - the number of bins to use for the histogram
         * @return the gradient magnitude histogram (CV_8U) with dimensions (numBins, numBins, numBins)
         */
        cv::Mat _extractFeatures(const cv::Mat& src) override{
            cv::Mat hist = cv::Mat::zeros( cv::Size( 256, 1 ), CV_8U );

            cv::Mat sx, sy;
            if (sobelX3x3(src, sx) != 0) {
                std::cerr << "Error applying sobelX3x3" << std::endl;
                exit(-1);
            }
            if (sobelY3x3(src, sy) != 0) {
                std::cerr << "Error applying sobelY3x3" << std::endl;
                exit(-1);
            }
            cv::Mat mag;
            if (magnitude(sx, sy, mag) != 0) {
                std::cerr << "Error getting gradient magnitude" << std::endl;
                exit(-1);
            }
            
            Histogram3D hist3D(numBins);

            return hist3D._extractFeatures(mag);
        }   
};



class MultiFeatureExtractor : public FeatureExtractor {
private: 
    std::vector <SingleFeatureExtractor*> featureExtractors;
    /**
     * Computes multiple feature vectors from an image
     * @param image the input image (CV_8UC3)
     * @return  a vector of 3D histograms (CV_8U) with dimension (numBins, numBins, numBins)
     */
    std::vector<cv::Mat> _extractFeatures(const cv::Mat& image) 
    {
        // make both histograms with a 8 bins
        Histogram2D hist2D(8);
        Histogram3D hist3D(8);

        std::vector<cv::Mat> histograms;
        for (SingleFeatureExtractor* featureExtractor : featureExtractors){
            histograms.push_back(featureExtractor->_extractFeatures(image));
        }

        return histograms;

    }

public:
    MultiFeatureExtractor(std::vector <SingleFeatureExtractor*> featureExtractors) : featureExtractors(featureExtractors) {}
    // Extract features from an image (cv::Mat) into a set of feature vector (std::vector<cv::Mat>)
    std::vector<cv::Mat> extractFeatures(const cv::Mat& image){
        return _extractFeatures(image);
    }
};


/**
 * Loads the embeddings (1,512) mat from the csv files which contains the resnet embeddings for each image.
 * @param resnetEmbeddingsFilePath the path to the csv file containing the resnet embeddings
 * @param imagePath the path to the image file
 */
cv::Mat readEmbeddingsFromFile(std::string resnetEmbeddingsFilePath, std::string imagePath){
    // get filename from path
    std::size_t found = imagePath.find_last_of("/\\");
    std::string imageName;
    if (found != std::string::npos){
        imageName = imagePath.substr(found+1);
    }else{
        imageName = imagePath; // if no path is found, just use the image name as this means there is no parent directory in the path
    }



    std::ifstream file(resnetEmbeddingsFilePath);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " + resnetEmbeddingsFilePath << std::endl;
        exit(-1);
    }
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<std::string> row;

        while (std::getline(ss, value, ',')) {  // Split by comma
            row.push_back(value);
        }

        if(row[0] == imageName){
            // convert the string to a vector of floats
            std::vector<float> embeddings;
            for (int i = 1; i < row.size(); i++){
                embeddings.push_back(std::stof(row[i]));
            }
            cv::Mat resnetEmbeddings = cv::Mat(embeddings);
            return resnetEmbeddings;
        }
        
    }
    file.close();

    std::cerr << "Embeddings not found for image: " + imagePath + " in file: " + resnetEmbeddingsFilePath << std::endl;
    exit(-1);

}

class ResnetFeatureExtractor : public FeatureExtractor{
    private:
        std::string resnetEmbeddingsFilePath;
        std::map<std::string, cv::Mat> resnetEmbeddingsMap;

    public:

        ResnetFeatureExtractor(std::string resnetEmbeddingsFilePath = DEFAULT_RESNET_EMBEDDINGS_FILE_PATH)
        {
            this->resnetEmbeddingsFilePath = resnetEmbeddingsFilePath;

            // preload the resnet embeddings into a map
            std::ifstream file(resnetEmbeddingsFilePath);
            if (!file.is_open()) {
                std::cerr << "Error opening file: " + resnetEmbeddingsFilePath << std::endl;
                exit(-1);
            }
            std::string line;
            while (std::getline(file, line)) {
                std::stringstream ss(line);
                std::string value;
                std::vector<std::string> row;

                while (std::getline(ss, value, ',')) {  // Split by comma
                    row.push_back(value);
                }

                // convert the string to a vector of floats
                std::vector<float> embeddings;
                for (int i = 1; i < row.size(); i++){
                    embeddings.push_back(std::stof(row[i]));
                }
                // the true flag is used to copy the embeddings to avoid dangling pointer issue where embedding gets corrupted
                cv::Mat resnetEmbeddings = cv::Mat(embeddings, true);
                if (resnetEmbeddings.empty()){
                    std::cerr << "Error creating cv::Mat from embeddings, embeddings empty" << std::endl;
                    exit(-1);
                }

                if(norm(resnetEmbeddings) == 0){
                    std::cerr << "Embeddings have zero norm" << std::endl;
                    exit(-1);
                }

                // no need for copy as assignment increases the refernce count of cv::Mat so the memory is not deallocated
                this->resnetEmbeddingsMap[row[0]] = resnetEmbeddings;
            
            }
            file.close();
        }



        /**
         * Unimplemented method
         */
        std::vector<cv::Mat> extractFeatures(const cv::Mat& image) override
        {
            std::cerr << "Not implemented" << std::endl;
            exit(-1);
        }

        /**
         * Extracts the resnet embeddings from the image file path
         * @param imagePath the path to the image file
         * @return the resnet embeddings (cv::Mat) (1,512) for the image
         */
        std::vector<cv::Mat> extractFeaturesFromFile(const std::string& imagePath) override{
        
            // Read the embeddings from the file
            std::vector<cv::Mat> resnetEmbeddings;

            //get the image name
            std::size_t found = imagePath.find_last_of("/\\");
            std::string imageName;
            if (found != std::string::npos){
                imageName = imagePath.substr(found+1);
            }else{
                imageName = imagePath; // if no path is found, just use the image name as this means there is no parent directory in the path
            }

 

            // check if the image is in the map
            if (resnetEmbeddingsMap.find(imageName) == resnetEmbeddingsMap.end()){
                std::cerr << "Embeddings not found for image: " + imageName + " in file: " + resnetEmbeddingsFilePath << std::endl;
                exit(-1);
            }
            
            //retrieve the embeddings from the map
            resnetEmbeddings.push_back(resnetEmbeddingsMap[imageName]);
            return resnetEmbeddings;
        }   

};

// Map of feature extractors to their names
std::map<std::string, FeatureExtractor*> featureExtractorMap = {
    {"CenterSquare", new CenterSquareFeatureExtractor(7)},
    {"Histogram3D", new Histogram3D(8)},
    {"MultiHistogram", new MultiFeatureExtractor({new Histogram2D(8), new Histogram3D(8)})},
    {"TextureAndColor", new MultiFeatureExtractor({new Histogram3D(8), new TextureExtractor(8)})},
    {"Resnet", new ResnetFeatureExtractor()},
};
