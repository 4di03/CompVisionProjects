
/**
 * Adithya Palle,
 * February 4, 2025
 * 
 * This file is an implementation of various supporting functions for feature extraction.
 * This includes various filtering methods, depth extraction, and reading embeddings from a file.
 */
#include "featureExtractor.h"
#include <fstream>  
#define DEPTH_ANYTHING_MODEL_PATH "/Users/adithyapalle/work/CS5330/depthAnything/da2-code/model_fp16.onnx"

/**
 * Applies a separable filter to the image.
 * 
 * if K is a seperable filter, then K = V * H where H and V are 1D filters.
 * This function then does V*(H*src)  which is equivalent to K*src by the associative and commutative properties of convolution.
 * @param src the source image
 * @param dst the destination image
 * @param verticalFilter the vertical filter (V)
 * @param horizontalFilter the horizontal filter (H)
 * @returns 0 if the operation was successful, -1 otherwise.
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
 * @param src the source image
 * @param dst the destination image
 * @returns 0 if the operation was successful, -1 otherwise.
 */
int sobelX3x3( const cv::Mat &src,  cv::Mat &dst ){
    std::vector<float> vert = {0.25,0.5,0.25};
    std::vector<float> horiz = {0.5,0,-0.5};
    return applySeparableFilter(src, dst, vert,horiz);
}

/**
 * Appies a 3x3 sobel Y filter to the image to highlight horizontal edges.
 * @param src the source image
 * @param dst the destination image
 * @returns 0 if the operation was successful, -1 otherwise.
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
            cv::Vec3b magnitude = cv::Vec3b(sqrt(pixelX[0]*pixelX[0] + pixelY[0]*pixelY[0]),
                                            sqrt(pixelX[1]*pixelX[1] + pixelY[1]*pixelY[1]),
                                            sqrt(pixelX[2]*pixelX[2] + pixelY[2]*pixelY[2]));

            rowDst[j] = magnitude;
        }


    }

    return 0;

}



/**
 * Loads the embeddings (1,512) mat from the csv files which contains the resnet embeddings for each image.
 * @param resnetEmbeddingsFilePath the path to the csv file containing the resnet embeddings
 * @param imagePath the path to the image file
 * @return the resnet embeddings for the image
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
/**
 * Gets depth values in range [0,255] from the input image. 0 represents the closest object and 255 represents the furthest object.
 * 
 * @param src The source image.
 * @param dst The destination single channel mat.
 * 
 * @returns 0 if the operation was successful, -1 otherwise.
 */
int getDepthValues(const cv::Mat&src, cv::Mat &dst){
    // initalize the network once
    static   DA2Network da_net(DEPTH_ANYTHING_MODEL_PATH);
    // scale according to the smaller dimension
    float scale_factor = 256.0 / (src.rows > src.cols ? src.cols : src.rows);
    da_net.set_input( src, scale_factor );
    da_net.run_network( dst, src.size() );

    return 0;
}
CompositeFeatureExtractor* textureHistogram = new CompositeFeatureExtractor({new TextureExtractor(), new Histogram3D(8, true)});
CompositeFeatureExtractor* noBlackTextureHistogram = new CompositeFeatureExtractor({new TextureExtractor(), new Histogram3D(8, false)});

// Map of feature extractors
std::map<std::string, FeatureExtractor*> featureExtractorMap = {
    {"CenterSquare", new CenterSquareFeatureExtractor(7)},
    {"Histogram3D", new Histogram3D(8)},
    {"MultiHistogram", new MultiFeatureExtractor({new Histogram2D(8), new Histogram3D(8)})},
    {"TextureAndColor", new MultiFeatureExtractor({new Histogram3D(8), textureHistogram})},
    {"Resnet", new ResnetFeatureExtractor()},
    {"DepthColor", new CompositeFeatureExtractor({new ForegroundExtractor(), new Histogram3D(8, false)})},
    {"EdgeUniformity", new CompositeFeatureExtractor({new TextureExtractor(), new FFTExtractor(true)})},
};
