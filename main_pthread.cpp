#include <iostream>
#include <utility>
#include <vector>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <cstring>
//#define PTHREAD
#define TEST
using namespace std;

#ifdef PTHREAD
#include <pthread.h>
pthread_mutex_t lock;
#endif

#ifdef TEST
#include <ctime>
clock_t startTime;
#define CURTIME(x) cout << #x << ": " << 1000*(clock()-startTime)/(double)CLOCKS_PER_SEC<<"ms"<<endl
#endif

struct Data {
    vector<double> features;
    int label;
    Data(vector<double>  f, int l) : features(std::move(f)), label(l){}
};

class NeuralNetwork{
public:
    // 常量设定
    static const int featuresNum = 1000;
    static const int inputNodeNum = 1000;
    static const int hideNodeNum = 5;
    static const int outputNodeNum = 1;
    static const int maxTrain = 5;
    constexpr static const double WR = 0.5; // Weight learning efficiency
    constexpr static const double TR = 0.5; // Threshold learning efficiency
    // 输入及参数
    vector<Data> trainDataSet;
    vector<Data> testDataSet;
    double hideWeight[inputNodeNum][hideNodeNum];
    double hideThresh[hideNodeNum];
    double outputWeight[hideNodeNum][outputNodeNum];
    double outputThresh[outputNodeNum];
    // 临时变量存放数组
    double hideResult[hideNodeNum];
    double outputResult[outputNodeNum];
    double hideDelta[hideNodeNum];
    double outputDelta[outputNodeNum];
    // 其他
    string trainFile, testFile, ansFile, predictOutFile;
    int trainSize;
    int testSize;

    void init(const string& train_file, const string& test_file,
            const string& ans_file, const string& predict_outfile){
#ifdef TEST
        CURTIME(开始类初始化);
#endif
        trainFile = train_file;
        testFile = test_file;
        ansFile = ans_file;
        predictOutFile = predict_outfile;
#ifdef TEST
        CURTIME(准备进行memset);
#endif
        memset(hideWeight, 0, sizeof(hideWeight));
        memset(hideThresh, 0, sizeof(hideThresh));
        memset(outputWeight, 0, sizeof(outputWeight));
        memset(outputThresh, 0, sizeof(outputThresh));
#ifndef PTHREAD
#ifdef TEST
        CURTIME(准备载入数据);
#endif
        assert(loadTrainData());
        trainSize = trainDataSet.size();
#endif
#ifdef TEST
        assert(loadTestData(true));
#else
        assert(loadTestData(false));
#endif
        testSize = testDataSet.size();
#ifdef TEST
        CURTIME(初始化结束);
#endif
    }

    void train(){
#ifdef PTHREAD
        pthread_mutex_lock(&lock);
#endif
        for(int iter=0; iter < maxTrain; iter++){
            for(int j=0; j < trainSize; j++){
                forward(j, trainDataSet);
                backward(j);
            }
#ifdef TEST
            cout << "目前处于第" << iter << "轮\t\ttrainAucc：" << getAccu(trainDataSet)
                << "\ttestAucc：" << getAccu(testDataSet) << endl;
#endif
            /*
            if(iter != 0 && iter % 5 == 0 && getAccu(trainDataSet) <= 0.20){
                cout << "Iter: " << iter << endl;
                break;
            }
             */
        }
#ifdef TEST
        CURTIME(训练结束);
#endif
        storePredict();
#ifdef TEST
        CURTIME(保存预测);
#endif
    }

    void forward(int order, vector<Data> &input){
        for(int i=0; i < hideNodeNum; i++){
            hideResult[i] = hideThresh[i];
            for(int j=0; j < inputNodeNum; j++){
                hideResult[i] += hideWeight[j][i] * input[order].features[j];
            }
            hideResult[i] = sigmoid(hideResult[i]);
        }

        for(int i=0; i < outputNodeNum; i++){
            outputResult[i] = outputThresh[i];
            for(int j=0; j < hideNodeNum; j++){
                outputResult[i] += outputWeight[j][i] * hideResult[j];
            }
            outputResult[i] = sigmoid(outputResult[i]);
        }
    }

    void backward(int order){
        for(int i=0; i < outputNodeNum;i++){\
            outputDelta[i] = (trainDataSet[order].label - outputResult[i]) * sigmoidD(outputResult[i]);
        }
        for(int i=0; i < hideNodeNum; i++){
            hideDelta[i] = 0;
            for(int j=0; j < outputNodeNum; j++){
                hideDelta[i] += outputWeight[i][j] * outputDelta[j];
            }
            hideDelta[i] *= sigmoidD(hideResult[i]);
        }

        for(int i=0; i < hideNodeNum; i++){
            for(int j=0; j < outputNodeNum; j++){
                outputWeight[i][j] += WR * outputDelta[j] * hideResult[i];
            }
        }
        for(int i=0; i < outputNodeNum; i++){
            outputThresh[i] += TR * outputDelta[i];
        }
        for(int i=0; i < inputNodeNum; i++){
            for(int j=0; j < hideNodeNum; j++){
                hideWeight[i][j] += WR * hideDelta[j] * trainDataSet[order].features[i];
            }
        }
        for(int i=0; i < hideNodeNum; i++){
            hideThresh[i] += TR * hideDelta[i];
        }
    }

    double getAccu(vector<Data> &input){
        double ans = 0.0;
        for(int i=0; i < input.size(); i++){
            forward(i, input);
            //cout << "### " << outputResult[0] << " " << input[i].label << endl;
            ans += abs((outputResult[0] >= 0.5 ? 1 : 0) - input[i].label);
        }
        return ans / input.size();
    }

    inline static double sigmoid(double x){
        return 1.0 / (1 + exp(-x));
    }

    inline static double sigmoidD(double x){
        return x * (1 - x);
    }
#ifndef PTHREAD
    bool loadTrainData()
    {
        ifstream infile(trainFile.c_str());
        string line;

        if (!infile) {
            cout << "打开训练文件失败" << endl;
            exit(0);
        }

        while (infile) {
            getline(infile, line);
            if (line.size() > featuresNum) {
                stringstream sin(line);
                char ch;
                double dataV;
                int i;
                vector<double> feature;
                i = 0;

                while (sin) {
                    char c = sin.peek();
                    if (int(c) != -1) {
                        sin >> dataV;
                        feature.push_back(dataV);
                        sin >> ch;
                        i++;
                    } else {
                        cout << "训练文件数据格式不正确，出错行为" << (trainDataSet.size() + 1) << "行" << endl;
                        return false;
                    }
                }
                int ftf;
                ftf = (int)feature.back();
                feature.pop_back();
                trainDataSet.emplace_back(feature, ftf);
            }
        }
        infile.close();
        return true;
    }
#endif
    bool loadTestData(int isHaveAns)
    {
        ifstream infile(testFile.c_str());
        string lineTitle;

        if (!infile) {
            cout << "打开测试文件失败" << endl;
            exit(0);
        }

        while (infile) {
            vector<double> feature;
            string line;
            getline(infile, line);
            if (line.size() > featuresNum) {
                stringstream sin(line);
                double dataV;
                int i;
                char ch;
                i = 0;
                while (i < featuresNum && sin) {
                    char c = sin.peek();
                    if (int(c) != -1) {
                        sin >> dataV;
                        feature.push_back(dataV);
                        sin >> ch;
                        i++;
                    } else {
                        cout << "测试文件数据格式不正确" << endl;
                        return false;
                    }
                }
                testDataSet.emplace_back(feature, 0);
            }
        }
        infile.close();

        if(!isHaveAns)
            return true;

        ifstream ansfile(ansFile.c_str());
        if (!ansfile) {
            cout << "打开答案文件失败" << endl;
            exit(0);
        }
        auto testIter = testDataSet.begin();
        while (ansfile) {
            string line;
            int aw;
            getline(ansfile, line);
            if (!line.empty()) {
                stringstream sin(line);
                sin >> aw;
                testIter->label = aw;
                testIter++;
            }
        }
        ansfile.close();

        return true;
    }

    int storePredict()
    {

        string line;
        ofstream fout(predictOutFile.c_str());
        if (!fout.is_open()) {
            cout << "打开预测结果文件失败" << endl;
        }
        for(int i=0; i < testSize; i++){
            forward(i, testDataSet);
            fout << (outputResult[i] > 0.5 ? 1 : 0) <<endl;
        }
        fout.close();
        return 0;
    }

};
#ifdef TEST
bool loadAnswerData(const string& awFile, vector<int> &awVec)
{
    ifstream infile(awFile.c_str());
    if (!infile) {
        cout << "打开答案文件失败" << endl;
        exit(0);
    }

    while (infile) {
        string line;
        int aw;
        getline(infile, line);
        if (!line.empty()) {
            stringstream sin(line);
            sin >> aw;
            awVec.push_back(aw);
        }
    }

    infile.close();
    return true;
}
#endif

NeuralNetwork network;

#ifdef PTHREAD
void loadTrainData(const char trainFile[])
{
    ifstream infile(trainFile);
    string line;

    if (!infile) {
        cout << "打开训练文件失败" << endl;
        exit(0);
    }

    while (infile) {
        getline(infile, line);
        if (line.size() > NeuralNetwork::featuresNum) {
            stringstream sin(line);
            char ch;
            double dataV;
            int i;
            vector<double> feature;
            i = 0;
            while (sin) {
                char c = sin.peek();
                if (int(c) != -1) {
                    sin >> dataV;
                    feature.push_back(dataV);
                    sin >> ch;
                    i++;
                } else {
                    cout << "训练文件数据格式不正确，出错行为" << (network.trainDataSet.size() + 1) << "行" << endl;
                    return;
                }
            }
            int ftf;
            ftf = (int)feature.back();
            feature.pop_back();
            network.trainDataSet.emplace_back(feature, ftf);
        }
    }
#ifdef TEST
    cout << "线程获取数据完毕" << endl;
#endif
    pthread_mutex_unlock(&lock);
    infile.close();
}
#endif

int main(int argc, char *argv[])
{
    string trainFile = "/data/train_data.txt";
    string testFile = "/data/test_data.txt";
    string predictFile = "/projects/student/result.txt";
    string answerFile = "/projects/student/answer.txt";

#ifdef TEST
    trainFile = "../data/train_data.txt";
    testFile = "../data/test_data.txt";
    predictFile = "../output/result.txt";
    answerFile = "../data/answer.txt";
    startTime = clock();
    CURTIME(开始);
#endif

#ifdef PTHREAD
    pthread_mutex_init(&lock, NULL);
    pthread_t readFile;
    pthread_mutex_lock(&lock);
    int ret = pthread_create(&readFile, NULL, reinterpret_cast<void *(*)(void *)>(loadTrainData),
            (void *)trainFile.c_str());
    assert(ret == 0);
#endif

    network.init(trainFile, testFile, answerFile, predictFile);

    // cout << "ready to train model" << endl;
    network.train();
#ifdef TEST
    vector<int> answerVec;
    vector<int> predictVec;
    int correctCount;
    double accurate;

    cout << "training ends, ready to store the model" << endl;
    //logist.storeModel();

    cout << "ready to load answer data" << endl;
    loadAnswerData(answerFile, answerVec);

    cout << "let's have a prediction test" << endl;
    //logist.predict();

    loadAnswerData(predictFile, predictVec);
    cout << "test data set size is " << predictVec.size() << endl;
    correctCount = 0;
    for (int j = 0; j < predictVec.size(); j++) {
        if (j < answerVec.size()) {
            if (answerVec[j] == predictVec[j]) {
                correctCount++;
            }
        } else {
            cout << "answer size less than the real predicted value: " << answerVec.size() << endl;
            break;
        }
    }

    accurate = ((double)correctCount) / answerVec.size();
    cout << "the prediction accuracy is " << accurate << endl;
#endif
    return 0;
}
