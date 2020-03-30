#include <iostream>
#include <utility>
#include <vector>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <cmath>
#define TEST
using namespace std;

struct Data {
    vector<double> features;
    int label;
    Data(vector<double>  f, int l) : features(std::move(f)), label(l){}
};
struct Param {
    vector<double> pMean; // positive mean
    vector<double> nMean; // negative mean
    int pNum, nNum;
    void clear() {pMean.clear(); nMean.clear(); pNum = nNum = 0;}
};


class LR {
public:
    void train();
    void predict();
    /*
    int loadModel();
    int storeModel();
     */
    LR(const string& trainFile, const string& testFile, const string& predictOutFile);

private:
    vector<Data> trainDataSet;
    vector<Data> testDataSet;
    vector<int> predictVec;
    Param param;
    string trainFile;
    string testFile;
    string predictOutFile;
#ifdef TEST
    string weightParamFile = "output/modelweight.txt";
#else
    string weightParamFile = "modelweight.txt";
#endif

private:
    bool init();
    bool loadTrainData();
    bool loadTestData();
    int storePredict(vector<int> &predict);
    void initParam();

private:
    int featuresNum;
};

LR::LR(const string& trainF, const string& testF, const string& predictOutF)
{
    trainFile = trainF;
    testFile = testF;
    predictOutFile = predictOutF;
    featuresNum = 0;
    assert(init());
}

bool LR::loadTrainData()
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

void LR::initParam()
{
    int i;
    for (i = 0; i < featuresNum; i++) {
        param.pMean.push_back(0.0);
        param.nMean.push_back(0.0);
    }
    param.pNum = param.nNum = 0;
}

bool LR::init()
{
    trainDataSet.clear();
    bool status = loadTrainData();
    if (!status) {
        return false;
    }
    featuresNum = trainDataSet[0].features.size();
    param.clear();
    initParam();
    return true;
}

void LR::train()
{
    for(auto & data : trainDataSet){
        vector<double>::iterator mean;
        if(data.label == 1){
            mean = param.pMean.begin();
            param.pNum++;
        }
        else{
            mean = param.nMean.begin();
            param.nNum++;
        }

        for(auto & num : data.features){
            *mean += num;
        }
    }

    for(auto p = param.pMean.begin(), n = param.nMean.begin(); p != param.pMean.end(); p++, n++){
        *p /= param.pNum, *n /= param.nNum;
    }
}

void LR::predict()
{
    loadTestData();
    for(auto & test: testDataSet){
        auto pMean = param.pMean.begin();
        auto nMean = param.nMean.begin();
        double score = 0.0;
        for(auto & num: test.features){
            if(abs(num - *pMean) <= abs(*pMean - *nMean) && abs(num - *nMean) <= abs(*pMean - *nMean))
            score += abs(num - *nMean) - abs(num - *pMean);
        }
        predictVec.push_back(score > 0 ? 1 : 0);
    }
    storePredict(predictVec);
}
/*
int LR::loadModel()
{
    string line;
    int i;
    vector<double> wtTmp;
    double dbt;

    ifstream fin(weightParamFile.c_str());
    if (!fin) {
        cout << "打开模型参数文件失败" << endl;
        exit(0);
    }

    getline(fin, line);
    stringstream sin(line);
    for (i = 0; i < featuresNum; i++) {
        char c = sin.peek();
        if (c == -1) {
            cout << "模型参数数量少于特征数量，退出" << endl;
            return -1;
        }
        sin >> dbt;
        wtTmp.push_back(dbt);
    }
    param.wtSet.swap(wtTmp);
    fin.close();
    return 0;
}

int LR::storeModel()
{
    cout << "#####" << endl;
    string line;
    int i;

    ofstream fout(weightParamFile.c_str());
    if (!fout.is_open()) {
        cout << "打开模型参数文件失败" << endl;
    }
    if (param.wtSet.size() < featuresNum) {
        cout << "wtSet size is " << param.wtSet.size() << endl;
    }
    for (i = 0; i < featuresNum; i++) {\
        fout << param.wtSet[i] << " ";
    }
    fout.close();
    return 0;
}
*/
bool LR::loadTestData()
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
    return true;
}

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

int LR::storePredict(vector<int> &predict)
{
    string line;
    int i;

    ofstream fout(predictOutFile.c_str());
    if (!fout.is_open()) {
        cout << "打开预测结果文件失败" << endl;
    }
    for (i = 0; i < predict.size(); i++) {
        fout << predict[i] << endl;
    }
    fout.close();
    return 0;
}

int main(int argc, char *argv[])
{
    vector<int> answerVec;
    vector<int> predictVec;
    int correctCount;
    double accurate;

    string trainFile = "/data/train_data.txt";
    string testFile = "/data/test_data.txt";
    string predictFile = "/projects/student/result.txt";
    string answerFile = "/projects/student/answer.txt";

#ifdef TEST
    trainFile = "../data/train_data.txt";
    testFile = "../data/test_data.txt";
    predictFile = "../output/result.txt";
    answerFile = "../data/answer.txt";
#endif

    LR logist(trainFile, testFile, predictFile);

    cout << "ready to train model" << endl;
    logist.train();

    cout << "training ends, ready to store the model" << endl;
    //logist.storeModel();

#ifdef TEST
    cout << "ready to load answer data" << endl;
    loadAnswerData(answerFile, answerVec);
#endif

    cout << "let's have a prediction test" << endl;
    logist.predict();

#ifdef TEST
    loadAnswerData(predictFile, predictVec);
    cout << "test data set size is " << predictVec.size() << endl;
    correctCount = 0;
    for (int j = 0; j < predictVec.size(); j++) {
        if (j < answerVec.size()) {
            if (answerVec[j] == predictVec[j]) {
                correctCount++;
            }
        } else {
            cout << "answer size less than the real predicted value" << endl;
        }
    }

    accurate = ((double)correctCount) / answerVec.size();
    cout << "the prediction accuracy is " << accurate << endl;
#endif

    return 0;
}
