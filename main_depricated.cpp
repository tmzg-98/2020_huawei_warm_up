#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <cstdio>
#define TEST
using namespace std;

#ifdef TEST
#include <ctime>
clock_t startTime;
#define CURTIME(x) cout << #x << ": " << 1000*(clock()-startTime)/(double)CLOCKS_PER_SEC<<"ms"<<endl
double predict(bool isWrite);
#endif
void netForward(const double input[], const int &label, bool isAddNum);
void backward(int order);
inline double sigmoid(double x);
inline double sigmoidD(double x);
inline int read(int &p, const char *str);
void loadTrainData();

// 常量设定
static const int inputNodeNum = 1000;
static const int hideNodeNum = 5;
static const int outputNodeNum = 1;
static const int maxTrain = 5;
static const int maxTrainUseSize = 4000;
constexpr static const double WR = 0.1; // Weight learning efficiency
constexpr static const double TR = 0.1; // Threshold learning efficiency

// 输入及参数
double hideWeight[inputNodeNum][hideNodeNum];
double hideThresh[hideNodeNum];
double outputWeight[hideNodeNum][outputNodeNum];
double outputThresh[outputNodeNum];
// 临时变量存放数组
char trainDataString[maxTrainUseSize * inputNodeNum * 10];
double trainData[maxTrainUseSize][inputNodeNum];
int trainLabel[maxTrainUseSize];
double hideResult[hideNodeNum];
double outputResult[outputNodeNum];
double hideDelta[hideNodeNum];
double outputDelta[outputNodeNum];
double tempTestInput[inputNodeNum];
// 其他
string trainFile, testFile, ansFile, predictOutFile;
int trainSize;
int correctNum = 0;


inline int read(int &p, const char *str) {
    int x = 0, w = 1;
    char ch;
    for(ch = str[p]; ch < '0' || ch > '9' ; ch = str[++p]){ // ch 不是数字时
        if (ch == '-') w = -1; // 判断是否为负
    }
    p += 2; // 跳过小数点及前面的0
    int order = 0;
    for(ch = str[p]; ch >= '0' && ch <= '9'; ch = str[++p], order++){
        if(order < 3){
            x = x * 10 + (ch - '0');  // 将新读入的数字’加’在 x 的后面
            // x 是 int 类型，char 类型的 ch 和 ’0’ 会被自动转为其对应的
            // ASCII 码，相当于将 ch 转化为对应数字
            // 此处也可以使用 (x<<3)+(x<<1) 的写法来代替 x*10
        }
    }
    for(int i=order; i < 3; i++){
        x *= 10;
    }
    return x * w;  // 数字 * 正负号 = 实际数值
}

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

    loadTrainData();
}

void train(){
#ifdef TEST
    CURTIME(开始训练);
#endif
    for(int iter=0; iter < maxTrain; iter++){
        correctNum = 0;
        for(int j=0; j < trainSize; j++){
            netForward(trainData[j], trainLabel[j], true);
            backward(j);
        }
#ifdef TEST
        cout << "目前处于第" << iter << "轮\t\ttrainAucc：" << (double)correctNum / trainSize
             << "\ttestAucc：" << predict(false) << endl;
        CURTIME(当前时间);
#endif
        if(correctNum > trainSize * 4 / 5){
            break;
        }
    }
#ifdef TEST
    CURTIME(训练结束);
#endif
}

double predict(bool isWrite){
#ifdef TEST
    int rightNum = 0, totalNum = 0;
    ifstream ans_file(ansFile);
    assert(ans_file.is_open());
#endif
    static FILE *test_file = fopen(testFile.c_str(), "rb+");
    fseek(test_file, 0, SEEK_END);
    long testSize = ftell(test_file);
    fseek(test_file, 0, SEEK_SET);

    static char *testString = (char*)malloc(sizeof(char) * testSize + 10);
    assert(fread(testString, 1, sizeof(char) * testSize, test_file) == testSize);
    ofstream predict_file(predictOutFile);
    int testP = 0;

    //int testCurTurn = 0;
    while(testP < testSize - 1000){
        //cout << "当前第" << testCurTurn++ << "轮： " << testP << " " << testSize << endl;
        for(double & i : tempTestInput){
            i = (double)read(testP, testString) / 1000;
           //cout << i << " @" << testP << "@ ";
        }
        //cout << endl;
        netForward(tempTestInput, 0, false);
        if(isWrite)
            predict_file << (outputResult[0] > 0.5 ? 1 : 0) << "\n";
#ifdef TEST
        int ans;
        ans_file >> ans;
        rightNum += ((outputResult[0] > 0.5 ? 1 : 0) == ans) ? 1 : 0;
        totalNum++;
#endif
    }
#ifdef TEST
    return (double)rightNum / totalNum;
#else
    return 0.0;
#endif
}

void netForward(const double input[], const int &label, bool isAddNum){
    for(int i=0; i < hideNodeNum; i++){
        hideResult[i] = hideThresh[i];
        for(int j=0; j < inputNodeNum; j++){
            hideResult[i] += hideWeight[j][i] * input[j];
        }
        hideResult[i] = sigmoid(hideResult[i]);
    }

    for(int i=0; i < outputNodeNum; i++){
        outputResult[i] = outputThresh[i];
        for(int j=0; j < hideNodeNum; j++){
            outputResult[i] += outputWeight[j][i] * hideResult[j];
        }
        outputResult[i] = sigmoid(outputResult[i]);
        if(isAddNum)
            correctNum += (outputResult[i] > 0.5 ? 1 : 0) == label ? 1 : 0;
    }
}

void backward(int order){
    for(int i=0; i < outputNodeNum;i++){\
            outputDelta[i] = (trainLabel[order] - outputResult[i]) * sigmoidD(outputResult[i]);
    }
    for(int i=0; i < hideNodeNum; i++){
        hideDelta[i] = 0;
        for(int j=0; j < outputNodeNum; j++){
            hideDelta[i] += outputWeight[i][j] * outputDelta[j];
        }
        hideDelta[i] = hideDelta[i] * sigmoidD(hideResult[i]);
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
            hideWeight[i][j] += WR * hideDelta[j] * trainData[order][i];
        }
    }
    for(int i=0; i < hideNodeNum; i++){
        hideThresh[i] += TR * hideDelta[i];
    }
}

inline double sigmoid(double x){
    return 1.0 / (1 + exp(-x));
}

inline double sigmoidD(double x){
    return x * (1 - x);
}

void loadTrainData(){
    FILE *fp = fopen(trainFile.c_str(), "rb+");
    assert(fp);
    fread(trainDataString, 1, sizeof(trainDataString), fp);
    int p = 0;
    for(int i=0; i < maxTrainUseSize; i++){
        for(int j=0; j < inputNodeNum; j++){
            trainData[i][j] = (double)read(p, trainDataString) / 1000;
            //cout << "*** " << trainData[i][j] << endl;
        }
        trainLabel[i] = read(p, trainDataString);
    }
    trainSize = maxTrainUseSize;
    fclose(fp);
}

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

int main(int argc, char *argv[])
{
    std::ios::sync_with_stdio(false);
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

    init(trainFile, testFile, answerFile, predictFile);

    // cout << "ready to train model" << endl;
    train();
    predict(true);
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
    CURTIME(结束);
#endif
    return 0;
}
