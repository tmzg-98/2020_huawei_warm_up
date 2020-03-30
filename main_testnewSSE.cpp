/*
 * 神经网络参考代码 https://blog.csdn.net/Pumpkin_love/article/details/79892874\
 * Windows下SSE参考博客:
 *      https://blog.csdn.net/chenchong_219/article/details/44683155
 *      https://www.cnblogs.com/zyl910/archive/2012/07/31/intrin02_sse.html
 *      https://www.cnblogs.com/dragon2012/p/5200698.html
 *      http://blog.sina.com.cn/s/blog_1511ffacb0102wihv.html
 * Linux下SSE参考：
 *      https://www.jianshu.com/p/70601b36540f
 *      http://infocenter.arm.com/help/basic/help.jsp?topic=/com.arm.doc.dui0204ic/
 *      https://www.cnblogs.com/xylc/p/5410517.html
 *      https://blog.csdn.net/EmSoftEn/article/details/51718763?utm_source=blogxgwz7
 *      https://www.jianshu.com/p/3212bdb50469
 * */
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <cstring>
//#define TEST
using namespace std;

#ifdef WINDOWS_SSE
#include <xmmintrin.h>
#define MEXTEND(x) _mm_set_ps1(x)
typedef __m128 M;
M M0VALUE = _mm_setzero_ps();
M M1VALUE = MEXTEND(1.0f);
#else
#include <arm_neon.h>
typedef float32x4_t M;
#define _mm_add_ps vaddq_f32
#define _mm_mul_ps vmulq_f32
#define _mm_sub_ps vsubq_f32
#define _mm_div_ps
#define MEXTEND(x) vdupq_n_f32(x)
M M0VALUE = MEXTEND(0.0f);
M M1VALUE = MEXTEND(1.0f);
inline M _mm_set_ps(float a, float b, float c, float d){
    float t[4] = {a, b, c, d};
    return vld1q_f32(t);
}
#endif

#ifdef TEST
#include <ctime>
clock_t startTime;
#define CURTIME(x) cout << #x << ": " << 1000*(clock()-startTime)/(float)CLOCKS_PER_SEC<<"ms"<<endl
#endif
float predict(bool isWrite);
void netForward(const M input[], const int &label, bool isAddNum);
void backward(int order);
inline float sigmoid(const float &x);
inline float sigmoidD(const float &x);
inline void sigmoid(M &x, const M &y);
inline void sigmoidD(M &x, const M &y);
void loadTrainData();

// 常量设定
static const int inputNodeNum = 1000;
static const int hideNodeNum = 4;
static const int outputNodeNum = 1;
static const int maxTrain = 12;
static const int maxTrainUseSize = 4000;
constexpr static const float WR = 0.2f; // Weight learning efficiency
constexpr static const float TR = 0.2f; // Threshold learning efficiency
static const M MWR {WR, WR, WR, WR};
static const M MTR {TR, TR, TR, TR};

// 输入及参数
char trainDataString[maxTrainUseSize * inputNodeNum * 10];
M hideWeight[inputNodeNum];
M hideThresh;
M outputWeight;
float outputThresh;
// 临时变量存放数组
M trainData[maxTrainUseSize][inputNodeNum / hideNodeNum];
int trainLabel[maxTrainUseSize];
M hideResult;
float outputResult;
M hideDelta;
float outputDelta;
M tempTestInput[inputNodeNum / hideNodeNum];
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
    hideThresh = M0VALUE;
    outputWeight = M0VALUE;
    outputThresh = 0;
}

void train(){
#ifdef TEST
    CURTIME(开始训练);
#endif
    loadTrainData();
#ifdef TEST
    CURTIME(完成数据加载);
#endif
    hideThresh = _mm_div_ps(M1VALUE, MEXTEND(2.0f));
    for(int iter=0; iter < maxTrain; iter++){
        correctNum = 0;
        for(int j=0; j < trainSize; j++){
            netForward(trainData[j], trainLabel[j], true);
            backward(j);
        }
#ifdef TEST
        clock_t t = clock();
        cout << "目前处于第" << iter << "轮\t\ttrainAucc：" << (float)correctNum / (float)trainSize
             << "\ttestAucc：" << predict(false) << endl;
        startTime += clock() - t;
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

float predict(bool isWrite){
#ifdef TEST
    int rightNum = 0, totalNum = 0;
    ifstream ans_file(ansFile);
    assert(ans_file.is_open());
#endif
    static FILE *test_file = fopen(testFile.c_str(), "rb");
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
        for(M & i : tempTestInput){
            i = _mm_set_ps((float)read(testP, testString) / 1000, (float)read(testP, testString) / 1000, (float)read(testP, testString) / 1000, (float)read(testP, testString) / 1000);
            //cout << i << " @" << testP << "@ ";
        }
        //cout << endl;
        netForward(tempTestInput, 0, false);
        if(isWrite)
            predict_file << (outputResult > 0.5 ? 1 : 0) << "\n";
#ifdef TEST
        int ans;
        ans_file >> ans;
        rightNum += ((outputResult > 0.5 ? 1 : 0) == ans) ? 1 : 0;
        totalNum++;
#endif
    }
#ifdef TEST
    return (float)rightNum / (float)totalNum;
#else
    return 0.0;
#endif
}

void netForward(const M input[], const int &label, bool isAddNum){
    hideResult = hideThresh;
    for(int i=0; i < inputNodeNum / hideNodeNum; i++){
        hideResult = _mm_add_ps(hideResult, _mm_mul_ps(hideWeight[i], input[i]));
    }
    sigmoid(hideResult, hideResult);

    M t = _mm_mul_ps(outputWeight, hideResult);
    outputResult = sigmoid(outputThresh + t[0] + t[1] + t[2] + t[3]);
    if(isAddNum)
        correctNum += (outputResult > 0.5 ? 1 : 0) == label ? 1 : 0;
}

void backward(int order){
    outputDelta = ((float)trainLabel[order] - outputResult) * sigmoidD(outputResult);
    hideDelta = _mm_mul_ps(outputWeight, MEXTEND(outputDelta));
    M t1;
    sigmoidD(t1, hideResult);
    hideDelta = _mm_mul_ps(hideDelta, t1);

    outputWeight = _mm_add_ps(outputWeight, _mm_mul_ps(MEXTEND(WR * outputDelta), hideResult));

    outputThresh += TR * outputDelta;

    for(int i=0; i < inputNodeNum / hideNodeNum; i++){
        hideWeight[i] = _mm_add_ps(hideWeight[i], _mm_mul_ps(_mm_mul_ps(MWR, trainData[order][i]), hideDelta));
    }
    hideThresh = _mm_add_ps(hideThresh, _mm_mul_ps(MTR, hideDelta));
}

inline float sigmoid(const float &x){
    return 1.0f / (1.0f + exp(-x));
}

inline float sigmoidD(const float &x){
    return x * (1.0f - x);
}

inline void sigmoid(M &x, const M &y){
#ifdef WINDOWS_SSE
    M t = _mm_set_ps(exp(-y[0]), exp(-y[1]), exp(-y[2]), exp(-y[3]));
    x = _mm_div_ps(M1VALUE, _mm_add_ps(M1VALUE, t));
#else
    x = _mm_set_ps(sigmoid(y[0]), sigmoid(y[0]), sigmoid(y[0]), sigmoid(y[0]));
#endif
}

inline void sigmoidD(M &x, const M &y) {
    x = _mm_mul_ps(y, _mm_sub_ps(M1VALUE, y));
}

void loadTrainData(){
#ifdef TEST
    CURTIME(准备fread);
#endif
    FILE *fp = fopen(trainFile.c_str(), "rb");
    assert(fp);
    fread(trainDataString, 1, sizeof(trainDataString), fp);
#ifdef TEST
    CURTIME(完成fread);
#endif
    int p = 0;
    for(int i=0; i < maxTrainUseSize; i++){
        //cout << "!!! ";
        for(int j=0; j < inputNodeNum / hideNodeNum; j++){
            trainData[i][j] = _mm_set_ps((float)read(p, trainDataString) / 1000, (float)read(p, trainDataString) / 1000, (float)read(p, trainDataString) / 1000, (float)read(p, trainDataString) / 1000);
            //cout << trainData[i][j] << " ";
        }
        trainLabel[i] = trainDataString[++p] - '0';
        p++;
        //cout << "$$$ " << trainLabel[i] << endl;
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
    CURTIME(预测结束);
    vector<int> answerVec;
    vector<int> predictVec;
    int correctCount;
    float accurate;

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

    accurate = ((float)correctCount) / answerVec.size();
    cout << "the prediction accuracy is " << accurate << endl;
    CURTIME(结束);
#endif
    return 0;
}
