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
 * 指定核心：
 *      https://blog.csdn.net/jackailson/article/details/50927136
 *      https://blog.csdn.net/qq_26697045/article/details/89457241
 * */
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <sched.h>
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
vector<int> AnswerVec;
#endif
float predict(int thread, bool isWrite);
void netForward(int thread, const M input[], const int &label, bool isAddNum);
void backward(int thread, int order);
inline float sigmoid(const float &x);
inline float sigmoidD(const float &x);
inline void sigmoid(M &x, const M &y);
inline void sigmoidD(M &x, const M &y);
// 常量设定
static const int inputNodeNum = 1000;
static const int hideNodeNum = 4;
static const int outputNodeNum = 1;
static const int maxTrain = 100;
static const int maxTrainUseSize = 4000;
constexpr static const float WR = 0.1f; // Weight learning efficiency
constexpr static const float TR = 0.1f; // Threshold learning efficiency
static const M MWR {0.1f, 0.1f, 0.1f, 0.1f};
static const M MTR {0.1f, 0.1f, 0.1f, 0.1f};
// 线程相关
static const int threadNum = 1;
cpu_set_t cpuSet[4];
pthread_t loadDataThread[2];
pthread_t trainThread[threadNum];
int sendNum[] = {0, 1, 2, 3};
// 输入及参数
char trainDataString[maxTrainUseSize * inputNodeNum * 10];
M hideWeight[threadNum][inputNodeNum];
M hideThresh[threadNum];
M outputWeight[threadNum];
float outputThresh[threadNum];
// 临时变量存放数组
M trainData[maxTrainUseSize][inputNodeNum];
int trainLabel[maxTrainUseSize];
int trainSize;
M *testData;
float *testLabel;
int testSize;
M hideResult[threadNum];
float outputResult[threadNum];
M hideDelta[threadNum];
float outputDelta[threadNum];
// 其他
string trainFile, testFile, ansFile, predictOutFile;
int correctNum[threadNum];

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
    trainFile = train_file;
    testFile = test_file;
    ansFile = ans_file;
    predictOutFile = predict_outfile;
    memset(hideWeight, 0, sizeof(hideWeight));
    memset(hideThresh, 0, sizeof(hideThresh));
    memset(outputWeight, 0, sizeof(outputWeight));
    memset(outputThresh, 0, sizeof(outputThresh));

    memset(correctNum, 0, sizeof(correctNum));
}

void train(int thread){
    /*
    int start = thread*(maxTrainUseSize/threadNum);
    int end = (thread+1)*(maxTrainUseSize/threadNum);
     */
    int start = 0;
    int end = maxTrainUseSize;
    for(int iter=0; iter < maxTrain; iter++){
        correctNum[thread] = 0;
        for(int j=start; j < end; j++){
            netForward(thread, trainData[j], trainLabel[j], true);
            backward(thread, j);
        }
#ifdef TEST
        cout << "目前处于第" << iter << "轮\t\ttrainAucc：" << (float)correctNum[thread] / (float)trainSize
             << "\ttestAucc：" << predict(thread, false) << endl;
        CURTIME(当前时间);
#endif
        if(correctNum[thread] > trainSize * 4 / 5){
            break;
        }
    }
#ifdef TEST
    CURTIME(训练结束);
#endif
}

float predict(int thread, bool isWrite){
#ifdef TEST
    int rightNum = 0, totalNum = 0;
#endif
    int testLabelOffset = -100;
    if(isWrite)
        testLabelOffset = testSize * thread;
    for(int i=0; i < testSize; i++){
        //cout << "当前第" << testCurTurn++ << "轮： " << testP << " " << testSize << endl;
        //cout << endl;
        netForward(thread, testData + i * 1000, 0, false);
#ifdef TEST
        rightNum += ((outputResult[thread] > 0.5 ? 1 : 0) == AnswerVec[i]) ? 1 : 0;
        totalNum++;
#endif
        if(isWrite){
            testLabel[testLabelOffset + i] = outputResult[thread];
        }
    }
#ifdef TEST
    return (float)rightNum / (float)totalNum;
#else
    return 0.0;
#endif
}

void netForward(int thread, const M input[], const int &label, bool isAddNum){
    hideResult[thread] = hideThresh[thread];
    for(int i=0; i < inputNodeNum; i++){
        hideResult[thread] = _mm_add_ps(hideResult[thread], _mm_mul_ps(hideWeight[thread][i], input[i]));
    }
    sigmoid(hideResult[thread], hideResult[thread]);

    M t = _mm_mul_ps(outputWeight[thread], hideResult[thread]);
    outputResult[thread] = sigmoid(outputThresh[thread] + t[0] + t[1] + t[2] + t[3]);
    if(isAddNum)
        correctNum[thread] += (outputResult[thread] > 0.5 ? 1 : 0) == label ? 1 : 0;
}

void backward(int thread, int order){
    outputDelta[thread] = ((float)trainLabel[order] - outputResult[thread]) * sigmoidD(outputResult[thread]);
    hideDelta[thread] = _mm_mul_ps(outputWeight[thread], MEXTEND(outputDelta[thread]));
    M t1;
    sigmoidD(t1, hideResult[thread]);
    hideDelta[thread] = _mm_mul_ps(hideDelta[thread], t1);

    outputWeight[thread] = _mm_add_ps(outputWeight[thread], _mm_mul_ps(MEXTEND(WR * outputDelta[thread]), hideResult[thread]));

    outputThresh[thread] += TR * outputDelta[thread];

    for(int i=0; i < inputNodeNum; i++){
        hideWeight[thread][i] = _mm_add_ps(hideWeight[thread][i], _mm_mul_ps(_mm_mul_ps(MWR, trainData[order][i]), hideDelta[thread]));
    }
    hideThresh[thread] = _mm_add_ps(hideThresh[thread], _mm_mul_ps(MTR, hideDelta[thread]));
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

void *loadTrainData(void *arg){
    CPU_ZERO(cpuSet+0);
    CPU_SET(0, cpuSet+0);
    assert(!pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), cpuSet+0));
    FILE *fp = fopen(trainFile.c_str(), "rb");
    assert(fp);
    fread(trainDataString, 1, sizeof(trainDataString), fp);
    int p = 0;
    for(int i=0; i < maxTrainUseSize; i++){
        //cout << "!!! ";
        for(int j=0; j < inputNodeNum; j++){
            trainData[i][j] = MEXTEND((float)read(p, trainDataString) / 1000);
            //cout << trainData[i][j] << " ";
        }
        trainLabel[i] = trainDataString[++p] - '0';
        p++;
        //cout << "$$$ " << trainLabel[i] << endl;
    }
    trainSize = maxTrainUseSize;
    fclose(fp);
    pthread_exit(NULL);
    return NULL;
}

void *loadTestData(void *arg){
    CPU_ZERO(cpuSet+1);
    CPU_SET(1, cpuSet+1);
    assert(!pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), cpuSet+1));
    static FILE *test_file = fopen(testFile.c_str(), "rb");
    fseek(test_file, 0, SEEK_END);
    long testStrSize = ftell(test_file);
    fseek(test_file, 0, SEEK_SET);

    static char *testString = (char*)malloc(sizeof(char) * testStrSize + 10);
    testData = (M*)malloc(sizeof(M) * testStrSize / 6);
    assert(fread(testString, 1, sizeof(char) * testStrSize, test_file) == testStrSize);
    testSize = 0;
    int testP = 0;

    while(testP < testStrSize - 1000){
        //cout << "当前第" << testCurTurn++ << "轮： " << testP << " " << testSize << endl;
        for(int i=0; i < 1000; i++){
            testData[testSize*1000+i] = MEXTEND((float)read(testP, testString) / 1000);
            //cout << i << " @" << testP << "@ ";
        }
        testSize++;
        //cout << endl;
    }
    testLabel = (float*)malloc(sizeof(float) * threadNum * testSize * 1000);
    pthread_exit(NULL);
}

void storePredictData(){
    ofstream predict_file(predictOutFile);
    /*
    int order = 0;
    for(int i=0; i < threadNum; i++){
        if(correctNum[order] < correctNum[i]){
            order = i;
        }
    }
     */
    for(int i=0; i < testSize; i++){
        ///*
        float o = 0;
        for(int j=0; j < threadNum; j++){
            o += testLabel[j * 1000 + i];
        }
        predict_file << (o / threadNum > 0.5 ? 1 : 0) << "\n";
        //*/
        //predict_file << (testLabel[order * 1000 + i] / threadNum > 0.5 ? 1 : 0) << "\n";
    }
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

void *packageTrainPredict(void *arg){
    int thread = *(int *)arg;
    CPU_ZERO(cpuSet+thread);
    CPU_SET(thread, cpuSet+thread);
    assert(!pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), cpuSet+thread));
    train(thread);
    predict(thread, true);
}

int main(int argc, char *argv[])
{
    //std::ios::sync_with_stdio(false);
    string trainFile = "/data/train_data.txt";
    string testFile = "/data/test_data.txt";
    string predictFile = "/projects/student/result.txt";
    string answerFile = "/projects/student/answer.txt";

#ifdef TEST
    trainFile = "../data/train_data.txt";
    testFile = "../data/test_data.txt";
    predictFile = "../output/result.txt";
    answerFile = "../data/answer.txt";
    loadAnswerData(answerFile, AnswerVec);
    startTime = clock();
    CURTIME(开始);
#endif
    init(trainFile, testFile, answerFile, predictFile);
#ifdef TEST
    CURTIME(开始训练);
#endif
    pthread_create(&loadDataThread[0], nullptr, loadTrainData, nullptr);
    pthread_create(&loadDataThread[1], nullptr, loadTestData, nullptr);
    pthread_join(loadDataThread[0], nullptr);
    pthread_join(loadDataThread[1], nullptr);
#ifdef TEST
    CURTIME(完成数据加载);
    cout << "ready to train model" << endl;
#endif
    /*
    pthread_create(&loadDataThread[0], nullptr, packageTrainPredict, sendNum);
    pthread_create(&loadDataThread[1], nullptr, packageTrainPredict, sendNum+1);
    pthread_create(&loadDataThread[2], nullptr, packageTrainPredict, sendNum+2);
    pthread_create(&loadDataThread[3], nullptr, packageTrainPredict, sendNum+3);
    pthread_join(loadDataThread[0], nullptr);
    pthread_join(loadDataThread[1], nullptr);
    pthread_join(loadDataThread[2], nullptr);
    pthread_join(loadDataThread[3], nullptr);
     */
    packageTrainPredict(sendNum);
    storePredictData();
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
