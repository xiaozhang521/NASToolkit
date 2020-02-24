/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2018, Natural Language Processing Lab, Northestern University.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 /*
  *
  * This is a simple impelementation of the feed-forward network-baesd language
  * model (FNNLM). See more details about FNNLM in
  * "A Neural Probabilistic Language Model" by Bengio et al.
  * Journal of Machine Learning Research 3 (2003) 1137-1155
  *
  * $Created by: ZHANG Yuhao (zhangyuhao@stumail.neu.edu.cn) 2020-01-14
  */

#include <math.h>
#include "NAS.h"
#include "../../tensor/XGlobal.h"
#include "../../tensor/XUtility.h"
#include "../../tensor/XDevice.h"
#include "../../tensor/XList.h"
#include "../../tensor/function/FHeader.h"
#include "../../network/XNet.h"
#include <vector>
#include <cuda_runtime.h>
using namespace nts;
typedef std::vector<int> vec;
typedef std::vector<vec> vec2D;
namespace nas{

float minmax = -0.01;
int batchSize = 50;
int bptt = 35;
int nodeNum = 9;


void convert2Id()
{
    char filePath[100] = "D:/Work/NAS/data/train.txt";
    //char filePath[100] = "D:/Work/NASToolkit/data/test1.txt";
    FILE* fp = fopen(filePath, "r");
    StrList dict;
    XList trainSents;
    if (fp)
    {
        char sentence[10000];
        char stopFlag[6] = "<eos>";
        //while (!feof(fp))
        while (NULL != fgets(sentence, sizeof(sentence), fp))
        {
            //fgets(sentence, 10000, fp);
            if (sentence[0] == ' ' && strlen(sentence) == 1)
                continue;
            int startIndex = 0;
            char word[100];
            while (sentence[startIndex] == ' ')
                startIndex++;
            int length = strlen(sentence);

            //if (sentence[length-1] == '\n')
            //    length--;
            //printf("%s|\n", sentence);
            memcpy(sentence + length - 1, stopFlag, 6);
            length += 5;
            //int wordNum = 0;
            //IntList* sentenceId = (IntList*)malloc(sizeof(IntList));
            IntList* sentenceId = new IntList();
            for (int i = startIndex; i < length; ++i)
            {
                if (sentence[i] == ' ' || i == length - 1)
                {
                    memcpy(word, sentence + startIndex, i - startIndex);
                    word[i - startIndex] = '\0';
                    startIndex = i + 1;
                    int VSize = dict.count;
                    //printf("%s ", word);
                    bool flag = true;
                    for (int j = 0; j < VSize; ++j)
                    {
                        //printf("%s %s", dict.Get(j), word);
                        if (!strcmp(dict.Get(j), word))
                        {
                            sentenceId->Add(j);
                            flag = false;
                            break;
                        }
                    }
                    if (flag)
                    {
                        char* dictWord = new char[strlen(word) + 1];
                        memcpy(dictWord, word, strlen(word) + 1);
                        sentenceId->Add(dict.count);
                        dict.Add(dictWord);
                        //printf("%s\n", dict[0]);
                    }
                }
            }
            trainSents.Add(sentenceId);
        }
        /*char fileOutPath[100] = "D:/Work/NAS/data/trainId.txt";
        FILE* fout = fopen(fileOutPath, "w");
        printf("%d", trainSents.count);
        for (int i = 0; i < trainSents.count; ++i)
        {
            IntList* sentenceId = (IntList*)trainSents.GetItem(i);
            for (int j = 0; j < sentenceId->count; ++j)
            {
                fprintf(fout, " %d", sentenceId->GetItem(j));
            }
            fputc('\n', fout);
        }*/
        //printf("%c|\n", sentence[0]);
        char fileOutPath[100] = "D:/Work/NAS/data/dict.txt";
        FILE* fout = fopen(fileOutPath, "w");
        for (int i = 0; i < dict.count; ++i)
        {
            char* string = dict.GetItem(i);
                fprintf(fout, "%s", string);
            fputc('\n', fout);
        }
        fclose(fp);
        fclose(fout);
    }
    else
    {
        printf("file open wrong \n");
    }
}

void readDict(StrList& dict)
{
    char fileDictPath[100] = "D:/Work/NAS/data/dict.txt";
    FILE* fp = fopen(fileDictPath, "r");
    char word[100];
    while (NULL != fgets(word, sizeof(word), fp))
    {
        char* dictWord = new char[strlen(word) + 1];
        memcpy(dictWord, word, strlen(word) + 1);
        dict.Add(dictWord);
    }
    fclose(fp);

}
 
void Init(RNNSearchModel& model)
{
    /* create embedding parameter matrix: vSize * eSize */
    InitTensor2D(&model.embeddingW, model.vSize, model.eSize, X_FLOAT, model.devID);

    /* create the output layer parameter matrix and bias term */
    InitTensor2D(&model.outputW, model.hSize, model.vSize, X_FLOAT, model.devID);
    InitTensor1D(&model.outputB, model.vSize, X_FLOAT, model.devID);
    model.outputW.SetVarFlag();
    model.outputB.SetVarFlag();

    /* then, we initialize model parameters using a uniform distribution in range
       of [-minmax, minmax] */
    model.embeddingW.SetDataRand(-minmax, minmax);
    model.outputW.SetDataRand(-minmax, minmax);

    /* all terms are set to zero */
    model.outputB.SetZeroAll();
}

void MakeTrainBatch(XTensor trainData, int index, int seqLength,XTensor &data, XTensor &targets)
{
    int minLen = min(seqLength, trainData.dimSize[0] - 1 - index);
    data = SelectRange(trainData, 0, index, index + minLen);
    targets = SelectRange(trainData, 0, index + 1, index + 1 + minLen);
}

void Forward(XTensor input,XTensor &output, RNNSearchModel &model)
{
    XTensor transInput;
    XTensor embeddings;
    XTensor rnnOut;
    XTensor fnnOut;
    transInput = Transpose(input, 0, 1);
    embeddings = Gather(model.embeddingW, transInput);
    RNNForword(embeddings, rnnOut, model.rnn);
    fnnOut = MMul(rnnOut, model.outputW) + model.outputB;
    output = LogSoftmax(fnnOut, 2);
}

void Train(XTensor trainData, RNNSearchModel& model)
{
    XNet autoDiffer;
    for (int i = 0; i < trainData.dimSize[0] - 1 - 1; ++i)
    {
        XTensor batchTrain;
        XTensor batchTarget;
        XTensor output;
        XTensor lossTensor;
        XTensor gold;
        MakeTrainBatch(trainData, i, model.bpttLength, batchTrain, batchTarget);
        gold = IndexToOnehot(batchTarget, model.vSize, 0);
        Forward(batchTrain, output, model);
        lossTensor = CrossEntropy(output, gold);
        autoDiffer.Backward(lossTensor);
        if ((i + 1) % 100 == 0)
            printf("%d / %d\n", i + 1, trainData.dimSize[0] - 1);
    }
}

int NASMain(int argc, const char** argv)
{
    //convert2Id();
    StrList dict;
    readDict(dict);

    //printf("%d\n", dict.count);
    RNNSearchModel model;
    model.vSize = dict.count;
    model.eSize = 32;
    model.hSize = 32;
    model.devID = 0;
    model.bpttLength = bptt;
    model.rnn.hiddenSize = model.hSize;
    model.rnn.inputSize = model.eSize;
    model.rnn.devID = model.devID;
    model.rnn.nodeNum = nodeNum;
    Init(model);
    InitCell(model.rnn);
    char filePath[100] = "D:/Work/NAS/data/trainId.txt";
    FILE* fp2 = fopen(filePath, "r");
    char sentence[10000];
    vec vecData;
    while (NULL != fgets(sentence, sizeof(sentence), fp2))
    {
        int length = strlen(sentence) - 1;
        int startIndex = 0;
        while (sentence[startIndex] == ' ')
            startIndex++;
        int wordId = 0;
        //vec *tmp = new vec();
        for (int i = startIndex; i < length ; ++i)
        {
            if (sentence[i] == ' ' || i == length - 1)
            {
                if (i == length - 1)
                {
                    wordId = (sentence[i] - '0') + 10 * wordId;
                }
                vecData.push_back(wordId);
                wordId = 0;
            }
            else
            {
                wordId = (sentence[i] - '0') + 10 * wordId;
            }
        }
        /*for (vec::iterator it2 = tmp->begin(); it2 != tmp->end(); ++it2)
        {
            printf("%d ", *it2);
        }
        printf("\n");*/
        //trainData.push_back(*tmp);
    }
    //for (vec2D::iterator it1 = trainData.begin(); it1 != trainData.end(); ++it1)
    /*vec2D::iterator it1 = trainData.end(); --it1;
    {
        vec tmp = *it1;
        for (vec::iterator it2 = tmp.begin(); it2 != tmp.end(); ++it2)
        {
            printf("%d ", *it2);
        }
        printf("\n");
    }*/
    XTensor trainData;
    InitTensor1D(&trainData, vecData.size(), X_INT, model.devID);
    trainData.SetData(&vecData[0], trainData.unitNum);
    int nbatch = trainData.unitNum / batchSize;
    /* not very clear to do this */
    trainData = SelectRange(trainData, 0, 0, nbatch * batchSize);
    trainData.Reshape(batchSize, nbatch);
    trainData = Transpose(trainData, 0, 1);
    Train(trainData, model);

    return 0;
}

}