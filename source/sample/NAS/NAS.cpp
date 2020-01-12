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
  * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-06-22
  */

#include <math.h>
#include "NAS.h"
#include "../../tensor/XGlobal.h"
#include "../../tensor/XUtility.h"
#include "../../tensor/XDevice.h"
#include "../../tensor/XList.h"
#include "../../tensor/function/FHeader.h"
#include "../../network/XNet.h"

using namespace nts;
namespace nas{

void convert2Id()
{

}

int NASMain(int argc, const char** argv)
{
    char filePath[100] = "D:/Work/NASToolkit/data/train.txt";
    //char filePath[100] = "D:/Work/NASToolkit/data/test1.txt";
    FILE* fp = fopen(filePath, "r");
    StrList dict;
    XList trainSents;
    if (fp)
    {
        char sentence[10000];
        char stopFlag[6] = "<eos>"; 
        //while (!feof(fp))
        while(NULL != fgets(sentence, sizeof(sentence), fp))
        {
            //fgets(sentence, 10000, fp);
            if (sentence[0]==' '&&strlen(sentence) == 1)
            //    printf("runhere");
            //printf("%d", strlen(sentence));
                continue;
            int startIndex = 0;
            char word[100];
            while(sentence[startIndex] == ' ')
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
                if (sentence[i] == ' ' || i ==  length - 1)
                {
                    memcpy( word, sentence + startIndex, i - startIndex);
                    word[i - startIndex] = '\0';
                    startIndex = i + 1;
                    int VSize = dict.count;
                    //printf("%s ", word);
                    bool flag = true;
                    for (int j = 0; j < VSize; ++j)
                    {
                        //printf("%s %s", dict.Get(j), word);
                        if (!strcmp(dict.Get(j),word))
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
                        dict.Add(dictWord);
                        sentenceId->Add(dict.count);
                        //printf("%s\n", dict[0]);
                    }
                }
            }
            trainSents.Add(sentenceId);
        }
        /*char fileOutPath[100] = "D:/Work/NASToolkit/data/trainId.txt";
        FILE* fout = fopen(fileOutPath, "w");
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
        char fileOutPath[100] = "D:/Work/NASToolkit/data/dict.txt";
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
    
    return 0;
}

}