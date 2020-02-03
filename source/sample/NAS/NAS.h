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
  * Journal of Machine Learning Research 3 (2003) 1137¨C1155
  *
  * $Created by: ZHANG Yuhao (zhangyuhao@stumail.neu.edu.cn) 2020-01-14
  * Today I was awarded as the most popular teacher in our college.
  * It was the great honour for me!!!
  */

#ifndef __NAS_H__
#define __NAS_H__

#include "../../tensor/XGlobal.h"
#include "../../tensor/XTensor.h"
#include "../../tensor/core/CHeader.h"

using namespace nts;

namespace nas
{

#define _EXIT_(x)// exit(x)
#define CheckErrors(x, msg) { if(!(x)) { fprintf(stderr, "Error! calling '%s' (%s line %d): %s\n", #x, __FILENAME__, __LINE__, msg);  _EXIT_(1); } }
#define ShowErrors(msg) { { fprintf(stderr, "Error! (%s line %d): %s\n", __FILENAME__, __LINE__, msg); _EXIT_(1); } } 

#define MAX_N_GRAM 8
#define MAX_HIDDEN_NUM 8

    struct DARTSCell
    {
        DARTSCell() {};
        ~DARTSCell() {};
    };


    /* an n-gram = a sequence of n words
       words[0..n-2] is the history, and
       words[n-1] is the word for prediction. */

    /* rnn search model */
    struct RNNSearchModel
    {
        /* word embedding */
        XTensor embeddingW;

        /* parameter matrix of the output layer */
        XTensor outputW;

        DARTSCell rnn;

        /* bias of the output layer */
        XTensor outputB;

        /* order of the language model */
        int n;

        /* embedding size */
        int eSize;

        /* hidden layer size */
        int hSize;

        /* vocabulary size */
        int vSize;

        /* id of the device for running the model */
        int devID;

        int bpttLength;

        /* indicates whether we use memory pool */
        bool useMemPool;

        /* memory pool */
        XMem* mem;

        RNNSearchModel() { n = -1; vSize = -1; devID = -1; mem = NULL; };
        ~RNNSearchModel() { delete mem; };
    };

 
    ///* the network built on the fly */
    //struct RNNSearchNET
    //{
    //    XTensor transInput;

    //    /* embedding result of the previous n - 1 words */
    //    XTensor embeddings[MAX_N_GRAM];

    //    /* concatenation of embeddings */
    //    XTensor embeddingCat;

    //    /* output of the hidden layers */
    //    XTensor hiddens[MAX_HIDDEN_NUM];

    //    /* state of the hidden layers (before activation function) */
    //    XTensor hiddenStates[MAX_HIDDEN_NUM];

    //    /* state before softmax */
    //    XTensor stateLast;

    //    /* output of the net */
    //    XTensor output;
    //};

    /* entrance of the program */
    int NASMain(int argc, const char** argv);

};

#endif
