
#ifndef __DARTSCELL_H__
#define __DARTSCELL_H__
#include "../../tensor/XGlobal.h"
#include "../../tensor/XTensor.h"
#include "../../tensor/core/CHeader.h"
#include "../../tensor/function/FHeader.h"
#include "../../tensor/XList.h"
using namespace nts;

namespace nas
{
#define _EXIT_(x)// exit(x)
#define CheckErrors(x, msg) { if(!(x)) { fprintf(stderr, "Error! calling '%s' (%s line %d): %s\n", #x, __FILENAME__, __LINE__, msg);  _EXIT_(1); } }
#define ShowErrors(msg) { { fprintf(stderr, "Error! (%s line %d): %s\n", __FILENAME__, __LINE__, msg); _EXIT_(1); } } 

struct NodeType
{
    char funcName[20];
    int preIndex;
};

struct DARTSCell
{
    int hiddenSize;
    int inputSize;
    int nodeNum;
    int devID;
    XTensor W0;
    TensorList WList;
    XTensor alphaWeight;
    DARTSCell() {};
    ~DARTSCell() {
        /* free memory */
        for (int i = 0; i < WList.count; ++i)
            delete WList[i];
    };
};
void InitCell(DARTSCell &cell);
void RNNForword(XTensor input, XTensor &output, DARTSCell &rnn);
void show(XTensor a);

};
#endif
