#include"DARTSCell.h"

using namespace nts;
namespace nas{

void show(XTensor a)
{
    if (a.order <= 0)
    {
        printf("Tensor order is blow 0\n");
    }
    for (int i = 0; i < a.order; ++i)
    {
        printf("%d ", a.dimSize[i]);
    }
    printf("\n");
}

void InitCell(DARTSCell &cell)
{
    InitTensor2D(&(cell.W0), cell.hiddenSize + cell.inputSize, 2 * cell.hiddenSize, X_FLOAT, cell.devID);
    InitTensor2D(&(cell.alphaWeight), cell.nodeNum, 4, X_FLOAT, cell.devID);
    //show(cell.W0);
    cell.WList.Clear();
    for (int i = 0; i < cell.nodeNum - 1; ++i)
    {
        /* This maybe leads to memory leak */
        XTensor *W = NewTensor2D(cell.hiddenSize, 2 * cell.hiddenSize, X_FLOAT, cell.devID);
        //InitTensor2D(&W, cell.hiddenSize, 2 * cell.hiddenSize, X_FLOAT, cell.devID);
        cell.WList.Add(W);
        
    }
}

XTensor (*getActivation(char name[]))(const XTensor &)
{
    XTensor(*funcPointer)(const XTensor &) = NULL;
    if (!strcmp(name, "tanh"))
    {
        funcPointer = HardTanH;
    }
    else if (!strcmp(name,"relu"))
    {
        funcPointer = Rectify;
    }
    else if (!strcmp(name, "sigmoid"))
    {
        funcPointer = Sigmoid;
    }
    else if (!strcmp(name, "identity"))
    {
        funcPointer = Identity;
    }
    else
    {
        printf("Don't support active function %s!\n", name);
    }
    return funcPointer;
}

void Cell(XTensor input,XTensor hidden, XTensor &newHidden, DARTSCell &rnnCell)
{
    //XTensor newHidden(hidden.order, hidden.dimSize, hidden.dataType, 1.0F, hidden.devID, hidden.mem);

    XTensor xhConcat;
    XTensor ch0;
    XTensor c0;
    XTensor h0;
    XTensor *s0 = NewTensor2D(hidden.dimSize[0], hidden.dimSize[1], X_FLOAT, rnnCell.devID);

    /* compute init state*/
    xhConcat = Concatenate(input, hidden, 1);
    ch0 = Split(MMul(xhConcat, rnnCell.W0), 1, 2);
    c0 = SelectRange(ch0, 0, 0, 1);
    h0 = SelectRange(ch0, 0, 1, 2);
    SqueezeMe(c0);
    SqueezeMe(h0);
    /* some question */
    *s0 = Sigmoid(c0) * (HardTanH(h0) - hidden);
    *s0 = *s0 + hidden;
    
    /* hidden sate of each node */
    TensorList stateList;
    stateList.Add(s0);
    XTensor preState;
    XTensor ch;
    XTensor c;
    XTensor h;
    XTensor states;
    for (int i = 1; i <  rnnCell.nodeNum; ++i)
    {
        for (int preIndex = 0; preIndex < i; ++preIndex)
        {
            preState = *stateList[preIndex];
            ch = Split(MMul(preState, *rnnCell.WList[preIndex]), 1, 2);
            c = SelectRange(ch, 0, 0, 1);
            h = SelectRange(ch, 0, 1, 2);
            SqueezeMe(c);
            SqueezeMe(h);
            XTensor *s = NewTensor2D(preState.dimSize[0], preState.dimSize[1], X_FLOAT, rnnCell.devID);

            /* To be clear */
            TensorList activeList;
            char funName[4][10] = { "tanh","relu","sigmoid","identity" };
            XTensor alpha = Softmax(rnnCell.alphaWeight, 1);
            for (int activeIndex = 0; activeIndex < 4; ++activeIndex)
            {
                XTensor *activeH = NewTensor2D(h.dimSize[0], h.dimSize[1], X_FLOAT, rnnCell.devID);
                XTensor(*activeFunc)(const XTensor &) = getActivation(funName[activeIndex]);
                float alpha = rnnCell.alphaWeight.Get2D(preIndex, activeIndex);
                *activeH = activeFunc(h);
                *activeH = *activeH * alpha;
                activeList.Add(activeH);
                h = ReduceMean(Stack(activeList, 1), 1);
            }
            for (int activeIndex = 0; activeIndex < activeList.count; ++activeIndex)
                delete activeList[activeIndex];
            *s = Sigmoid(c) * (h - preState);
            *s = *s + preState;
            stateList.Add(s);
        }
    }
    states = Stack(stateList, 2);
    /* free memory */
    for (int i = 0; i < stateList.count; ++i)
        delete stateList[i];
    newHidden = ReduceMean(states, 2);

}

void RNNForword(XTensor input, XTensor &output, DARTSCell &rnn)
{
    CheckErrors(input.order==3, "input tensor shape must be [batchSize, sencentLength, embeddingSize]!");
    int batchSize = input.dimSize[0];
    int senLength = input.dimSize[1];
    TensorList hiddenList;
    XTensor hidden;
    InitTensor2D(&hidden, batchSize, rnn.hiddenSize, input.dataType, input.devID, false);
    hidden.SetZeroAll();
    //XTensor newHidden[35];
    for (int index = 0; index < senLength; ++index)
    {
        XTensor inputSlice;
        XTensor *newHidden = NewTensor2D(batchSize, rnn.hiddenSize, X_FLOAT, rnn.devID);
        inputSlice = SelectRange(input, 1, index, index + 1);
        SqueezeMe(inputSlice, 1);
        Cell(inputSlice, hidden, *newHidden, rnn);
        hiddenList.Add(newHidden);
        hidden = *newHidden;
        //hiddenList[0]->Dump(stderr);
    }
    /* question here */
    output = Stack(hiddenList, 0);
    for (int i = 0; i < hiddenList.count; ++i)
        delete hiddenList[i];
}

};
