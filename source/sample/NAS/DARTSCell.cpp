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

void Cell(XTensor input,XTensor hidden, XTensor &newHidden, DARTSCell &rnnCell)
{
    //XTensor newHidden(hidden.order, hidden.dimSize, hidden.dataType, 1.0F, hidden.devID, hidden.mem);

    XTensor xhConcat;
    XTensor ch0;
    XTensor c0;
    XTensor h0;
    XTensor s0;

    /* compute init state*/
    xhConcat = Concatenate(input, hidden, 1);
    ch0 = Split(MMul(xhConcat, rnnCell.W0), 1, 2);
    c0 = SelectRange(ch0, 0, 0, 1);
    h0 = SelectRange(ch0, 0, 1, 2);
    SqueezeMe(c0);
    SqueezeMe(h0);
    /* some question */
    s0 = Sigmoid(c0) * (HardTanH(h0) - hidden);
    s0 = s0 + hidden;
    
    /* hidden sate of echo node */
    TensorList stateList;
    stateList.Add(&s0);
    XTensor preState;
    XTensor ch;
    XTensor c;
    XTensor h;
    XTensor s[9];
    XTensor states;
    for (int i = 0; i <  rnnCell.nodeNum - 1; ++i)
    {
        preState = *stateList[i];
        ch = Split(MMul(preState, *rnnCell.WList[i]), 1, 2);
        c = SelectRange(ch, 0, 0, 1);
        h = SelectRange(ch, 0, 1, 2);
        SqueezeMe(c);
        SqueezeMe(h);
        /* some question */
        s[i] = Sigmoid(c) * (HardTanH(h) - preState);
        s[i] = s[i] + preState;
        stateList.Add(&s[i]);
    }

    states = Stack(stateList, 2);
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
    for (int index = 0; index < senLength; ++index)
    {
        XTensor inputSlice;
        XTensor newHidden;
        inputSlice = SelectRange(input, 1, index, index + 1);
        SqueezeMe(inputSlice, 1);
        Cell(inputSlice, hidden, newHidden, rnn);
        hiddenList.Add(&newHidden);
        hidden = newHidden;
        //hiddenList[0]->Dump(stderr);
    }
    /*for (int i = 0; i < hiddenList.count; ++i)
    {
        show(hiddenList[i]);
    }*/
    /* question here */
    output = Stack(hiddenList, 2);
}

};

