#include"DARTSCell.h"

using namespace nts;
namespace nas{

void show(XTensor a)
{
    for (int i = 0; i < a.order; ++i)
    {
        printf("%d ", a.dimSize[i]);
    }
    printf("\n");
}

void Cell(XTensor input,XTensor hidden, XTensor newHidden, DARTSCell rnnCell)
{
    //XTensor newHidden(hidden.order, hidden.dimSize, hidden.dataType, 1.0F, hidden.devID, hidden.mem);
    XTensor initState;
    initState = Concatenate(input, hidden, 1);
    
    MMul();
    /*
    c0, h0 = torch.split(xh_prev.mm(self._W0), self.nhid, dim=-1)
    c0 = c0.sigmoid()
    h0 = h0.tanh()
    s0 = h_prev + c0 * (h0-h_prev)
    return s0
    */

    show(initState);
    printf("run here\n");
}

void RNNForword(XTensor input, XTensor &output, DARTSCell rnn)
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
    output = Stack(hiddenList, 0);
}

};

