import torch


def get_recall_precision(confusion_matrix):
    '''
    정답이 1개인 예측의 confusion matrix
    '''
    num_class = len(confusion_matrix)
    eps = 1e-3
    precision = 0
    recall = 0
    for row in range(num_class):
        tp = confusion_matrix[row,row].item()
        fp = torch.sum(confusion_matrix[row]).item() - tp
        fn = torch.sum(confusion_matrix[:,row]).item() - tp
        precision += (1/num_class)*(tp/(tp+fp+eps))
        recall += (1/num_class)*(tp/(tp+fn+eps))
    return recall, precision


if __name__ == '__main__':
    con = torch.Tensor([[62., 26., 11., 10., 31.,  1., 53.,  3.],
                        [ 3.,  4.,  8.,  0.,  3.,  0.,  2.,  6.],
                        [ 0.,  2., 12.,  4.,  1.,  2.,  1., 15.],
                        [ 4.,  9., 11., 49., 14., 22.,  7.,  4.],
                        [ 2., 13.,  1.,  5.,  5.,  4.,  5.,  5.],
                        [ 3., 31., 29.,  9., 29., 27.,  9., 51.],
                        [ 7.,  2.,  6.,  0.,  1.,  0., 12.,  4.],
                        [ 1., 12., 21., 23.,  5., 44., 11., 11.]])
    r, p = get_recall_precision(con)
    print(r, p)