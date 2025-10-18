import torch
import torch.nn.functional as F

from core import evaluation

def test_acc(net, criterion, testloader, epoch=None, **options):
    net.eval()
    correct, total = 0, 0

    torch.cuda.empty_cache()

    with torch.no_grad():
        # Known class prediction
        for data, labels in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()

            x, y = net(data, True)
            logits, _ = criterion(x, y)
            predictions = logits.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

    # Accuracy
    acc = float(correct) * 100. / float(total)
    # print('Acc: {:.5f}'.format(acc))
    return acc



    return acc
