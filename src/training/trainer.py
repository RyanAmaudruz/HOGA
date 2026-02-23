import torch
import torch.nn.functional as F

from src.utils.preprocess import post_processing


def train(model, train_loader, optimizer, device, args):
    model.train()
    total_loss = 0
    for i, (x, y, r_y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        r_y = r_y.to(device)

        optimizer.zero_grad()
        out1, out2, out3, attn = model(x)

        y1 = y.squeeze(1).clone().detach()
        for i in range(y1.size()[-1]):
            if y1[i] == 0 or y1[i] == 5:
                y1[i] = 1
            if y1[i] == 2:
                y1[i] = 4
            if y1[i] > 2:
                y1[i] = y1[i] - 1
            y1[i] = y1[i] - 1

        y2 = y.squeeze(1).clone().detach()
        for i in range(y2.size()[-1]):
            if y2[i] > 2:
                y2[i] = y2[i] - 1
            if y2[i] == 0 or y2[i] == 4:
                y2[i] = 1
            y2[i] = y2[i] - 1

        y3 = r_y.squeeze(1).clone().detach()
        for i in range(y3.size()[-1]):
            if y3[i] == 0 or y3[i] == 4:
                y3[i] = 3
            y3[i] = y3[i] - 1

        loss = (
            F.cross_entropy(out1, y1)
            + args.lda1 * F.cross_entropy(out2, y2)
            + args.lda2 * F.cross_entropy(out3, y3)
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    loss = total_loss / len(train_loader)
    return loss


@torch.no_grad()
def test_all(model, test_loader, device, file_name=None):
    model.eval()

    test_acc_r = 0
    test_acc_s = 0
    all_y_shared = []
    all_pred_shared = []
    all_test_attn = []
    for i, (x, y, r_y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        r_y = r_y.to(device)

        out1, out2, out3, test_attn = model(x)
        y_pred_shared = post_processing(out1, out2).to(device)
        y_pred_root = out3.argmax(dim=-1, keepdim=True)

        y_shared = y.squeeze(1).clone().detach()
        y_root = r_y.squeeze(1).clone().detach()
        s5 = (y_shared == 5).nonzero(as_tuple=True)[0]
        s0 = (y_shared == 0).nonzero(as_tuple=True)[0]
        y_shared[s5] = 1
        y_shared[s0] = 1
        r0 = (y_root == 0).nonzero(as_tuple=True)[0]
        r4 = (y_root == 4).nonzero(as_tuple=True)[0]
        y_root[r0] = 3
        y_root[r4] = 3
        y_root = y_root - 1
        y_root = torch.reshape(y_root, (y_root.shape[0], 1))
        y_shared = torch.reshape(y_shared, (y_shared.shape[0], 1))

        test_acc_r += y_pred_root.eq(y_root).double().sum()
        test_acc_s += y_pred_shared.eq(y_shared).double().sum()

        all_y_shared.append(y_shared.detach().cpu().squeeze().numpy())
        all_pred_shared.append(y_pred_shared.detach().cpu().squeeze().numpy())
        all_test_attn.append(test_attn.detach().cpu().squeeze().numpy())

    test_acc_r /= len(test_loader.dataset)
    test_acc_s /= len(test_loader.dataset)

    return 0, 0, test_acc_r, 0, 0, test_acc_s
