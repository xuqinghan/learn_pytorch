import torch.optim as optim
import torch
from torch import nn

from models import Net
from prepare_data import loader_tr, loader_ts

from time import process_time

#gpu or cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model
model = Net(28*28, 300, 100, 10)
model.to(device)

#criterion 用来产生每批的 fn_loss
criterion = nn.CrossEntropyLoss()

lr = 0.01
momentum = 0.9
#optimizer
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)


losses_tr = []
acces_tr = []
losses_ts = []
acces_ts = []



#------train---------------
num_epoches = 20
for epoch in range(num_epoches):
    print(f'epoch {epoch}, training....')
    loss_tr = 0
    acc_tr = 0
    if epoch %5 == 0:
        #每隔5轮
        optimizer.param_groups[0]['lr'] *=0.9

    #改成训练模式
    model.train()
    start = process_time()
    for img, label in loader_tr:
        #loader每次产生1批数据
        #print(img.shape)
        #print(label.shape)
        #img.shape torch.Size([64, 1, 28, 28])
        # label.shape torch.Size([64])
        #x y 送入 gpu
        img = img.to(device)
        label = label.to(device)

        print('img', img.shape)
        #展平
        img = img.view(img.size(0), -1)
        print('img', img.shape)
        #forward
        out = model(img)
        #这个才是损失函数
        print('out', out.shape)
        loss = criterion(out, label)

        #print('out', out.shape)
        #out torch.Size([64, 10])
        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #记录误差 一个数
        loss_tr += loss.item()
        _,  pred = out.max(1)
        #print(pred) #[64]
        num_correct = (pred==label).sum().item()
        #img.shape[0] 是每批数量
        acc = num_correct / img.shape[0]
        acc_tr += acc

    #累计当前批次的损失
    loss_tr_epoch = loss_tr/len(loader_tr)
    acc_tr_epoch = acc_tr/len(loader_tr)
    print(f'train done. {process_time()- start} sec, loss={loss_tr_epoch:.4f}, acc_tr={acc_tr_epoch:.4f}')
    losses_tr.append(loss_tr_epoch)
    acces_tr.append(acc_tr_epoch)

    loss_ts = 0
    acc_ts = 0
    #改成eval模式
    model.eval()
    for img, label in loader_ts:
        #x y 送入 gpu
        img = img.to(device)
        label = label.to(device)
        #展平
        img = img.view(img.size(0), -1)
        #forward
        out = model(img)
        #这个才是损失函数
        fn_loss = criterion(out, label)
        #记录误差 一个数
        loss_ts += fn_loss.item()
        _,  pred = out.max(1)
        num_correct = (pred==label).sum().item()
        acc = num_correct / img.shape[0]
        acc_ts += acc

    #累计当前批次的损失
    loss_ts_epoch = loss_ts/len(loader_ts)
    acc_ts_epoch = acc_ts/len(loader_ts)
    print(f'test sec, loss={loss_ts_epoch:.4f}, acc_tr={acc_ts_epoch:.4f}')
    losses_ts.append(loss_ts_epoch)
    acces_ts.append(acc_ts_epoch)




#保存模型
#torch.save(model, 'model.pkl')

