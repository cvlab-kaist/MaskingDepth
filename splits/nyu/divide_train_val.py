import random

PER = 0.15

with open('train_unsup_nyu.txt','r') as f:
    train_set = f.readlines()
    val_set_num = int(len(train_set) * PER)
    val_set = []
    
    for i in range(val_set_num):
        index = random.randint(0, (len(train_set)-1))
        val_set.append(train_set.pop(index))

    train_f = open('train.txt','w')
    val_f = open('val.txt','w')

    for i in range(len(train_set)):
        train_str = train_set[i].split(',')[0][:-4] + '\n'
        train_f.write(train_str) 
    for i in range(len(val_set)):
        val_str = train_set[i].split(',')[0][:-4] + '\n'
        val_f.write(val_str)

    train_f.close()
    val_f.close()