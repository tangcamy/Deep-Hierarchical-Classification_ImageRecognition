# Deep Level
This branch is Level_3.

# Deep Hierarchical Classification
This is a non-official implementation of [Deep Hierarchical Classification for Category Prediction in E-commerce System][1]. 
The github form : [Github - Ugenteraan Manogaran][2]

## setting 
- 下載測試集:process_cifar100.py 
- 路徑確認設定:runtime_args.py

## Training 
- 使用DeepHC.py 進行訓練 ; 原本(作者train.py)會在執行from plot import plot_loss_acc函式出錯
- plot.py 修改成DeepHC.py的型態，但包成函式執行還是會出錯 ; 原本作者(plot_oldversion.py )

## Cutsomer Data Training 
1. dataPickle_Transform.py : 原本資料建立meta資料表 ＆ train/test 資料集資訊。
    - section-1 < meta >: coarse_label_names,fine_label_names.
    - setction-2 train / test >  create : filenames , fine_labels , coarse_labels.
    - section-3 < read meta dic for data transform index > & < train / test >  create : filenames , fine_labels , coarse_labels.
    - [資料夾說明]
        1. preimages/train,preimage/test 的照片要放在 dataset/images。
        2. meta,train,test 要放在 dataset/pickle_files。
        3. 會產生一個pickleTotal.csv彙整表。
2. process_dataset.py : 建立train.csv /  test.csv。
3. level_dic.py : 建立 level_1 與 Level_2 階層字典。
4. runtime_args.py : 訓練前確認路徑＆相關參數設定。
5. model/resnet50.py ：num_classes類別數量需要修改。(coarse_label_names個數 & fine_label_names個數)
6. resize.py : 照片 resize 。
7. DeepHC.py ：模型訓練。


[1]: https://arxiv.org/pdf/2005.06692.pdf "Deep Hierarchical Classification for Category Prediction in E-commerce System"
[2]:https://github.com/Ugenteraan/Deep_Hierarchical_Classification "Github - Ugenteraan Manogaran"