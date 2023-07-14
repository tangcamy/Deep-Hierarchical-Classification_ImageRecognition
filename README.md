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
    - setction-2 train / test >  create : filenames , coarse_labels, fine_labels ,third_labels
    - section-3 < read meta dic for data transform index > & < train / test >  create : filenames , coarse_labels, fine_labels ,third_labels
    - [資料夾說明]
        1. dataPickle_Transform/preimages/train , dataPickle_Transform/preimage/test 的照片要放在 dataset/images。
        2. dataPickle_Transform/picklefiles/meta,train,test 要放在 dataset/pickle_files。
        3. 會產生一個pickleTotal.csv彙整表。
2. process_dataset.py : 建立train.csv /  test.csv。
3. level_dic.py : 建立 level_1 與 Level_2 與 level_3 階層字典。
4. runtime_args.py : 訓練前確認路徑＆相關參數設定。
5. resize.py : 照片 resize 。
    - Traintype = True
6. model/resnet50.py ：model 結構原本兩層改成三層。
    - num_classes類別數量需要修改。(coarse_label_names個數 & fine_label_names個數 & third_label_names個數)。
    - 需要新增區域：linear_lvl3 , softmax_reg3 , level_3。
7. model/hierarchical_loss.py :loss 結構原本兩層改成三層。
    - 新增self.level_third_labels，hierarchical_two_label，numeric_hierarchy_two。
    - calculate_dloss調整D_l的check_hierarchy。
8. Load_dataset.py:
    - from level_dict import hierarchy,hierarchy_two，新增hierarchy_two。
    - read_meta新增output : self.third_labels。
    - check if the hierarchy_two dictionary
    - add subtwoclass。
9. DeepHC.py ：模型訓練結構兩層改成三層。
    - from level_dict import hierarchy,hierarchy_two，新增hierarchy_two。
    - 調整HierarchicalLossNetwork：hierarchical_labels_one，hierarchical_labels_two，total_level=3。
    - 新增batch_y3:sample['label_3'].to(device)。
    - 新增subtwoclass_pred: model-layer3 output。
    - 新增accuracy_subtwoclass_epoch.png。


## Data Inference
1. dataPickle_detect.py : 資料丟入之前先轉換成detect(pickle),照片名稱預先處理（之後這部份可以考慮不用）。
    - save detect pickle 轉換在：dataPickle_Transform/pickle_files/detect
    - save detect image 轉換在： dataPickle_Transform/preimages/detect;
2. process_detect.py:需先將必要的資料父至於對應資料夾中，會產生出一個detect.csv。
    - detect pickle 複製:(dataPickle_Transform/pickle_files/) 複製貼到 (data/pickle_files/)。
    - image 複製::(dataPickle_Transform/preimages/) 複製貼到 (data/detect_imgs/)。
    - detect.csv : 程式產生儲存在 dataset/。
3. resize.py : data/detect_imgs 照片記得resize。
    - Traintype = False
4. detect.py :  Inference預測，結果dataset/result/detect_predict.csv。

## .py檔案稍微修改
### Training-Part
1. load_dataset.py : 原本retrun_labe（image,label_1,label_2) 新增一個image_path
2. helper.py:
    - read_meta函示：多一個輸出third_label_names。
3. level_dict.py:多一個字典。
    - hierarchy:layer_1 & layer_2
    - hierarchy_2:layer_2 & layer_3
4. resnet50.py:模型架構修改Resnet50 或 Resnet101.
    - self.num_blocks的部份調整。
### Inference-Part
1. dataPickle_detect.py：新增third_label_names

[1]: https://arxiv.org/pdf/2005.06692.pdf "Deep Hierarchical Classification for Category Prediction in E-commerce System"
[2]:https://github.com/Ugenteraan/Deep_Hierarchical_Classification "Github - Ugenteraan Manogaran"