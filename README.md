# Deep Hierarchical Classification

This is a non-official implementation of [Deep Hierarchical Classification for Category Prediction in E-commerce System][1]. 
The github form : [Github - Ugenteraan Manogaran][2]

## setting 
- 下載測試集:process_cifar100.py 
- 路徑確認設定:runtime_args.py

## Training 
- 使用DeepHC.py 進行訓練 ; 原本(作者train.py)會在執行from plot import plot_loss_acc函式出錯
- plot.py 修改成DeepHC.py的型態，但包成函式執行還是會出錯 ; 原本作者(plot_oldversion.py )

[1]: https://arxiv.org/pdf/2005.06692.pdf "Deep Hierarchical Classification for Category Prediction in E-commerce System"
[2]:https://github.com/Ugenteraan/Deep_Hierarchical_Classification "Github - Ugenteraan Manogaran"