# ai-wen-mang-guo-bu-liang-pin-fen-lei-jing-sai-109318083
ai-wen-mang-guo-bu-liang-pin-fen-lei-jing-sai-109318083 created by GitHub Classroom


<h2>1.做法說明</h2>
 
* 訓練前先進行資料處理 : baseline01.py
* 確認座標軸 合併資料並建立model:trainalex.ipynb
* svm 轉換為 nn :nn03.ipynb

<h2>2.程式方塊圖與寫法</h2>

![image](https://github.com/MachineLearningNTUT/regression-109318083/blob/main/Diagram.jpg)

<h2>3.畫圖做結果分析</h2>
* 於nn03.ipynb
  
  
  
    def transform(listdir,label,lenSIZE):
<h2>4.討論預測值誤差很大的，是怎麼回事？</h2>
* 可能魔星還需要再進行調整    
    def transform(listdir,label,lenSIZE):
   

<h2>5.如何改進？</h2>

    如果private 預測下降表示模型可再繼續疊加或者改變參數權重
