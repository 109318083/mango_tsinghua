# ai-wen-mang-guo-bu-liang-pin-fen-lei-jing-sai-109318083
ai-wen-mang-guo-bu-liang-pin-fen-lei-jing-sai-109318083 created by GitHub Classroom


<h2>1.做法說明</h2>
* 訓練前先進行資料處理



<h2>2.程式方塊圖與寫法</h2>

![image](https://github.com/MachineLearningNTUT/regression-109318083/blob/main/Diagram.jpg)

<h2>3.畫圖做結果分析</h2>

    def transform(listdir,label,lenSIZE):
          label_str=[]
          for i in range (lenSIZE):
              temp = listdir[label[i]]
              label_str.append(temp)
          return label_str

    images = read_images('/content/drive/MyDrive/CharacterML_Hw/test/test/') 
    model = load_model('/content/drive/MyDrive/CharacterML_Hw/train/model10_128.h5')

    predict = model.predict_classes(images, verbose=1)
    print(predict)
    label_str=transform(np.loadtxt('name.txt',dtype='str'),predict,images.shape[0])
<h2>4.討論預測值誤差很大的，是怎麼回事？</h2>
    
    目前Public Leaderboard 為1.0完全猜對

<h2>5.如何改進？</h2>

    如果private 預測下降表示模型可再繼續疊加或者改變參數權重
