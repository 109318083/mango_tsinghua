# ai-wen-mang-guo-bu-liang-pin-fen-lei-jing-sai-109318083
ai-wen-mang-guo-bu-liang-pin-fen-lei-jing-sai-109318083 created by GitHub Classroom


<h2>1.做法說明</h2>

step1 讀入圖片標籤，設定batch size、epochs次數、壓縮的圖片大小


        map_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson', 
        3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel', 
        7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lenny_leonard', 11: 'lisa_simpson', 
        12: 'marge_simpson', 13: 'mayor_quimby', 14: 'milhouse_van_houten', 15: 'moe_szyslak', 
        16: 'ned_flanders', 17: 'nelson_muntz', 18: 'principal_skinner', 19: 'sideshow_bob'}


        pic_size = 64
        batch_size = 64
        epochs = 500
        num_classes = len(map_characters)

        images = []
        labels = []
        name  = []


step2 將其分成訓練集測試集

        def read_main(path):
        global images
        image,labels,name = read_images_labels(path,i=0)
        images = np.array(images,dtype=np.float32)/255
        labels = np_utils.to_categorical(labels,num_classes=20)
        np.savetxt('name.txt',name,delimiter=' ',fmt='%s')
        return images ,labels

        images,labels = read_main('/content/drive/MyDrive/CharacterML_Hw/train/characters-20')
        x_train,x_test,y_train,y_test = train_test_split(images,labels ,test_size=0.1)
step3 設定模型開始訓練

    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', activation ='relu', input_shape = x_train.shape[1:]))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(filters = 86, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation = "softmax"))
    model.summary()
    model.compile(optimizer = 'Adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
    
    datagen = ImageDataGenerator(zoom_range=0.1,width_shift_range=0.05,height_shift_range=0.05,horizontal_flip=True)
    datagen.fit(x_train)

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
