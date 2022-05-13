%加载初始参数
%加载用于网络初始化的参数。对于迁移学习，网络初始化参数是初始预训练网络的参数。
trainingSetup = load("D:\MATLAB\bin\AlexNet.mat");

%导入数据
%导入训练和验证数据。
imdsTrain = imageDatastore("D:\data_set\flower_data\train","IncludeSubfolders",true,"LabelSource","foldernames");
imdsValidation = imageDatastore("D:\data_set\flower_data\val","IncludeSubfolders",true,"LabelSource","foldernames");

%增强设置
imageAugmenter = imageDataAugmenter(...
    "RandXReflection",true);

% 调整图像大小以匹配网络输入层。
augimdsTrain = augmentedImageDatastore([227 227 3],imdsTrain,"DataAugmentation",imageAugmenter);
augimdsValidation = augmentedImageDatastore([227 227 3],imdsValidation);

%设置训练选项
%指定训练时要使用的选项。
opts = trainingOptions("adam",...
    "ExecutionEnvironment","gpu",...
    "InitialLearnRate",0.001,...
    "MaxEpochs",50,...
    "LearnRateDropPeriod",30,...
    "LearnRateSchedule","piecewise",...
    "MiniBatchSize",32,...
    "Shuffle","every-epoch",...
    "Plots","training-progress",...
    "ValidationData",augimdsValidation);

%创建层组
layers = [
    imageInputLayer([227 227 3],"Name","data","Mean",trainingSetup.data.Mean)
    convolution2dLayer([11 11],96,"Name","conv1","BiasLearnRateFactor",2,"Stride",[4 4],"Bias",trainingSetup.conv1.Bias,"Weights",trainingSetup.conv1.Weights)
    reluLayer("Name","relu1")
    crossChannelNormalizationLayer(5,"Name","norm1","K",1)
    maxPooling2dLayer([3 3],"Name","pool1","Stride",[2 2])
    groupedConvolution2dLayer([5 5],128,2,"Name","conv2","BiasLearnRateFactor",2,"Padding",[2 2 2 2],"Bias",trainingSetup.conv2.Bias,"Weights",trainingSetup.conv2.Weights)
    reluLayer("Name","relu2")
    crossChannelNormalizationLayer(5,"Name","norm2","K",1)
    maxPooling2dLayer([3 3],"Name","pool2","Stride",[2 2])
    convolution2dLayer([3 3],384,"Name","conv3","BiasLearnRateFactor",2,"Padding",[1 1 1 1],"Bias",trainingSetup.conv3.Bias,"Weights",trainingSetup.conv3.Weights)
    reluLayer("Name","relu3")
    groupedConvolution2dLayer([3 3],192,2,"Name","conv4","BiasLearnRateFactor",2,"Padding",[1 1 1 1],"Bias",trainingSetup.conv4.Bias,"Weights",trainingSetup.conv4.Weights)
    reluLayer("Name","relu4")
    groupedConvolution2dLayer([3 3],128,2,"Name","conv5","BiasLearnRateFactor",2,"Padding",[1 1 1 1],"Bias",trainingSetup.conv5.Bias,"Weights",trainingSetup.conv5.Weights)
    reluLayer("Name","relu5")
    maxPooling2dLayer([3 3],"Name","pool5","Stride",[2 2])
    fullyConnectedLayer(4096,"Name","fc6","BiasLearnRateFactor",2,"Bias",trainingSetup.fc6.Bias,"Weights",trainingSetup.fc6.Weights)
    reluLayer("Name","relu6")
    dropoutLayer(0.5,"Name","drop6")
    fullyConnectedLayer(4096,"Name","fc7","BiasLearnRateFactor",2,"Bias",trainingSetup.fc7.Bias,"Weights",trainingSetup.fc7.Weights)
    reluLayer("Name","relu7")
    dropoutLayer(0.5,"Name","drop7")
    fullyConnectedLayer(6,"Name","fc")
    softmaxLayer("Name","prob")
    classificationLayer("Name","classoutput")];

%训练网络
%使用指定的选项和训练数据对网络进行训练。
[net, traininfo] = trainNetwork(augimdsTrain,layers,opts);
