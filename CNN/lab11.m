%openExample('nnet/TrainABasicConvolutionalNeuralNetworkForClassificationExample')
clc
clear all

% Load and Explore Image Data
% Load the digit sample data as an image datastore. imageDatastore automatically labels 
% the images based on folder names and stores the data as an ImageDatastore object. 
% An image datastore enables you to store large image data, including data that does not fit 
% in memory, and efficiently read batches of images during training of a 
% convolutional neural network.
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%Display some of the images in the datastore and save 20 images in the array Im.
Im = uint8(zeros(28, 28, 20));
figure;
perm = randperm(10000,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
    arr = imread(imds.Files{perm(i)});
    
    Im(:,:,i) = arr; 
end


% Calculate the number of images in each category. 
% labelCount is a table that contains the labels and the number of images having each label. 
% The datastore contains 1000 images for each of the digits 0-9, for a total of 10000 images. 
% You can specify the number of classes in the last fully connected layer of your network as 
% the OutputSize argument.

labelCount = countEachLabel(imds);

%You must specify the size of the images in the input layer of the network. 
%Check the size of the first image in digitData. Each image is 28-by-28-by-1 pixels.

img = readimage(imds,1);
size(img)

% Specify Training and Validation Sets
% Divide the data into training and validation data sets, so that each category in 
% the training set contains 750 images, and the validation set contains the remaining images 
% from each label. splitEachLabel splits the datastore digitData into two new datastores, 
% trainDigitData and valDigitData.

numTrainFiles = 750;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

%Define the CNN Architecture

layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];


%An imageInputLayer is where you specify the image size (28-by-28-by-1). 


% Convolutional Layer In the convolutional layer, the first argument is filterSize, 
% which is the height and width of the filters the training function uses while scanning 
% along the images. In this example, the number 3 indicates that the filter size is 3-by-3. 
% You can specify different sizes for the height and width of the filter. 
% The second argument is the number of filters, numFilters, which is the number of neurons 
% that connect to the same region of the input. This parameter determines the number of feature maps. 
% Use the 'Padding' name-value pair to add padding to the input feature map. 
% For a convolutional layer with a default stride of 1, 'same' padding ensures that the 
% spatial output size is the same as the input size. 
% You can also define the stride and learning rates for this layer using name-value pair arguments of convolution2dLayer.


% Batch Normalization Layer Batch normalization layers normalize the activations and gradients propagating through a network, making network training an easier optimization problem. Use batch normalization layers between convolutional layers and nonlinearities, such as ReLU layers, to speed up network training and reduce the sensitivity to network initialization. Use batchNormalizationLayer to create a batch normalization layer.
% 
% ReLU Layer The batch normalization layer is followed by a nonlinear activation function. The most common activation function is the rectified linear unit (ReLU). Use reluLayer to create a ReLU layer.
% 
% Max Pooling Layer Convolutional layers (with activation functions) are sometimes followed by a down-sampling operation that reduces the spatial size of the feature map and removes redundant spatial information. Down-sampling makes it possible to increase the number of filters in deeper convolutional layers without increasing the required amount of computation per layer. One way of down-sampling is using a max pooling, which you create using maxPooling2dLayer. The max pooling layer returns the maximum values of rectangular regions of inputs, specified by the first argument, poolSize. In this example, the size of the rectangular region is [2,2]. The 'Stride' name-value pair argument specifies the step size that the training function takes as it scans along the input.
% 
% Fully Connected Layer The convolutional and down-sampling layers are followed by one or more fully connected layers. As its name suggests, a fully connected layer is a layer in which the neurons connect to all the neurons in the preceding layer. This layer combines all the features learned by the previous layers across the image to identify the larger patterns. The last fully connected layer combines the features to classify the images. Therefore, the OutputSize parameter in the last fully connected layer is equal to the number of classes in the target data. In this example, the output size is 10, corresponding to the 10 classes. Use fullyConnectedLayer to create a fully connected layer.
% 
% Softmax Layer The softmax activation function normalizes the output of the fully connected layer. The output of the softmax layer consists of positive numbers that sum to one, which can then be used as classification probabilities by the classification layer. Create a softmax layer using the softmaxLayer function after the last fully connected layer.
% 
% Classification Layer The final layer is the classification layer. This layer uses the probabilities returned by the softmax activation function for each input to assign the input to one of the mutually exclusive classes and compute the loss. To create a classification layer, use classificationLayer.

% Specify Training Options

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');


% Train Network Using Training Data
net = trainNetwork(imdsTrain,layers,options);

% Classify Validation Images and Compute Accuracy
% Predict the labels of the validation data using the trained network, and calculate the final validation accuracy. 
% Accuracy is the fraction of labels that the network predicts correctly. In this case, more than 99% of the predicted labels match the true labels of the validation set.

% YPred = classify(net,imdsValidation);
% YValidation = imdsValidation.Labels;
% accuracy = sum(YPred == YValidation)/numel(YValidation);


 X = uint8(zeros(28));
 
 %Predict the labels of the images from array Im, using the trained network

 figure
for j = 1:20
    X = Im(:,:,j); 
    indxx(j,:) = predict(net,X);
    [pr,ind_max] = max(indxx'); 
    pred_labels = ind_max-1;
    subplot(4,5,j);
    imshow(X);
    title(pred_labels(j));
end

%========================================================
% Individual work
% Task1:
% Display 10 images from Validation Set (imdsValidation) and save it in the array Im_valid.
% Predict the labels of the images from Im_valid using the trained network.
% 
% Task2:
% Train CNN only on two classes: "6" and "7".
% Display 10 images of the given classes and save it in the array Im_valid.
% Predict the labels of the images from Im_valid using new network.
%
% Task3:
% Take an image of some different class and predict it;s label using new network.
% l