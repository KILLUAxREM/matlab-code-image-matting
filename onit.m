% Define paths to the dataset
imageFolder = fullfile('AIM-500/AIM-500/original/');
maskFolder = fullfile('AIM-500/AIM-500/mask/');
trimapFolder = fullfile('AIM-500/AIM-500/trimap/');
usrFolder = fullfile('AIM-500/AIM-500/usr/');

% Create image datastores
imdsOriginal = imageDatastore(imageFolder, 'FileExtensions', {'.jpg', '.png'}, 'LabelSource', 'foldernames');
imdsMask = imageDatastore(maskFolder, 'FileExtensions', {'.png'}, 'LabelSource', 'foldernames');
imdsTrimap = imageDatastore(trimapFolder, 'FileExtensions', {'.png'}, 'LabelSource', 'foldernames');
imdsUsr = imageDatastore(usrFolder, 'FileExtensions', {'.png'}, 'LabelSource', 'foldernames');

% Combine image datastores
ds = combine(imdsOriginal, imdsTrimap, imdsUsr, imdsMask);

% Define the preprocessing function
inputSize = [256 256 3]; % Adjust based on your image dimensions

preprocessFn = @(x) {imresize(x{1}, inputSize(1:2)), imresize(x{2}, inputSize(1:2)), imresize(x{3}, inputSize(1:2)), imresize(x{4}, inputSize(1:2))};

% Apply preprocessing function
ds = transform(ds, preprocessFn);
% Load or define a pre-trained U-Net model (if available)
% Example: Define a simple U-Net-like architecture
numClasses = 1; % For single-channel mask output

inputLayer = imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none');

% Define the encoder
encoder = [
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'enc_conv1')
    reluLayer('Name', 'enc_relu1')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'enc_conv2')
    reluLayer('Name', 'enc_relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')
];

% Define the bottleneck
bottleneck = [
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'bottleneck_conv1')
    reluLayer('Name', 'bottleneck_relu1')
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'bottleneck_conv2')
    reluLayer('Name', 'bottleneck_relu2')
];

% Define the decoder
decoder = [
    transposedConv2dLayer(2, 64, 'Stride', 2, 'Name', 'transconv1')
    reluLayer('Name', 'dec_relu1')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'dec_conv1')
    reluLayer('Name', 'dec_relu2')
    convolution2dLayer(1, numClasses, 'Padding', 'same', 'Name', 'final_conv')
    regressionLayer('Name', 'output')
];

% Combine layers into a layer graph
lgraph = layerGraph([
    inputLayer
    encoder
    bottleneck
    decoder
]);
options = trainingOptions('adam', ...
    'MaxEpochs', 1000, ...
    'MiniBatchSize', 4, ...
    'InitialLearnRate', 0.0001, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ValidationData', ds, ...
    'Shuffle', 'every-epoch');
% Train the network
net = trainNetwork(ds, lgraph, options);
% Load the background image and resize it to match the input size
backgroundImage = imread('/MATLAB Drive/anime/pixelcut-export (2).jpeg');
backgroundImage = imresize(backgroundImage, inputSize(1:2)); % Ensure it matches the network input size

% Load the test image (the image with the object to extract)
testImage = imread('AIM-500/AIM-500/original/o_2d4043cb.jpg');
testImageResized = imresize(testImage, inputSize(1:2)); % Resize to match the input size of the network

% Predict the mask for the test image using the trained network
predictedMask = predict(net, testImageResized);

% Convert the predicted mask to binary (values 0 or 1)
binaryMask = predictedMask > 0.5;  % Threshold the predicted mask

% Ensure the binary mask is 2D (since the predicted mask might be single-channel)
binaryMask = logical(binaryMask(:,:,1));  % Just take the first channel for binary mask

% Now, convert the binary mask to 3 channels so it can be applied to the RGB image
binaryMask3Channel = cat(3, binaryMask, binaryMask, binaryMask);

% Extract the object from the test image using the mask
objectExtracted = testImageResized .* uint8(binaryMask3Channel);  % Use element-wise multiplication

% Invert the binary mask for the background (1 for background, 0 for object)
inverseMask3Channel = ~binaryMask3Channel;  % Invert the binary mask for background

% Apply the inverse mask to the background image
backgroundMasked = backgroundImage .* uint8(inverseMask3Channel);  % Use element-wise multiplication

% Combine the object and the background to create the final composite image
compositeImage = backgroundMasked + objectExtracted;

% Display the final composite image
imshow(compositeImage);
