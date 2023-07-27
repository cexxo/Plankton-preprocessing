clear all
warning off

%load dataset
datas=44;
load(strcat('Datas_',int2str(datas)),'DATA');
%NF=size(DATA{3},1); %number of folds
NF = size(DATA{3},1);
DIV=DATA{3};%for the division between training and test set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%I added a division by 4 since it was to huge for my computer :(
DIM1=ceil(DATA{4}/5);%training patterns number
DIM2=ceil(DATA{5}/5);%total patterns number
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
yE=DATA{2};%label of all the patterns
NX=DATA{1};%images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
edgeMethod = 'Canny';
method = 0;             %1 for the bilateral filter and canny; 2 for polar coordinates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%load pre-trained AlexNet
net = alexnet;  %load AlexNet
siz=[227 227];
%parameters
miniBatchSize = 15;                         %Originally it was 30
learningRate = 1e-4;
metodoOptim='sgdm';
options = trainingOptions(metodoOptim,...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',15,...                      %epochs were 30, i'm gonna put it to 15 since i don't have a dedicated GPU
    'InitialLearnRate',learningRate,...
    'Verbose',false,...
    'Plots','training-progress');
numIterationsPerEpoch = floor(DIM1/miniBatchSize);


for fold=1:NF%for each fold
    close all force
    
    trainPattern=(DIV(fold,1:DIM1));
    testPattern=(DIV(fold,DIM1+1:DIM2));
    y=yE(DIV(fold,1:DIM1));%training label
    yy=yE(DIV(fold,DIM1+1:DIM2));%test label
    numClasses = max(y);%number of classes
    
    %create training set
    clear nome trainingImages
    for pattern=1:DIM1
        IM=NX{DIV(fold,pattern)};%image
        
        %insert here any pre-processing on the IM image
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %So here i should filter the image and i should find a way to
        %extract useful features. One of the paper suggests to extract
        %either local or global features. I'm probably gonna start with
        %that and then move to some other approach like gabor features or
        %something i've seen in computer vision for texture recognition.
        %The second paper suggests to combine cartesian coordinates with
        %polar coordinates, so that's what i'll try next.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %EXTRACTION OF GLOBAL FEATURES
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %I've opted for a strong smoothing degree such as 450 to blend
        %better the areas. I opted to do this cause i'd like to try an run
        %the canny edge detector to use that as input for the network.
        %Having more uniform region should lead to a lower rate of not
        %useful edges. I've decided to try first the sobel operator and
        %then i'll check the canny operator for the edge methodd.
        if method == 1
            smoothingDegree=300+(fold*50);%I'm gonna set it according to the folder
            spatialSigma=3;%Gonna set it to 3 and see if there are exceptions since with 4 it doesn't work always.
            copyIM = IM;
            IM = rgb2gray(IM);
            try
                IM=imbilatfilt(IM,smoothingDegree,spatialSigma);
                IM2 = edge(IM,edgeMethod);
            catch ME
                fprintf('Exception\n')
            end
            %montage({copyIM,IM2})
            %x = input("Prompt");
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Image in polar coordinates. I'm gonna try to use magnitude and
        %fase of the image through first order derivatives.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if method == 2
            copyIM = IM;
            IM = rgb2gray(IM);
            [IMdx,IMdy] = imgradientxy(IM);
            [IMmagnitude, IMdirection] = imgradient(IM,'prewitt');
            IM = IMdirection;
            %montage({copyIM,IMdx,IMdy,IMmagnitude, IMdirection})
            %x = input("Prompt");
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        IM=imresize(IM,[siz(1) siz(2)]);%you have to do image resize to make it compatible with CNN
        if size(IM,3)==1
            IM(:,:,2)=IM;
            IM(:,:,3)=IM(:,:,1);
        end
        trainingImages(:,:,:,pattern)=IM;
    end
    imageSize=size(IM);
    
    %data augmentation
    imageAugmenter = imageDataAugmenter( ...
        'RandXReflection',true, ...
        'RandXScale',[1 2]);
    trainingImages = augmentedImageDatastore(imageSize,trainingImages,categorical(y'),'DataAugmentation',imageAugmenter);
    
    %tuning della rete
    % The last three layers of the pretrained network net are configured for 1000 classes.
    %These three layers must be fine-tuned for the new classification problem. Extract all layers, except the last three, from the pretrained network.
    layersTransfer = net.Layers(1:end-3);
    layers = [
        layersTransfer
        fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
        softmaxLayer
        classificationLayer];
    netTransfer = trainNetwork(trainingImages,layers,options);
    
    %test set
    clear nome test testImages
    for pattern=ceil(DIM1)+1:ceil(DIM2)
        IM=NX{DIV(fold,pattern)};%image
        
        %insert here any pre-processing on the IM image
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %So here i should filter the image and i should find a way to
        %extract useful features. One of the paper suggests to extract
        %either local or global features. I'm probably gonna start with
        %that and then move to some other approach like gabor features or
        %something i've seen in computer vision for texture recognition.
        %The second paper suggests to combine cartesian coordinates with
        %polar coordinates, so that's what i'll try next.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %EXTRACTION OF GLOBAL FEATURES
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %I've opted for a strong smoothing degree such as 450 to blend
        %better the areas. I opted to do this cause i'd like to try an run
        %the canny edge detector to use that as input for the network.
        %Having more uniform region should lead to a lower rate of not
        %useful edges. I've decided to try first the sobel operator and
        %then i'll check the canny operator for the edge methodd.
        if method == 1
            smoothingDegree=300+(50*fold);
            spatialSigma=3;%Gonna set it to 3 and see if there are exceptions since with 4 it doesn't work always.
            copyIM = IM;
            IM = rgb2gray(IM);
            try
                IM=imbilatfilt(IM,smoothingDegree,spatialSigma);
                IM2 = edge(IM,edgeMethod);
            catch ME
                fprintf('Exception\n')
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Image in polar coordinates. I'm gonna try to use magnitude and
        %fase of the image through first order derivatives.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if method == 2
            copyIM = IM;
            IM = rgb2gray(IM);
            [IMdx,IMdy] = imgradientxy(IM);
            [IMmagnitude, IMdirection] = imgradient(IM,'prewitt');
            IM = IMdirection;
            %montage({copyIM,IMdx,IMdy,IMmagnitude, IMdirection})
            %x = input("Prompt");
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        IM=imresize(IM,[siz(1) siz(2)]);
        if size(IM,3)==1
            IM(:,:,2)=IM;
            IM(:,:,3)=IM(:,:,1);
        end
        testImages(:,:,:,pattern-ceil(DIM1))=uint8(IM);
    end
    
    %classifying test patterns
    [outclass, score{fold}] =  classify(netTransfer,testImages);
    
    %accuracy
    [a,b]=max(score{fold}');
    ACC(fold)=sum(b==yy)./length(yy);

    %save whatever you need
    if method == 1
        fid = fopen('resultsBilateralCanny.txt','w');
        fprintf(fid,'%6.2f  %12.8f\n',ACC);
    elseif method == 2
        fid = fopen('resultsPolar.txt','w');
        fprintf(fid,'%6.2f  %12.8f\n',ACC);
    else
        fid = fopen('results.txt','w');
        fprintf(fid,'%6.2f  %12.8f\n',ACC);
    end
    %%%%%
    
end


