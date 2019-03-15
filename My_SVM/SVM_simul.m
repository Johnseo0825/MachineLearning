clear, clc, close all;

%% CR system setting
N_sample = 100;
LoSNR = -20;
UpSNR = 0;
SNRdB = linspace(LoSNR,UpSNR,4);
SNR = 10.^(SNRdB./10);

test_SNRdB = linspace(LoSNR,UpSNR,100);
test_SNR = 10.^(test_SNRdB(randperm(100,4))./10);

Ns = 100;
P_N0 = ones(1,10);         % [W]

%% Train / Test data generation
for i=1:length(SNR)
    
    hx(i) = sqrt(SNR(i).*P_N0(i));
    %hx2(i) = sqrt(test_SNR(i).*P_N0(i));
    
        for k=1:N_sample            
            N0 = wgn(Ns,1,P_N0(i),'dBW');            
            train_y_1 = mean( abs(hx(i) + N0).^2 );
            train_y_0 = mean( abs(N0).^2 );
            Train_Data(k,i) = train_y_0;
            Train_Data(N_sample+k,i) = train_y_1;
            
            %test_N0 = wgn(Ns,1,P_N0(i),'dBW');
            %test_y_1 = mean( abs(hx2(i) + test_N0).^2 );
            %test_y_0 = mean( abs(N0).^2 );
            %Test_Data(k,i) = test_y_0;
            %Test_Data(N_sample+k,i) = test_y_1;
        end
end
%plot(Train_Data), hold on, grid on;

Train_label(1:N_sample,1) = 0;
Train_label(N_sample + 1 : 2*N_sample,1) = 1;

for i=1:length(test_SNR)
        
    hx2(i) = sqrt(test_SNR(i).*P_N0(i));
    
        for k=1:N_sample    
            test_N0 = wgn(Ns,1,P_N0(i),'dBW');
            test_y_1 = mean( abs(hx2(i) + test_N0).^2 );
            test_y_0 = mean( abs(N0).^2 );
            Test_Data(k,i) = test_y_0;
            Test_Data(N_sample+k,i) = test_y_1;
        end
end

%% Train / Test data set
X = randn(N_sample*2,10);
idx_test = randperm(10,length(SNR));
%X(:,[1 3 5 7 ]) = Train_Data(:,:);
X(:,idx_test) = Train_Data(:,:);
y = Train_label;

rand_num = randperm(200);

X_train = X(rand_num(1:150),:);
y_train = y(rand_num(1:150),:);

X_test = X(rand_num(151:end),:);
y_test = y(rand_num(151:end),:);

% bar_X = randn(N_sample*2,10);
% %bar_X(:,[1 3 5 7]) = Test_Data(:,:);
% bar_X(:,idx_test) = Test_Data(:,:);
% X_test = bar_X(rand_num(1:150),:);         % Test set
% y_test = y(rand_num(1:150),:);       % Test set label

%% CV partition (CV = Cross Validation)
c = cvpartition(y_train,'k',5);         % cross vailidation(교차검증)을 위한 validation set 설정 

%% feature selection
opts = statset('display','iter');       % Option 
classf = @(train_data, train_labels, test_data, test_labels)...
    sum(predict(fitcsvm(train_data, train_labels, 'KernelFunction','rbf'), test_data)~=test_labels);        % fitcsvm(train_data, train_labels, 'KernelFunction','rbf') = > 모델 설정
[feature_selection, history] = sequentialfs(classf,X_train,y_train,'cv',c,'options',opts,'nfeatures',2);

%% Best hyperparameter
X_train_w_best_feature = X_train(:,feature_selection);
Md1 = fitcsvm(X_train_w_best_feature, y_train,'KernelFunction','rbf',...
    'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus','Showplots',true));

%% Test
X_test_w_best_features = X_test(:,feature_selection);

test_accuracy = sum(predict(Md1,X_test_w_best_features) == y_test)/length(y_test)

%% Hyperplane 확인
figure;
hgscatter = gscatter(X_train_w_best_feature(:,1),X_train_w_best_feature(:,2),y_train);
hold on;
h_sv=plot(Md1.SupportVectors(:,1),Md1.SupportVectors(:,2),'ko','markersize',8);

%% test set의 data를 하나 하나씩 넣어보자.

gscatter(X_test_w_best_features(:,1),X_test_w_best_features(:,2),y_test,'rb','vx')

%% decision plane
XLIMs = get(gca,'xlim');
YLIMs = get(gca,'ylim');
[xi,yi] = meshgrid([XLIMs(1):0.01:XLIMs(2)],[YLIMs(1):0.01:YLIMs(2)]);
dd = [xi(:), yi(:)];
pred_mesh  = predict(Md1, dd);
redcolor = [1, 0.8, 0.8];
bluecolor = [0.8, 0.8, 1];
pos = find(pred_mesh == 0);
h1 = plot(dd(pos,1), dd(pos,2),'s','color',redcolor,'Markersize',5,'MarkerEdgeColor',redcolor,'MarkerFaceColor',redcolor);
pos = find(pred_mesh == 1);
h2 = plot(dd(pos,1), dd(pos,2),'s','color',bluecolor,'Markersize',5,'MarkerEdgeColor',bluecolor,'MarkerFaceColor',bluecolor);
uistack(h1,'bottom');
uistack(h2,'bottom');
legend([hgscatter;h_sv],{'Idle','Occupied','support vectors'})









