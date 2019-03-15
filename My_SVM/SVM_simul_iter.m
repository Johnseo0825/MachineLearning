clear, clc, close all;

%% CR system setting
N_sample = 100;
LoSNR = -20;
UpSNR = 0;
SNRdB = linspace(LoSNR,UpSNR,4);
%SNRdB = [-20 -20 -20 -20];
SNR = 10.^(SNRdB./10);

test_SNRdB = linspace(LoSNR,UpSNR,100);
test_SNR = 10.^(test_SNRdB(randperm(100,4))./10);
%Ns = 20:20:100;
Ns = 80;
%Ns = [10 50 100];

tic;
                    %출처: https://kimphysics.tistory.com/entry/Matlab-진행바진행-막대-예제 [김물리]
                    %waitbar를 초기화하고 handler에 할당해줍니다.
                    handler = waitbar(0,'Initializing waitbar...');
                    %남은시간 추정을 위해 계산을 시작한 시간을 기록합니다.
                    start = clock;
for iter_Ns=1:length(Ns)
                     pct = iter_Ns/length(Ns); %진행률
                     etr = etime(clock,start)*(1/pct-1); %예상시간
                     waitbar(pct,handler, sprintf('Computing... %d%% eta %d sec', round(pct*100), round(etr)));
    
    
    for iter=1:1

P_N0 = ones(1,10);         % [W]

%% Train / Test data generation
        for i=1:length(SNR)

            hx(i) = sqrt(SNR(i).*P_N0(i));
            mu_H1(i) = (hx(i).^2 ./P_N0(i) + 1).*P_N0(i);
            sigma_H1(i) = sqrt( (P_N0(i).^4)./Ns(iter_Ns) .* (2*SNR(i)+1) );
            threshold(i) = qfunc(0.95)*sigma_H1(i)+mu_H1(i);
            %hx2(i) = sqrt(test_SNR(i).*P_N0(i));

                for k=1:N_sample            
                    N0 = wgn(Ns(iter_Ns),1,P_N0(i),'dBW');            
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

            Eval_data = Train_Data(rand_num(151:end),:);
            Local_decision = Eval_data >= threshold;
            Eval_label = y(rand_num(151:end),:);

            Mat_xor = xor(Local_decision,Eval_label);
            Mat_and = and(Local_decision,Eval_label);
            Accuracy(iter_Ns,iter) = 1-length(find(Mat_xor))/200
            Prob_D(iter_Ns,iter) = length(find(Mat_and))/ ( length(find(y_test))*length(SNR) )

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

%classf_lin = @(train_data, train_labels, test_data, test_labels)...
    %sum(predict(fitcsvm(train_data, train_labels, 'KernelFunction','linear'), test_data)~=test_labels);        % fitcsvm(train_data, train_labels, 'KernelFunction','rbf') = > 모델 설정

%classf_poly = @(train_data, train_labels, test_data, test_labels)...
    %sum(predict(fitcsvm(train_data, train_labels, 'KernelFunction','polynomial'), test_data)~=test_labels);        % fitcsvm(train_data, train_labels, 'KernelFunction','rbf') = > 모델 설정

[feature_selection, history] = sequentialfs(classf,X_train,y_train,'cv',c,'options',opts,'nfeatures',2);
%[feature_selection_lin, history_lin] = sequentialfs(classf_lin,X_train,y_train,'cv',c,'options',opts,'nfeatures',2);
%[feature_selection_poly, history_poly] = sequentialfs(classf_poly,X_train,y_train,'cv',c,'options',opts,'nfeatures',2);

%% Best hyperparameter
X_train_w_best_feature = X_train(:,feature_selection);
Md1 = fitcsvm(X_train_w_best_feature, y_train,'KernelFunction','rbf',...
    'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus','Showplots',true));

% X_train_w_best_feature_lin = X_train(:,feature_selection_lin);
% Md_lin = fitcsvm(X_train_w_best_feature_lin, y_train,'KernelFunction','linear',...
%     'OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%     'expected-improvement-plus','Showplots',false));
% 
% X_train_w_best_feature_poly = X_train(:,feature_selection_poly);
% Md_poly = fitcsvm(X_train_w_best_feature_poly, y_train,'KernelFunction','polynomial',...
%     'OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%     'expected-improvement-plus','Showplots',false));

%% Test
X_test_w_best_features = X_test(:,feature_selection);
% X_test_w_best_features_lin = X_test(:,feature_selection_lin);
% X_test_w_best_features_poly = X_test(:,feature_selection_poly);

test_accuracy(iter_Ns,iter) = sum(predict(Md1,X_test_w_best_features) == y_test)/length(y_test)
% test_accuracy_lin(iter_Ns,iter) = sum(predict(Md_lin,X_test_w_best_features_lin) == y_test)/length(y_test)
% test_accuracy_poly(iter_Ns,iter) = sum(predict(Md_poly,X_test_w_best_features_poly) == y_test)/length(y_test)

iter

    end
end
toc;

figure()
SVM_plot      = plot(Ns,mean( test_accuracy, 2 ).*100,'-or','LineWidth',1.5,'MarkerSize',8 ), grid on, hold on;
% SVM_plot_lin  = plot(Ns,mean( test_accuracy_lin, 2 ).*100,'-vb','LineWidth',1.5,'MarkerSize',8 ), hold on;
% SVM_plot_poly = plot(Ns,mean( test_accuracy_poly, 2 ).*100,'-sk','LineWidth',1.5,'MarkerSize',8 );
Simul_plot    = plot(Ns,mean( Accuracy,2).*100,'-xk','LineWidth',1.5,'MarkerSize',8 ), grid on, hold on;
xlabel('Number of Sensing Samples'), ylabel('Sensing Accuracy [%]');
%legend([SVM_plot;SVM_plot_lin;SVM_plot_poly;Simul_plot],{'SVM-rbg','SVM-linear','SVM-polynomial','Threshold'});
legend([SVM_plot;Simul_plot],{'SVM-rbg','Threshold'});
%% Hyperplane 확인
figure;
hgscatter = gscatter(X_train_w_best_feature(:,1),X_train_w_best_feature(:,2),y_train);
hold on;
h_sv=plot(Md1.SupportVectors(:,1),Md1.SupportVectors(:,2),'ko','markersize',8);

%% test set의 data를 하나 하나씩 넣어보자.

test_scatter = gscatter(X_test_w_best_features(:,1),X_test_w_best_features(:,2),y_test,'rb','vx');

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
legend([hgscatter;test_scatter;h_sv],{'Train: Idle','Train: Occupied','Test: Idle','Test: Occupied','support vectors'})









