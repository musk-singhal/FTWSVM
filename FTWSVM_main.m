clear all;close all;clc;

%------------------------------------------------------------------------------------------------------------------
%reading csv data
dataset_name='ham';
% dataset_name='ecg';
% dataset_name='tecator';
% dataset_name='weather';
%  dataset_name='coffee';
% dataset_name='wafer';
% dataset_name='growth';
%MULTI-CLASS
% dataset_name='phoneme'; 
% dataset_name='yeastcellcycle';

a=11;  %input('the state=')
KERNEL='lin';
% KERNEL='rbf';

[A,deriv1,deriv2]=read_data(dataset_name);

B=A;  %to save class labels for future use

class=1;
A=B(1:end,2:end); %dropping column 1 of labels from A

%------------------------------------------------------------------------------------------------------------------
%normalizing ORIGINAL data between -1 and 1
Max=max(max(A));
Min=min(min(A));
A=2*(A-Min)./(Max-Min)-1;

%normalizing 1ST DERIVATIVE data between -1 and 1
Max1=max(max(deriv1));
Min1=min(min(deriv1));
deriv1=2*(deriv1-Min1)./(Max1-Min1)-1;

%normalizing 2ND DERIVATIVE data between -1 and 1
Max2=max(max(deriv2));
Min2=min(min(deriv2));
deriv2=2*(deriv2-Min2)./(Max2-Min2)-1;
%------------------------------------------------------------------------------------------------------------------

%SAVING CLASS LABELS as 1 or -1 in d
d=(B(1:end,1)==class)*2-1;
% hist(d);

k=10; % 10-Fold cross validation
% k=size(A,1); % LEAVE-ONE-OUT-CROSS-VALIDATION

output=1;

[sm , sn]=size(A);

cpu_time = 0;
indx = [0:k];
indx = floor(sm*indx/k);    %last row numbers for all 'segments'

% split training set from test set

trainCorr=[];
testCorr=[];
thistoc=[];thistoc0=[];thistoc1=[];thistoc2=[];

rand('state',a);
r=randperm(size(d,1));
d=d(r,:);  %labels
A=A(r,:);  %original
deriv1=deriv1(r,:);  %1st derivative
deriv2=deriv2(r,:);  %2nd derivative

%------------------------------------------------------------------------------------------------------------------



for i = 1:k
    Ctest = []; dtest = [];Ctrain = []; dtrain = []; Ctrain_deriv1 = []; Ctrain_deriv2 = []; Ctest_deriv1=[];Ctest_deriv2=[];
   
    Ctest = A((indx(i)+1:indx(i+1)),:);
    dtest = d(indx(i)+1:indx(i+1));
    Ctrain = A(1:indx(i),:);
    Ctrain = [Ctrain;A(indx(i+1)+1:sm,:)];
    dtrain = [d(1:indx(i));d(indx(i+1)+1:sm,:)];

    %1st derivative of Ctest
    Ctest_deriv1 = deriv1((indx(i)+1:indx(i+1)),:);

    %2nd derivative of Ctest
    Ctest_deriv2 = deriv2((indx(i)+1:indx(i+1)),:);

    %1st derivative of Ctrain 
    Ctrain_deriv1 = deriv1(1:indx(i),:);
    Ctrain_deriv1 = [Ctrain_deriv1 ; deriv1(indx(i+1)+1:sm,:)];

    %2nd derivative of Ctrain 
    Ctrain_deriv2 = deriv2(1:indx(i),:);
    Ctrain_deriv2 = [Ctrain_deriv2 ; deriv2(indx(i+1)+1:sm,:)];

    %------------------------------------------------------------------------------------------------------------------
    %separate class samples to feed into TWSVM

    r=find(dtrain>0); %class 1
    r1=setdiff(1:length(Ctrain(:,1)),r); %class -1
    Y1=dtrain(r,:);
    Y2=dtrain(r1,:);
    cc=Ctrain(r,:); %positive class samples
    dd=Ctrain(r1,:); %negative class samples

    cc1=Ctrain_deriv1(r,:); %1st derivative of positive class samples
    dd1=Ctrain_deriv1(r1,:); %1st derivative of negative class samples

    cc2=Ctrain_deriv2(r,:); %2nd derivative of positive class samples
    dd2=Ctrain_deriv2(r1,:); %2nd derivative of negative class samples

    %------------------------------------------------------------------------------------------------------------------

    %----------------------TWIN---------------------------

    % Define hyperparameter values

%     kernel=1;%kernel type(1:linear, 2:polynomial, 3:RBF)
%     p=0.7;%Kernel parameter (degree for polynomial, kernel width for RBF)
            
          FunPara.c1=0.1;
          FunPara.c2=0.1;
          FunPara.c3=0.1;
          FunPara.c4=0.1;
          FunPara.kerfPara.type = KERNEL;
          FunPara.kerfPara.pars=0.7;
    %------------------------------------------------------------------------------------------------------------------

    DataTrain.A=cc;DataTrain.B=dd;

    tic;
    [predY_train] = TWSVM(Ctrain,DataTrain,FunPara); %TRAINING PREDICTIONS
    thistoc0(i,1)=toc;

    err_train_original = sum(predY_train ~= dtrain);
    tmpTrainCorr_original(i,1)=1-err_train_original/length(Ctrain(:,1));

    
    DataTrain_deriv1.A=cc1;DataTrain_deriv1.B=dd1;
    tic;
    select_features=0;
    [predY_train1] = TWSVM(Ctrain_deriv1,DataTrain_deriv1,FunPara); %TRAINING PREDICTIONS
    thistoc1(i,1)=toc;

    err_train_deriv1 = sum(predY_train1 ~= dtrain);
    tmpTrainCorr_deriv1(i,1)=1-err_train_deriv1/length(Ctrain_deriv1(:,1));

    DataTrain_deriv2.A=cc1;DataTrain_deriv2.B=dd1;
    tic;
    select_features=0;
    [predY_train2] = TWSVM(Ctrain_deriv2,DataTrain_deriv2,FunPara); %TRAINING PREDICTIONS
    thistoc2(i,1)=toc;

    err_train_deriv2 = sum(predY_train2 ~= dtrain);
    tmpTrainCorr_deriv2(i,1)=1-err_train_deriv2/length(Ctrain_deriv2(:,1));
    
    %------------------------------------------------------------------------------------------------------------------
    %VOTING SCHEME - TRAINING

    [predicted_final_train,votes_train]=voting(predY_train,predY_train1,predY_train2);

    err_final_train = sum(predicted_final_train ~= dtrain);
    tmpTrainCorr(i,1)=1-err_final_train/length(Ctrain(:,1));
    %------------------------------------------------------------------------------------------------------------------

    [predicted_original] = TWSVM(Ctest,DataTrain,FunPara); %TESTING PREDICTIONS

    err_original = sum(predicted_original ~= dtest);
    tmpTestCorr_original(i,1)=1-err_original/length(Ctest(:,1));

    [predicted_deriv1] =  TWSVM(Ctest_deriv1,DataTrain_deriv1,FunPara); %TESTING PREDICTIONS

    err_deriv1 = sum(predicted_deriv1 ~= dtest);
    tmpTestCorr_deriv1(i,1)=1-err_deriv1/length(Ctest_deriv1(:,1));

    [predicted_deriv2] = TWSVM(Ctest_deriv2,DataTrain_deriv2,FunPara); %TESTING PREDICTIONS

    err_deriv2 = sum(predicted_deriv2 ~= dtest);
    tmpTestCorr_deriv2(i,1)=1-err_deriv2/length(Ctest_deriv2(:,1));
    
    %------------------------------------------------------------------------------------------------------------------
    %VOTING SCHEME - TESTING

    [predicted_final,votes_test]=voting(predicted_original,predicted_deriv1,predicted_deriv2);

    err_final = sum(predicted_final ~= dtest);
    tmpTestCorr(i,1)=1-err_final/length(Ctest(:,1));

    %------------------------------------------------------------------------------------------------------------------
    thistoc(i,1)=thistoc0(i,1)+thistoc1(i,1)+thistoc2(i,1); 
    
    %  FOLD WISE OUTPUT

    if output==1
        fprintf(1,'________________________________________________\n');
        fprintf(1,'Fold %d\n',i);
        fprintf(1,'Training set correctness: %3.2f%%\n',tmpTrainCorr(i,1)*100);
        fprintf(1,'Testing set correctness: %3.2f%%\n',tmpTestCorr(i,1)*100);
        fprintf(1,'Elapse time: %10.4f\n',thistoc(i,1));
    end
    
end % end of for (looping through test sets)

%------------------------------------------------------------------------------------------------------------------
%FINAL AVERAGE PERFORMANCE OF THE MODEL ON K-FOLD CROSS VALIDATION

%VOTING RULE
trainCorr = sum(tmpTrainCorr*100)/k;
testCorr = sum(tmpTestCorr*100)/k;
cpu_time=sum(thistoc)/k;
testcorrstd = std(100*tmpTestCorr,1);
traincorrstd = std(100*tmpTrainCorr,1);

%ORIGINAL DATASET
trainCorr_original = sum(tmpTrainCorr_original*100)/k;
testCorr_original = sum(tmpTestCorr_original*100)/k;
cpu_time0=sum(thistoc0)/k;
testcorrstd_original = std(100*tmpTestCorr_original,1);
traincorrstd_original = std(100*tmpTrainCorr_original,1);

%1ST DERIVATIVE DATASET
trainCorr_deriv1 = sum(tmpTrainCorr_deriv1*100)/k;
testCorr_deriv1 = sum(tmpTestCorr_deriv1*100)/k;
cpu_time1=sum(thistoc1)/k;
testcorrstd_deriv1 = std(100*tmpTestCorr_deriv1,1);
traincorrstd_deriv1 = std(100*tmpTrainCorr_deriv1,1);

%2ND DERIVATIVE DATASET
trainCorr_deriv2 = sum(tmpTrainCorr_deriv2*100)/k;
testCorr_deriv2 = sum(tmpTestCorr_deriv2*100)/k;
cpu_time2=sum(thistoc2)/k;
testcorrstd_deriv2 = std(100*tmpTestCorr_deriv2,1);
traincorrstd_deriv2 = std(100*tmpTrainCorr_deriv2,1);

%------------------------------------------------------------------------------------------------------------------
if output == 1
    fprintf(1,'===============AVERAGE RESULTS===============================');
    fprintf(1,'\n=================VOTING SCHEME===============================');
    fprintf(1,'\nTraining set correctness: %3.2f%%',trainCorr);
    fprintf(1,'\nTesting set correctness: %3.2f%%',testCorr);
    fprintf(1,'\nAverage cpu_time: %10.4f',cpu_time);
    fprintf(1,'\n Training set Std deviation: %3.2f% \n',traincorrstd);
    fprintf(1,'\n Testing set Std deviation: %3.2f% \n',testcorrstd);

    fprintf(1,'\n=================d=0===============================');
    fprintf(1,'\nTraining set correctness: %3.2f%%',trainCorr_original);
    fprintf(1,'\nTesting set correctness: %3.2f%%',testCorr_original);
    fprintf(1,'\nAverage cpu_time: %10.4f',cpu_time0);
    fprintf(1,'\n Training set Std deviation: %3.2f% \n',traincorrstd_original);
    fprintf(1,'\n Testing set Std deviation: %3.2f% \n',testcorrstd_original);

    fprintf(1,'\n=================d=1===============================');
    fprintf(1,'\nTraining set correctness: %3.2f%%',trainCorr_deriv1);
    fprintf(1,'\nTesting set correctness: %3.2f%%',testCorr_deriv1);
    fprintf(1,'\nAverage cpu_time: %10.4f',cpu_time1);
    fprintf(1,'\n Training set Std deviation: %3.2f% \n',traincorrstd_deriv1);
    fprintf(1,'\n Testing set Std deviation: %3.2f% \n',testcorrstd_deriv1);

    fprintf(1,'\n=================d=2===============================');
    fprintf(1,'\nTraining set correctness: %3.2f%%',trainCorr_deriv2);
    fprintf(1,'\nTesting set correctness: %3.2f%%',testCorr_deriv2);
    fprintf(1,'\nAverage cpu_time: %10.4f',cpu_time2);
    fprintf(1,'\n Training set Std deviation: %3.2f% \n',traincorrstd_deriv2);
    fprintf(1,'\n Testing set Std deviation: %3.2f% \n',testcorrstd_deriv2);
end
%------------------------------------------------------------------------------------------------------------------

% N={dataset_name KERNEL 'TWSVM' k a; 'training' 'testing' 'cpuTime' 'std_train' 'std_test';trainCorr testCorr cpu_time traincorrstd testcorrstd;trainCorr_original testCorr_original cpu_time0 traincorrstd_original testcorrstd_original;trainCorr_deriv1 testCorr_deriv1 cpu_time1 traincorrstd_deriv1 testcorrstd_deriv1;trainCorr_deriv2 testCorr_deriv2 cpu_time2 traincorrstd_deriv2 testcorrstd_deriv2};
% xlswrite('C:/Users/ADMIN/OneDrive/Desktop/nonlinear.xls',N,'append','G36'); %1,8,15,22,29,36,43,50,58

confusionchart(dtrain,predicted_final_train);
