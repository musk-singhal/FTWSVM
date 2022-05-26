%SHOULD RETURN FUNCTIONAL DATASET WITH FIRST COLUMN OF LABELS 

function [A,deriv1,deriv2]=read_data(dataset_name)
    if (strcmp(dataset_name,'ham')==true)
        A = csvread('/Users/mnc/Downloads/MATLAB/Ham/Ham.csv',1,0); %read starting from row:1 and column:0
        deriv1 = csvread('/Users/mnc/Downloads/MATLAB/Ham/Ham_first_derivative.csv');
        deriv2 = csvread('/Users/mnc/Downloads/MATLAB/Ham/Ham_second_derivative.csv');
    end
     if (strcmp(dataset_name,'ecg')==true)
        A = csvread('/Users/mnc/Downloads/MATLAB/ECG200/ECG200.csv',1,0); %read starting from row:1 and column:0
        deriv1 = csvread('/Users/mnc/Downloads/MATLAB/ECG200/ECG200_first_derivative.csv');
        deriv2 = csvread('/Users/mnc/Downloads/MATLAB/ECG200/ECG200_second_derivative.csv');
    end
    if (strcmp(dataset_name,'coffee')==true)
        A = csvread('/Users/mnc/Downloads/MATLAB/Coffee/Coffee.csv',1,0); %read starting from row:1 and column:0
        deriv1 = csvread('/Users/mnc/Downloads/MATLAB/Coffee/coffee_first_derivative.csv');
        deriv2 = csvread('/Users/mnc/Downloads/MATLAB/Coffee/coffee_second_derivative.csv');
    end
    if (strcmp(dataset_name,'growth')==true)
        A = csvread('/Users/mnc/Downloads/MATLAB/Growth/growth.csv',1,0); %read starting from row:1 and column:0
        deriv1 = csvread('/Users/mnc/Downloads/MATLAB/Growth/growth_first_derivative.csv');
        deriv2 = csvread('/Users/mnc/Downloads/MATLAB/Growth/growth_second_derivative.csv');
    end
    if (strcmp(dataset_name,'tecator'))==true
        A = csvread('/Users/mnc/Downloads/MATLAB/Tecator/tecator.csv',1,0); 
        A(216:end,:)=[];
        labels=(A(1:end,end-1)<=20.00)*2-1;
%         hist(labels);
        A(:,101:end) = [];
        A=[labels A];
        deriv1 = csvread('/Users/mnc/Downloads/MATLAB/Tecator/tecator_first_derivative.csv');
        deriv2 = csvread('/Users/mnc/Downloads/MATLAB/Tecator/tecator_second_derivative.csv');
    end
    if (strcmp(dataset_name,'weather')==true)
        A = csvread('/Users/mnc/Downloads/MATLAB/weather/CanadianWeather.csv',1,1); %read starting from row:0 and column:0
        tmp = A(:,36:70); %extract class labels from precipitation
        labels=sum(tmp,1);
        labels=(labels<=600)*2-1;
%         hist(labels)
        A=A';        %transpose matrix A
        A(36:end,:)=[];  %take only temperature measurements
        
        A=[labels' A];
        deriv1 = csvread('/Users/mnc/Downloads/MATLAB/weather/weather_first_derivative.csv');
        deriv2 = csvread('/Users/mnc/Downloads/MATLAB/weather/weather_second_derivative.csv');
    end
    if (strcmp(dataset_name,'wafer')==true)
        A = csvread('/Users/mnc/Downloads/MATLAB/Wafer/Wafer.csv',1,0); %read starting from row:1 and column:0
        deriv1 = csvread('/Users/mnc/Downloads/MATLAB/Wafer/Wafer_first_derivative.csv');
        deriv2 = csvread('/Users/mnc/Downloads/MATLAB/Wafer/Wafer_second_derivative.csv');
    end
    if (strcmp(dataset_name,'phoneme')==true)
        A = readtable('/Users/mnc/Downloads/MATLAB/Phoneme/Phoneme.csv'); 
        A=A{:,:};
        labels = A(:,end);
        A(:,end) = [];
        A=[labels A];
        deriv1 = csvread('/Users/mnc/Downloads/MATLAB/Phoneme/Wafer_first_derivative.csv');
        deriv2 = csvread('/Users/mnc/Downloads/MATLAB/Phoneme/Wafer_second_derivative.csv');
    end
    if (strcmp(dataset_name,'yeastcellcycle')==true)
        A = csvread('/Users/mnc/Downloads/MATLAB/Yeastcellcycle/yeastcellcycle.csv',1,0); 
        deriv1 = csvread('/Users/mnc/Downloads/MATLAB/Yeastcellcycle/yeastcellcycle_first_derivative.csv');
        deriv2 = csvread('/Users/mnc/Downloads/MATLAB/Yeastcellcycle/yeastcellcycle_second_derivative.csv');
    end
