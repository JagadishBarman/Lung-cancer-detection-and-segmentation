%  ANN_分類結節
 %%
clc,clear,close all

% %% Train_訓練
% filename = 'GLDM_面積.xlsx';%train_f1.xlsx
% 
% subsetA = xlsread(filename,1); % 讀取訓練樣本
% subsetA_c = xlsread(filename,2); % 讀取訓練樣本
% class= subsetA_c; % 分類標籤
% 
% % 特徵值歸一化
% [input,minI,maxI] = premnmx( subsetA' )  ; %
% %創造結果矩陣
% s = length( class) ;
% output = zeros( s , 2  ) ;
% for i = 1 : s 
%    output( i , class( i )  ) = 1 ;
% end
% % 建立類神經網路
% net = newff( minmax(input) , [60 3] , { 'logsig' 'purelin' } , 'traingdx' ) ; 
% %{
%     minmax(input)：獲取輸入信號的最大值和最小值
%     [10,3]：表示使用2層網路，第一層網路節點數為10，第二層網路節點數為3；
%     { 'logsig' 'purelin' }：
%     表示每一層相應神經元的啟動函數；
%      即：第一層神經元的啟動函數為logsig（線性函數），第二層為purelin（對數S形轉移函數）
% 
%     'traingdx'：表示學習規則採用的學習方法為traingdx（梯度下降自我調整學習率訓練函數）
% %}
% % 設置參數
% net.trainparam.show =50 ;% ;% 顯示中間結果的週期
% net.trainparam.epochs = 10000;%最大反覆運算次數（學習次數）
% net.trainparam.goal = 0.0001 ;%神經網路訓練的目標誤差
% net.trainParam.lr = 0.01 ;%學習速率（Learning rate）
% 
% % 開始訓練
% %其中input為訓練集的輸入信號，對應output為訓練集的輸出結果
% net = train( net, input , output' ) ;
% 
% % 存取結果
% % save('net','net'); 
% % save('minI.mat','minI'); 
% % save('maxI.mat','maxI'); 
%================================訓練完成====================================%
load('net.mat')
load('maxI.mat')
load('minI.mat')
%% Test_測試
filename = 'GLDM_面積_t.xlsx';
%  filename = 'data_feature\train_f1.xlsx';
% filename = 'data_feature\test_1_data.xlsx'; % 只有1樣本

subsetB = xlsread(filename,1); % 讀取測試樣本
subsetB_c = xlsread(filename,2); % 讀取測試樣本
c= subsetB_c' ; % 分類標籤
  % 實際分類，用以之後與預測結果比較，計算出準確率

% 特徵值歸一化
testInput = tramnmx ( subsetB' , minI, maxI ) ;%,t8,t9,t10,t12,t6,t7

% 特徵值歸一化
Y = sim( net , testInput ) 
 
%計算正確率
[s1 , s2] = size( Y ) ;
hitNum = 0 ;
predicter=zeros(1,s2);
for i = 1 : s2
    [m , Index] = max( Y( : ,  i ) ) ;
    if( Index  == c(i)   ) 
        hitNum = hitNum + 1 ; 
    end
    [m , predicter(i)] = max( Y( : ,  i ) ) ;
end
sprintf('正確率是 %3.3f%%',100 * hitNum / s2 )
