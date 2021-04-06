%  ANN_�������`
 %%
clc,clear,close all

% %% Train_�V�m
% filename = 'GLDM_���n.xlsx';%train_f1.xlsx
% 
% subsetA = xlsread(filename,1); % Ū���V�m�˥�
% subsetA_c = xlsread(filename,2); % Ū���V�m�˥�
% class= subsetA_c; % ��������
% 
% % �S�x���k�@��
% [input,minI,maxI] = premnmx( subsetA' )  ; %
% %�гy���G�x�}
% s = length( class) ;
% output = zeros( s , 2  ) ;
% for i = 1 : s 
%    output( i , class( i )  ) = 1 ;
% end
% % �إ������g����
% net = newff( minmax(input) , [60 3] , { 'logsig' 'purelin' } , 'traingdx' ) ; 
% %{
%     minmax(input)�G�����J�H�����̤j�ȩM�̤p��
%     [10,3]�G��ܨϥ�2�h�����A�Ĥ@�h�����`�I�Ƭ�10�A�ĤG�h�����`�I�Ƭ�3�F
%     { 'logsig' 'purelin' }�G
%     ��ܨC�@�h�������g�����Ұʨ�ơF
%      �Y�G�Ĥ@�h���g�����Ұʨ�Ƭ�logsig�]�u�ʨ�ơ^�A�ĤG�h��purelin�]���S���ಾ��ơ^
% 
%     'traingdx'�G��ܾǲ߳W�h�ĥΪ��ǲߤ�k��traingdx�]��פU���ۧڽվ�ǲ߲v�V�m��ơ^
% %}
% % �]�m�Ѽ�
% net.trainparam.show =50 ;% ;% ��ܤ������G���g��
% net.trainparam.epochs = 10000;%�̤j���йB�⦸�ơ]�ǲߦ��ơ^
% net.trainparam.goal = 0.0001 ;%���g�����V�m���ؼл~�t
% net.trainParam.lr = 0.01 ;%�ǲ߳t�v�]Learning rate�^
% 
% % �}�l�V�m
% %�䤤input���V�m������J�H���A����output���V�m������X���G
% net = train( net, input , output' ) ;
% 
% % �s�����G
% % save('net','net'); 
% % save('minI.mat','minI'); 
% % save('maxI.mat','maxI'); 
%================================�V�m����====================================%
load('net.mat')
load('maxI.mat')
load('minI.mat')
%% Test_����
filename = 'GLDM_���n_t.xlsx';
%  filename = 'data_feature\train_f1.xlsx';
% filename = 'data_feature\test_1_data.xlsx'; % �u��1�˥�

subsetB = xlsread(filename,1); % Ū�����ռ˥�
subsetB_c = xlsread(filename,2); % Ū�����ռ˥�
c= subsetB_c' ; % ��������
  % ��ڤ����A�ΥH����P�w�����G����A�p��X�ǽT�v

% �S�x���k�@��
testInput = tramnmx ( subsetB' , minI, maxI ) ;%,t8,t9,t10,t12,t6,t7

% �S�x���k�@��
Y = sim( net , testInput ) 
 
%�p�⥿�T�v
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
sprintf('���T�v�O %3.3f%%',100 * hitNum / s2 )
