function [curve_f ] =pooling_test(A, X,Y)
% max_pooling
% A為輸入矩陣
% X,Y 為池化步數


% %% test
% clc,clear,close all
% 
% % A=[1,2,3,1;4,5,6,2;7,8,9,3;5,6,8,7];
% A=[1,2,3,4;5,6,7,8;9,10,11,12;13,14,15,16,];
% X=2;
% Y=2;

[xx yy]=size(A);
mask=zeros(X,Y);
[mm nn]=size(mask);

gap=1;
curve_f=zeros(xx-gap,yy-gap);
%   for xx1=1:1:(xx-mm+1)
%     
% for x1=xx1+1:1:xx
%     
%      
%     for yy1=1:1:(yy-nn+1)
%       for y1=yy1+1:1:yy
%         curve_f(xx1,yy1)=max(max(A(xx1:1:x1,yy1:1:y1)));
%         
%         end
%     end
%     end
% end
% end
for xx1=1:1:(xx-1)
    for yy1=1:1:(yy-1)
        
 curve_f(xx1,yy1)=max(max(A(xx1:1:(xx1+mm-1),yy1:1:(yy1+nn-1))));
 
    end
end
end
 