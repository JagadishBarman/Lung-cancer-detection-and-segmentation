function [ position ] =bwposition( img )
% 自製讀取二值化影像白色點之座標
% out_position為 n*2 之存放座標矩陣
% img為讀入之二值化影像

%% test%%%
% clc,clear,close all
% [filename,pathname] = uigetfile({'*.jpg';'*.*'},'please load one image');
% 
% if isequal([filename,pathname],[0,0]);
%     msgbox('請重新選擇影像。');
%     return;
% end
% 
% im_org_name=[pathname,filename]; % 讀入原圖路徑名
% img=logical(imread(im_org_name));  % 讀入原圖
% figure,imshow(img),title('original');  % 顯示原圖 


%%

[nr nc]=size(img);
p_l=sum(sum(img));
position=zeros(p_l,2);

% for p_l1=1:p_l
%     for p_x=1:nr
%         for p_y=1:nc
            
            [p_x1,p_y1]=find(img);
            
            position(:,1)=p_x1;
             position(:,2)=p_y1;
%         end
%     end
% end

end
