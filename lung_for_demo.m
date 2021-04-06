function [  n1, im_bw3, im_bwf,bw] = lung_for_demo( img,img_adj,num)
% 肺部影像分割
  level=graythresh(img);
        bw_img =im2bw(img,level);
%    bw_img  = my_2Dotsu(img);
%     figure,imshow(bw_img),title('bw_img'); %%% %%%%
    
    %/*****第一次標記**********/
    [im_label,n1] = bwlabel(bw_img,8); % 標記show_X_fill內的面積數
% %     % 連通標記示意圖
% %     figure,imshow(im_label);
% %     hold on
% %     for i=1:n1
% %         R=find(im_label(:)==i)
% %         x(i)=length(R);
% %         [M V]=max(x);
% %         if x(i)==M
% %             [x_p y_p]=find(im_label(:)==i);      
% %             text(y_p,x_p,num2str(1),'Color','red','FontSize',14);
% %         end
% %     end
    
    for n11=1:n1
        l1=find(im_label==n11);
%                 [l11,l12]=find(im_label==n11);
        X1(n11)=length(l1);
              end
    [number lable]=max(X1);
    im_label(im_label~=lable)=0;
    bw_img=logical(im_label);
    figure,imshow(bw_img),title('label_1'); % 顯示第一次標記結果 %%%
%     hold on
% [i11 i22]=find( bw_img==1)
% for ii=1:80:length(i11)
%      text(i22(ii),i11(ii),num2str(1),'Color','red','FontSize',8);
% end
    im_fill=imfill(bw_img,'holes');
%     figure,imshow(im_fill),title('im_fill');% 顯示肺部全輪廓 %%%
    
    in_lung=uint8(im_fill).*img_adj{num};
%     figure,imshow(in_lung),title('in_lung'); %%%
    
    bw=in_lung;
    in_lung(in_lung==0)=[];
%      threshold=graythresh(in_lung);
%         bw_img =im2bw(img,level);
    [in_lung threshold]= my_2Dotsu( in_lung);%     level=graythresh(in_lung); %     bw=im2bw(bw,level);
    bw=my_bw(bw,threshold);
%     figure;imshow(bw),title('bw'); %%%
    
    im_bw2 = bwareaopen(bw, 10000,8); % 去除過小點(內部節節或腫瘤)
%     figure,imshow(im_bw2),title('im_bw2');  % %%%

%     out_lung=im_bw2;
% %     figure,imshow(out_lung),title('out_lung');  % %%%
%     
%     out_lung_asq=uint8(logical(out_lung)).*img; % 原始外圍之影像
        
    % 將肺部區域反白以提取內部輪廓
    im_fill=imfill(im_bw2,'holes'); % 使用補洞法
    im_bw2(repmat(~im_fill,1))=1; % 把整個肺部區(imfill)變0(黑)，
    % 找出前面二值化影像後的肺部外圍變1將其變白(1)。
    % 即將背景像素設為1，所以im_bw2中有節節處變為0(黑)，其他韋1(白)
    %figure,imshow(im_fill); % 5,title('im_fill')
    % % figure,imshow(~im_fill),title('~1');
    % figure,imshow(~im_bw2),title('~2');
    % figure,imshow(im_bw2); % 6,title('~3')
    
    
    
    [M N]=size(im_bw2);
    
    im_bw3=zeros(M,N);
    im_bw3= imcomplement(im_bw2);% 將肺部內部(有腫瘤區域>左肺與右肺)提取，貼入新圖，並反白
%     for i=1:M
%         for j=1:N
%             im_bw3(i,j)=1-im_bw2(i,j); 
%         end 
%     end

%     figure,imshow(im_bw3),title('im_bw3');   % 顯示左肺與右肺遮罩 %%%
    
    im_bwf=im_bw3.*bw;  % 使用點乘，抓出肺內部的點
%     figure,imshow(im_bwf),title('img_bwf');  % %%%
%           
%     im_grayd=double(img);
%     img_bwf3= uint8(im_bw3.*im_grayd);  % 2塊遮罩乘上原始灰階圖，顯示腫瘤原本灰階值
% %     figure,imshow(img_bwf3),title('img_bwf3'); %%%顯示肺內部資訊


end

