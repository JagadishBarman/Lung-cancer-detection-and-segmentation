function [  n1, im_bw3, im_bwf,bw] = lung_for_demo( img,img_adj,num)
% �ͳ��v������
  level=graythresh(img);
        bw_img =im2bw(img,level);
%    bw_img  = my_2Dotsu(img);
%     figure,imshow(bw_img),title('bw_img'); %%% %%%%
    
    %/*****�Ĥ@���аO**********/
    [im_label,n1] = bwlabel(bw_img,8); % �аOshow_X_fill�������n��
% %     % �s�q�аO�ܷN��
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
    figure,imshow(bw_img),title('label_1'); % ��ܲĤ@���аO���G %%%
%     hold on
% [i11 i22]=find( bw_img==1)
% for ii=1:80:length(i11)
%      text(i22(ii),i11(ii),num2str(1),'Color','red','FontSize',8);
% end
    im_fill=imfill(bw_img,'holes');
%     figure,imshow(im_fill),title('im_fill');% ��ܪͳ������� %%%
    
    in_lung=uint8(im_fill).*img_adj{num};
%     figure,imshow(in_lung),title('in_lung'); %%%
    
    bw=in_lung;
    in_lung(in_lung==0)=[];
%      threshold=graythresh(in_lung);
%         bw_img =im2bw(img,level);
    [in_lung threshold]= my_2Dotsu( in_lung);%     level=graythresh(in_lung); %     bw=im2bw(bw,level);
    bw=my_bw(bw,threshold);
%     figure;imshow(bw),title('bw'); %%%
    
    im_bw2 = bwareaopen(bw, 10000,8); % �h���L�p�I(�����`�`�θ~�F)
%     figure,imshow(im_bw2),title('im_bw2');  % %%%

%     out_lung=im_bw2;
% %     figure,imshow(out_lung),title('out_lung');  % %%%
%     
%     out_lung_asq=uint8(logical(out_lung)).*img; % ��l�~�򤧼v��
        
    % �N�ͳ��ϰ�ϥեH������������
    im_fill=imfill(im_bw2,'holes'); % �ϥθɬ}�k
    im_bw2(repmat(~im_fill,1))=1; % ���Ӫͳ���(imfill)��0(��)�A
    % ��X�e���G�ȤƼv���᪺�ͳ��~����1�N���ܥ�(1)�C
    % �Y�N�I�������]��1�A�ҥHim_bw2�����`�`�B�ܬ�0(��)�A��L��1(��)
    %figure,imshow(im_fill); % 5,title('im_fill')
    % % figure,imshow(~im_fill),title('~1');
    % figure,imshow(~im_bw2),title('~2');
    % figure,imshow(im_bw2); % 6,title('~3')
    
    
    
    [M N]=size(im_bw2);
    
    im_bw3=zeros(M,N);
    im_bw3= imcomplement(im_bw2);% �N�ͳ�����(���~�F�ϰ�>���ͻP�k��)�����A�K�J�s�ϡA�äϥ�
%     for i=1:M
%         for j=1:N
%             im_bw3(i,j)=1-im_bw2(i,j); 
%         end 
%     end

%     figure,imshow(im_bw3),title('im_bw3');   % ��ܥ��ͻP�k�;B�n %%%
    
    im_bwf=im_bw3.*bw;  % �ϥ��I���A��X�ͤ������I
%     figure,imshow(im_bwf),title('img_bwf');  % %%%
%           
%     im_grayd=double(img);
%     img_bwf3= uint8(im_bw3.*im_grayd);  % 2���B�n���W��l�Ƕ��ϡA��ܸ~�F�쥻�Ƕ���
% %     figure,imshow(img_bwf3),title('img_bwf3'); %%%��ܪͤ�����T


end

