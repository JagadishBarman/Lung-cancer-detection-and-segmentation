%% test
 clc;clear;close all;
[filename,pathname] = uigetfile({'*.jpg';'*.*'},'please load one image');
AA=1;


spacing1=0.8;
spacing=spacing1;
slic_thick= 1; 
dname =[pathname(1:length(pathname)-1)];


load ('Mod.mat');
load svmStruct;


imcount=0;
counting=0;
counting1=0;
lmeansum1=0;
meanyx1=0;
ll=repmat(uint8(0),512,512,300);
lung3=repmat(logical(0),512,512,300);
% reme=zeros(512);
thr_d=repmat(uint8(0),512,512,300);
imlocation=[];
se1=strel('disk',1);
se20=strel('disk',15);

% tic
for num=1:300
    clear tagg;
    imcount=1+imcount;
    im_n=int2str(imcount);
    im_name=[dname '\' im_n '.jpg'];
    
    % 有無檔案，無檔案停止
    exc=exist(im_name,'file');
    
%     if exc==0
%         msgbox('請重新選擇影像。');
%         return;
%     end
    
    if exc~=0
        im=imread(im_name);
        im=im(:,:,1);
        [yy,xx]=size(im);
        if yy==666
            im=im(78:589,:);
        end
        
%           figure,imshow(im),title('原圖');
          img_org{num}=im;
          
        %% 影像前處理  
      
        im= medfilt2(im);
%         figure,imshow(im),title('中值濾波');
        img_filter{num}=im;
   
       im=imadjust(im);
%        figure,imshow(im),title('gamma轉換');
         img_adjust{num}=im;
         
        im=imsharpen(im,'Amount',2);
%         figure,imshow(im),title('銳化'); % 顯示自適應質方圖增強後切片影像 %
        img_sharp{num}=im;
        
        %% 影像分割
        x=im;
        thr_d(:,:,num)=x; % 三維
        
%        [bw_img, n1,lung,im_bwf]= lung_segmentation( im,img_adjust,num);
        [  n1,lung] = lung_segmentation1( im,img_adjust,num);
        if n1~=0
            lung3(:,:,num)=lung;
            
            img_lung=x.*uint8(logical(lung));
            ll(:,:,num)=img_lung;
            
            lmean=img_lung;
            lmean(lung==0)=[];
            lmeansum=sum(lmean);
            [meanx,meanyx]=size(lmean);
            lmeansum1=lmeansum1+lmeansum;
            meanyx1=meanyx1+meanyx;
        else
            break
        end
    else
        imcount=imcount-1;
        break
    end
end

%三維修正肺部輪廓
lung3=lung3(:,:,1:imcount);
ll=ll(:,:,1:imcount);
L1 = bwconncomp(lung3);
for NO=1:L1.NumObjects;
    NO1(NO)=max(size(L1.PixelIdxList{NO}));
end
[NM,loca1]=max(NO1);
NO1(loca1)=0;
[NM,loca2]=max(NO1);
lung3 = bwlabeln(lung3,26);
lung3(lung3==loca1)=-1;
lung3(lung3==loca2)=-1;
lung3(lung3~=-1)=0;
lung3=logical(lung3);

%%直方圖平移
lungm=round(lmeansum1/meanyx1);
refinevalue=lungm-42;
image=ll-refinevalue;

image=uint8(lung3).*image;
clear lung3 ll

%% 偵測系統
afterbw=repmat(logical(0),512,512,imcount);
reme=repmat(logical(0),512,512,imcount);
for num2=2:imcount-1
    clear mM CDp Ratioo CD;
    img_lung=image(:,:,num2);
    img_lung1=image(:,:,num2-1);
    img_lung3=image(:,:,num2+1);
    img=img_lung+img_lung1+img_lung3;
    new=img;
    img(image(:,:,num2)==0)=[];
    level=graythresh(img);
    new=im2bw(new,level);
    new=imerode(new,se1);
    img2=bwareaopen(new,9);
    img2=imdilate(img2,se1);
    new=img2;
    afterbw(:,:,num2)=img2;
%     afterbw=cat(3,afterbw,ggg15);
    new=bwareaopen(new,10);
    im_bwf=uint8(new).*img_lung; % 使用點乘，抓出肺內部的點
%         figure,imshow(im_bwf),title('肺內部資訊影像')
    [train_data,n2]=bwlabel(im_bwf);    
    counting=n2+counting;
    if n2==0
        train_data=zeros(512);
%         reme=cat(3,reme,trainda);
        reme(:,:,num2)=train_data;
    end
    
    % 特徵擷取
    
    bigdata=regionprops(train_data,'Perimeter','Area','Centroid','MajorAxisLength','MinorAxisLength');
    weight1=regionprops(train_data,img_lung1,'WeightedCentroid');
    weight2=regionprops(train_data,im_bwf,'WeightedCentroid');
    weight3=regionprops(train_data,img_lung3,'WeightedCentroid');
    loading=[];
    for k=1:n2
        A=bigdata(k).Area;
        P=bigdata(k).Perimeter;
        Ratioo(k)=A/P;
        MajorL=bigdata(k).MajorAxisLength;
        MinorL=bigdata(k).MinorAxisLength;
        if A>150 && MinorL/MajorL>0.6;
            loading=cat(2,loading,k);
        end
        mM(k)=MinorL./MajorL;
        CEN=bigdata(k).Centroid;
        C1=weight1(k).WeightedCentroid;
        C2=weight2(k).WeightedCentroid;
        C3=weight3(k).WeightedCentroid;
        CX1=(C1(1)-CEN(1))^2;
        CY1=(C1(2)-CEN(2))^2;
        CD1=CX1+CY1;
        CX2=(C2(1)-CEN(1))^2;
        CY2=(C2(2)-CEN(2))^2;
        CD2=CX2+CY2;
        CX3=(C3(1)-CEN(1))^2;
        CY3=(C3(2)-CEN(2))^2;
        CD3=CX3+CY3;
        CD(k)=(CD1+CD2+CD3)/slic_thick;
        CDp(k)=((CD1+CD2+CD3)/A)/slic_thick;
    end
end

    if n2>0
        TestingSample=[Ratioo;CDp;mM;CD]';
    end