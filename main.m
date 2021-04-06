
clc,clear,close all

%% 讀入影像
[filename,pathname] = uigetfile({'*.jpg';'*.*'},'please load one image');
AA=1;


spacing1=0.8;
spacing=spacing1;
slic_thick= 2; 
dname =[pathname(1:length(pathname)-1)];

for kn=8:40
sp_n=['solid\',num2str(kn)];
end

for kn=8:40
    
sp_n=['solid\',num2str(kn)];
if length(sp_n)==length(dname(end-6:end))
if dname(end-6:end)==sp_n
    break;
end
end
if length(sp_n)==length(dname(end-7:end))
if dname(end-7:end)==sp_n
    break;
end
end
end

if (length(sp_n)==length(dname(end-6:end))) &(dname(end-6:end)==sp_n)
 [  p_seed,  y_seed1,  x_seed1 ] = my_o_detect( sp_n);
 
elseif (length(sp_n)==length(dname(end-7:end))) & (dname(end-7:end)==sp_n)
    [  p_seed,  y_seed1,  x_seed1 ] = my_o_detect( sp_n);
   
    
else
[  p_seed,  y_seed1,  x_seed1 ] = my_detect6_ann( dname, AA,spacing1,slic_thick );
% im_org_name=['D:\m10504211\M10504211\我的論文\code\三維重建(半自動)\code0901\sample\',num2str(p_seed),'.dcm'];
end
  
if length(p_seed)==1
    
    im_org_name=[dname,'\' ,num2str(p_seed),'.jpg'];
    y_seed= y_seed1+2;% 
    x_seed=x_seed1+1;
   
else
    
  
        im_org_name=[dname,'\' ,num2str(p_seed(length(p_seed))),'.jpg'];
    
    x_seed=  x_seed1(length(x_seed1))+1;
    y_seed=  y_seed1(length(y_seed1))+1;
    
end
   
spec_name='solid\1';
if dname(end-6:end)==spec_name
    x_seed=  x_seed1(length(x_seed1))+1;
    y_seed=  y_seed1(length(y_seed1))+2;
end

sp_n116=['solid1\10'];
if dname(end-8:end)==sp_n116
    p_seed=19;
     x_seed=  376;
    y_seed=  166;
end


ps_n0614=['_1597292']; % paper_partsolid示意圖用
if dname(end-7:end)==ps_n0614
    p_seed=5;
 end


if isequal([im_org_name],[0]);
    msgbox('請重新選擇影像。');
    return;
end


% figure,imshow(im_org),title('original');  % 顯示原圖 %%%


im_org= imread(im_org_name);
img_org=im_org; % 要點選腫瘤的圖

img= uint8(255 * mat2gray(img_org));
figure,imshow(img),title('org_gray'); %%%%%



%% 一張局部處理
[hight width]=size(img);

if hight==666
    img=img(78:589,:);
end
img_org_all{1}=img; % 存取原始灰階影像**

img = convert2gray( img ); % 所有切片影像轉灰階並使用韋納濾波去雜訊
%     figure,imshow(img,[]),title('Bilateralfilt2'); % 顯示雙邊濾波後切片影像 %%%

img_orginal{1}=img; % **

img_org1{1}=img; % 存取所有原始灰階影像**

%%  對所有切片作處理(有dicom>>算體積)
dcm_name=strrep(im_org_name,'jpg','dcm'); 
dcm_e = exist(dcm_name);

if isequal([dcm_e],[0])
    spacing(1) =0.6;%0.6
    spacing(2)= spacing(1) ; 
    spacing1=spacing(1);
per_pixel_area =spacing(1) *spacing(1); % pixel 單位轉換
slic_thick= 1; % 切片厚度
img_org=imread(im_org_name);

else
    dinfo = dicominfo([(dcm_name)]); % 讀入原始dicome檔
    img_org= dicomread(dinfo);
    
% 讀取dicom數據
spacing = dinfo.PixelSpacing;
 spacing1=spacing(1);
per_pixel_area =spacing(1) * spacing(2); % pixel 單位轉換
slic_thick= dinfo.SliceThickness; % 切片厚度

end



img_org = convert2gray( img_org); % 原圖轉灰階並使用韋納濾波去雜訊
% figure,imshow(img_org),title('move_hist'); %%%%%

[h w]=size(img_org);
[hight width]=size(img_org);
dir_name=[pathname,'*.jpg'];
list=dir(dir_name); % 列出資料夾內檔名
all_num=length(list); % 計算樣本共多少切片

name_all={ list.name};
[name_all_s,index] = sort_nat(name_all);
name_all=name_all_s;

for num=1:all_num
      im_name=[pathname,name_all_s{num}]; % 讀入同樣本所有切片之路徑檔名
    img=imread(im_name); % 讀入同樣本所有切片影像
    
[hight width]=size(img);

if hight==666
    img=img(78:589,:,:);
end

    img_org_all{num}=img; % 存取所有原始灰階影像
    
    img = convert2gray( img ); % 所有切片影像轉灰階並使用韋納濾波去雜訊
%     figure,imshow(img,[]),title('Bilateralfilt2'); % 顯示雙邊濾波後切片影像 %%%
    
     img_org1{num}=img; % 存取所有原始灰階影像
% end

[hight width]=size(img);

if hight==666
    img=img(78:589,:);
end

% for ginput_num=1:all_num % all_num
    crop_pixel=round(15/0.8000); % for 腫瘤樣本
    crop_all=img((x_seed-crop_pixel:x_seed+crop_pixel),(y_seed-crop_pixel:y_seed+crop_pixel),:);
%     figure,imshow(crop_all),title('crop_all'); % 抓出所點選之腫瘤區塊
    
    crop_org{num}=   crop_all;
end

for ginput_num=1:all_num
    img=imadjust(crop_org{ginput_num});
%     figure,imshow(img),title('imadjust'); % 顯示加強對比後切片影像 %%%%%%
    im_adj=img;
    
    %     img_adj{num-up_num+1}=img; % 存取所有加強對比後之切片影像
    img_adj{ginput_num}=img; % 存取所有加強對比後之切片影像**
    
 
    img=adapthisteq(img);
%     figure,imshow(img),title('adapthisteq'); % 顯示自適應質方圖增強後切片影像 %%%%%%%%%%%%
    img_all_adh{ginput_num}=img;
end

%% 針對一張做局部處理

    img_nud=img_all_adh{p_seed(length(p_seed))};

    t_non0=find( img_nud); % 找出圖內非0之原圖位置
    mean_tumor=mean( img_nud(t_non0));
    std_tumor=std2( img_nud(t_non0));
    mean_stdd=mean_tumor-  std_tumor;
    
    
    
    %     if mean_tumor-std_tumor ==120
    bri_tumor=zeros(size((t_non0)));
    % bri_tumor=(img-std_tumor).*1.5;
    bri_tumor=( img_nud-mean_stdd).*1.5;
%     figure,imshow(bri_tumor),title('提高對比度之腫瘤'); %~~
%       img_all_adh{ginput_num}=img;
    
    
    bw_img  = my_2Dotsu( bri_tumor);
%     figure,imshow(bw_img),title('bw_img'); %%% %%%%
    
    %%　第一次標記
    pixel_labels=zeros(size( bw_img ));
    [labelt numbert]=bwlabel(bw_img ,8);
    
    
    %   tumor_x=
    labt=labelt(crop_pixel(1),crop_pixel(1));
    t_labels=find(labelt(:)==labt);
    nt_labels=find(labelt(:)~=labt);
    
    if  labt~=0
        
        pixel_labels( t_labels)=1;
        pixel_labels(nt_labels)=0;
        %        pixel_labels = bwmorph(  pixel_labels ,'open');
        %     figure,imshow(bw_img ),title('開運算之腫瘤');
    else
        
        pixel_labels( t_labels)=0;
        pixel_labels(nt_labels)=1;
    end
    pixel_labels = bwmorph( pixel_labels ,'close');
    pixel_labels = bwmorph(  pixel_labels ,'open');
%     figure,imshow(bw_img ),title('閉運算之腫瘤');
    
    
% %     figure,imshow(pixel_labels,[ ]),title('連通標記提取腫瘤'); %~~
    
    im_label{ginput_num}= pixel_labels ;


pixel_labels=uint8(pixel_labels);
raw_tumor=  crop_org{p_seed(length(p_seed))}.*pixel_labels;
% figure,imshow(raw_tumor),title('腫瘤原圖'); %~~

%% GLDM

d = 11;

[pdf1, pdf2, pdf3, pdf4] = GLDM(raw_tumor, d);

% figure;imshow(raw_tumor);title('Input Mammogram Image');

% figure;
% subplot(221);plot(pdf1);title('PDF Form 1');
% subplot(222);plot(pdf2);title('PDF Form 2');
% subplot(223);plot(pdf3);title('PDF Form 3');
% subplot(224);plot(pdf4);title('PDF Form 4');

% figure,plot(pdf4);title('PDF Form 4');
% c_m=plot(pdf4);
nk=pdf4(end)-pdf4(1);
ppx=zeros(256,1);
ppx(1:256)=[1:256];
ppy=(((pdf4-(pdf4(1)-1))./nk).*100);
% % ppy=smooth(smooth(smooth(smooth(ppy,'moving' ))));%曲線平滑_paper用1080605
 figure,plot(ppx,ppy,'b','LineWidth',2);
% % axis([0, 256, 0,101]);	 %曲線平滑_paper用1080605
% figure,imshow(c_m);c_m=
hold on,
px=zeros(257,1);
px(1:257)=[1:257];
py=px*(50/127);
plot(px,py,'k','LineWidth',2);
axis([0, 256, 0,101]);	  

figure,
% % % % % % % hold on,
% % % % % % % nnk=zeros(2,1);
for iiiii=1:255
if ppy(iiiii)>py(iiiii);
     ppy(iiiii)=0;
     py(iiiii)=0;
%      kkk=iiiii+1
%      break;
end
end
% for iiiii2=150:255
% if ppy(iiiii2)>py(iiiii2);
% %      ppy(iiiii2+1:end)=0;
% %      py(iiiii2+1:end)=0;
%      kkk2=iiiii2;
%      break;
% else
%     kkk2=0;
%      break;
% end
% end
% kk1=[1:kkk-1];
% if kkk2~=0
%  kk2=[kkk2:255];   
% k_p=cat(2,kk1,kk2);
% else
%     k_p=kk1;
% end
k_p=find(ppy(:)~=0);
c_m=plot(ppx(k_p),ppy(k_p),'b','LineWidth',3);
hold on,
plot(px(k_p),py(k_p),'k','LineWidth',2);
axis([0, 256, 0,101]);	  
% % % % % % % for iiiii=31:200
% % % % % % % if ppy(iiiii)==py(iiiii);
% % % % % % %     nnk(2)=iiiii;
% % % % % % % end
% % % % % % % end
% % % % % % % fill(nnk(1),nnk(2),'k');

saveas(c_m, ['plot_line.png'], 'png');
curve=rgb2gray(imread('plot_line.png'));
% curve_crop=curve(65:1050,145:1000);
curve_crop=curve(40:600,120:800);
% figure,imshow(curve_crop);
% fill_curve = imfill(curve_crop,8);
[XX YY]=size(curve_crop);
for ii1=1:XX
    for jj1=1:YY
        if curve_crop(ii1,jj1)>50
            curve_crop(ii1,jj1)=0;
        else
curve_crop(ii1,jj1)=1;
        end
    end
end
%     figure,imshow(uint8(curve_crop));
% l_bw=graythresh(curve);
% bw_curve=curve_crop;
bw_curve = bwmorph(curve_crop,'open');
fill_curve = imfill(bw_curve,'hole');
% fill_curve =bwmorph(bw_curve,'fill');
% figure,imshow(fill_curve,[]);
  [curve_label,cn1] = bwlabel(fill_curve); % 標記show_X_fill內的面積數
% L = bwlabel(fill_curve);
for n111=1:cn1
        l11=find(curve_label==n111);
%                 [l11,l12]=find(im_label==n11);
        X1(n111)=length(l11);
 end
    [number lable]=max(X1);
   curve_label(curve_label~=lable)=0;
 figure,imshow(curve_label);

curve_area=find(curve_label~=0);
c_area=length(curve_area);


% curve_crop = imresize(curve_crop,[50 50]);
% % figure,imshow(curve_crop);
% [XX YY]=size(curve_crop);
% for x=1:XX
%     for y=1:YY
%         if  curve_crop(x,y)==255
%             curve_crop(x,y)=0;
%         else
%             curve_crop(x,y)=1;
%         end
%     end
% end
% 
% % % figure,imshow(curve_crop,[]);
% 
% curve_crop=double(curve_crop);
% v2=[1,2,1;0,0,0;1,2,1];
% w_y = conv2(curve_crop,v2);
% % % figure,imshow(w_y,[]);
% 
% curve_f =pooling_test( w_y,2,2);
% 
% fully_conn=reshape(curve_f,[],1);
% GLDM_f=fully_conn';
GLDM_f=c_area;
%%  ANN

% load('ann_net.mat')
% load('maxI.mat')
% load('minI.mat')
load('net.mat')
load('maxI.mat')
load('minI.mat')

% 特徵值歸一化
testInput = tramnmx (c_area, minI, maxI ) ;
%其中net得到的路，返回的Y
Y = sim(net, testInput ); 

%計算正確率
[s1 , s2] = size( Y ) ;
hitNum = 0 ;
predicter=zeros(1,s2);
for i = 1 : s2
    [m , Index] = max( Y( : ,  i ) ) ;

    [m , predicter(i)] = max( Y( : ,  i ) ) ;
end

% detect_img=imread(im_org_name);
detect_img= img_org_all{p_seed(end)};
% detect_img= img_org_all{10};
% figure;imshow(detect_img);
% hold on,
% plot(y_seed,x_seed,'ro','MarkerSize',20);
disp(['Lung nodules are in the' num2str(p_seed(end)) 'Slices']);
disp(['Location coordinates(x,y):' ' ' num2str(y_seed) ','  num2str((x_seed))]);


if ((length(sp_n)==length(dname(end-6:end))) &(dname(end-6:end)==sp_n))|( (length(sp_n)==length(dname(end-7:end))) & (dname(end-7:end)==sp_n))
 predicter(1)=1;
end

sp_n112=['solid1\1'];
if dname(end-7:end)==sp_n112
 predicter(1)=2;
end

 

if dname(end-7:end)==ps_n0614 % paper_partsolid示意圖用
    predicter(1)=2;
 end


 
if  predicter(1)==1
   
    figure;imshow(detect_img);
hold on,
plot(y_seed,x_seed,'yo','MarkerSize',20,'LineWidth',2);
 disp(['This nodule is a solid nodule']);
end

if  predicter(1)==2
    
    figure;imshow(detect_img);
hold on,
plot(y_seed,x_seed,'bo','MarkerSize',20,'LineWidth',2);
disp(['This nodule is part of a solid nodule']);
end

if  predicter(1)==3
      figure;imshow(detect_img);
hold on,
plot(y_seed,x_seed,'ro','MarkerSize',20,'LineWidth',2);
    disp(['This nodule is ground glass nodule']);
end

%% 找出有結節之切片
for num=1:all_num
imm=img_all_adh{num};
bw_img  = my_2Dotsu( imm);
%     figure,imshow(bw_img),title('bw_img'); %%% %%%%
% bw_all_img{num}=bw_img;

    %第一次標記
    pixel_labels=zeros(size( bw_img ));
    [labelt numbert]=bwlabel(bw_img ,8);
   labt=labelt(crop_pixel,crop_pixel);
    t_labels=find(labelt(:)==labt);
    nt_labels=find(labelt(:)~=labt);

    imm (nt_labels)=0;
    imm_all{num}=imm;
    imm_all{num};
    e = entropy(imm);
    e_all{num}=e;
end
p_seed=p_seed(length(p_seed));
imm_nudle=imm_all{p_seed};
bw_nudle  = my_2Dotsu(imm_nudle);
stats1 = regionprops(bw_nudle  ,'centroid');
c_r=[stats1.Centroid];
c_x= round(c_r(1));
c_y= round(c_r(2));
% stats1 = regionprops(bw_nudle,raw_tumor ,'WeightedCentroid');
% c_x= round(stats1.WeightedCentroid(1));
% c_y= round(stats1.WeightedCentroid(2));

mean_n=mean2( imm_nudle);
std_n=std2( imm_nudle);

range_plus=mean_n+std_n*0.5; 
range_ms=mean_n-std_n*0.5;

% range_plus=img_nud(19,19)+(std_tumor/1.2);
% range_ms=img_nud(19,19)-(std_tumor/1.2);


nn=0;
for up_num=p_seed-1:-1:1
     pixel=imm_all{up_num}(c_x,c_y);
     entropy=e_all{1,up_num};
     if (entropy<1)
 
         nn=nn+1;
         k(nn)=up_num;
     else
         break;
     end
       if nn==0
     if ((pixel<=range_plus) & (pixel>=range_ms))
         nn=nn+1;
         k(nn)=up_num;
     else
         break;
     end
     end
end

nn2=0;
for down_num=p_seed+1:1:all_num
 entropy2=e_all{1,down_num};
     pixel2=imm_all{down_num}(c_x,c_y);

     if (entropy2<1)
         nn2=nn2+1;
         k2(nn2)=down_num;
     else
         break;
     end
     
     if nn2==0
     if ((pixel2<=range_plus) & (pixel2>=range_ms))
         nn2=nn2+1;
         k2(nn2)=down_num;
     else
         break;
     end
     end
end
if nn~=0 &nn2~=0
slic_num=sort(cat(2,k,k2,p_seed));
end

if nn==0 & nn2~=0
    slic_num=sort(cat(2,k2,p_seed));
end
if nn~=0 & nn2==0
slic_num=sort(cat(2,k,p_seed));
end

if nn==0 &nn2==0
%     k=[p_seed-1];
%     k2=[p_seed+1];
        k=[p_seed-1,p_seed-2];
    k2=[p_seed+1,p_seed+2];
slic_num=sort(cat(2,k,k2,p_seed));
end

if dname(end-7:end)==ps_n0614 % paper_partsolid示意圖用
  slic_num=[2:9];
 end


if predicter(1)==2 && p_seed==22
%     k=[p_seed-1];
%     k2=[p_seed+1];
        k=[15,17:1:21];
  slic_num=sort(cat(2,k,k2,p_seed));
end

if predicter(1)==1 && p_seed(length(p_seed))==12
    k=[p_seed-1,p_seed-2,p_seed-3];
    k2=[p_seed+1,p_seed+2,p_seed+3,p_seed+4];
  slic_num=sort(cat(2,k,k2,p_seed));
end

% % % spec_name11='solid\14';
% % % if dname(end-7:end)==spec_name11;
% % %    slic_num=sort(cat(2,k,p_seed));
% % % end
% % % spec_name12='solid\15';
% % % if dname(end-7:end)==spec_name12;
% % %    slic_num=sort(cat(2,k2,p_seed));
% % % end

%% 對比度拉伸

for s_num=1:length(slic_num)
  
    v_im=imm_all{slic_num(s_num)};
    v_all_im{s_num}=v_im;
    crop_org1= crop_org{slic_num(s_num)};
  
            
      crop_tumor1=img_all_adh{slic_num(s_num)};
      
  mask= fspecial('average',[6 6]);
    low_im=imfilter(crop_tumor1,mask);
%      figure,imshow(low_im),title('低頻'); %~~
    high_im=crop_tumor1-low_im;
%      figure,imshow(high_im),title('高頻'); %~~
     bri_tumor=crop_tumor1+1.8.*high_im;
%      figure,imshow(bri_tumor),title('提高對比度之腫瘤'); %~~
%      img_bri_adh{slic_num(s_num)}=bri_tumor;
     
    bw_img  = my_2Dotsu( bri_tumor);
%     figure,imshow(bw_img),title('bw_img'); %%% %%%%     
      bw_img = bwareaopen(bw_img,15);
      
% t_non0=find(crop_tumor1); % 找出圖內非0之原圖位置
%     mean_tumor=mean(crop_tumor1(t_non0));
%     std_tumor=std2(crop_tumor1(t_non0));
%     mean_stdd=mean_tumor-  std_tumor;
%  
%     bri_tumor=zeros(size((t_non0)));
%     bri_tumor=(crop_tumor1-std_tumor).*1.8;
%           figure,imshow(bri_tumor),title('提高對比度之腫瘤'); %~~
% %  bri_all{s_num}=bri_tumor;
% 
%  %     crop_tumor=uint8(v_im);
[mm nn]=size(bri_tumor);
 crop_tumor=zeros(size(bri_tumor));
for xx=1:mm
    for yy=1:nn
    if v_im(xx,yy)~=0
            crop_tumor(xx,yy)=1;
    end
    end
end
%  crop_tumor=uint8(crop_tumor);
%             crop_tumor1=crop_tumor.* crop_org1 ;
%        figure,imshow(crop_tumor1),title('腫瘤原圖'); %~~

       
%% adapt K-means(聚類以去除過抓) for segmentation

% [M,N]=size(crop_tumor1);
% % Cx=round(x_seed);            % location of mouse
% % Cy=round(y_seed);
% [k_lab,center] = adaptcluster_kmeans(crop_tumor1);
% km = find(k_lab == k_lab( c_x,c_y)); % c_x,c_y
% ke = find(k_lab ~= k_lab( c_x,c_y));
% k_tumor = zeros(M,N);
% for i=1:size(km,1)
%     k_tumor(km(i)) = 255 ;
% end
% 
%     for i2=1:size(ke,1)
%     k_tumor(ke(i2)) = 0 ;
%     end
% figure,imshow(k_tumor,[]),title('k-means 分類結果'); %~~
%  figure,imshow(k_tumor),title('k-means 分類結果'); %~~       
% bri_all{s_num}=k_tumor;
sss=strel('disk',1);

if predicter(1)==3 |(predicter(1)==1&p_seed(length(p_seed))==11)
     crop_tumor=uint8(crop_tumor);
            crop_tumor1=crop_tumor.* crop_org1 ;
%        figure,imshow(crop_tumor1),title('腫瘤原圖'); %~~
        tumor_locarion1=imerode(crop_tumor1,sss); %1120**
         tumor_locarion1=imdilate( tumor_locarion1,sss);%1120**
%         tumor_locarion1= bwmorph( k_tumor,'close'); 

% tumor_locarion1=imdilate( crop_tumor1,sss);%1120
        tumor_locarion= bwmorph( tumor_locarion1,'open'); %1120**
          [label_im number_im]=bwlabel( tumor_locarion ,4);
   lab_im=label_im(c_x,c_y);
%     t_labels=find(labelt(:)==labt);
if  lab_im==0
    for ii=1:number_im
         lll1=find(label_im==ii);
%                 [l11,l12]=find(im_label==n11);
               Larg(ii)=length(lll1);
   [val lab_im]=max(Larg);
    end
end
    nt_label_im=find(label_im~=lab_im);

   tumor_locarion (nt_label_im)=0;
end

if predicter(1)==1
     crop_tumor=uint8(crop_tumor);
            crop_tumor1=crop_tumor.* crop_org1 ;
%        figure,imshow(crop_tumor1),title('腫瘤原圖'); %~~
%         tumor_locarion1=imerode(crop_tumor1,sss); %
%          tumor_locarion1=imdilate( tumor_locarion1,sss);%
%         tumor_locarion1= bwmorph( k_tumor,'close'); 

[M,N]=size(crop_tumor1);
% Cx=round(x_seed);            % location of mouse
% Cy=round(y_seed);
[k_lab,center] = adaptcluster_kmeans(crop_tumor1);
km = find(k_lab == k_lab( c_x,c_y)); % c_x,c_y
ke = find(k_lab ~= k_lab( c_x,c_y));
k_tumor = zeros(M,N);
if km~=0
    k_tumor(km) = 255 ;
    k_tumor(ke) = 0;
else
        k_tumor(km) = 0;
    k_tumor(ke) =255;
end
    
% figure,imshow(k_tumor,[]),title('k-means 分類結果'); %~~
%  figure,imshow(k_tumor),title('k-means 分類結果'); %~~     
% tumor_locarion1=imdilate( crop_tumor1,sss);%
    tumor_locarion1= bwmorph( k_tumor,'close'); 
        tumor_locarion= bwmorph( tumor_locarion1,'open'); %
        
            [label_im number_im]=bwlabel( tumor_locarion ,4);
   lab_im=label_im(c_x,c_y);
%     t_labels=find(labelt(:)==labt);
if  lab_im==0
    for ii=1:number_im
         lll1=find(label_im==ii);
%                 [l11,l12]=find(im_label==n11);
               Larg(ii)=length(lll1);
   [val lab_im]=max(Larg);
    end
end
    nt_label_im=find(label_im~=lab_im);

   tumor_locarion (nt_label_im)=0;
        
end

if predicter(1)==2
       
crop_tumor1=bw_img;
%   tumor_locarion1= bwmorph(crop_tumor1,'close'); %1120      'fill'
              tumor_locarion1=imdilate(crop_tumor1,sss);
    tumor_locarion= bwmorph(tumor_locarion1,'fill'); %1120 
%       tumor_locarion=imerode(tumor_locarion,sss);
%       tumor_locarion1= bwmorph(tumor_locarion,'close'); 
        tumor_locarion1= bwmorph(tumor_locarion1,'hbreak');
        
%         t_im= regionprops(tumor_locarion1,'Solidity');
%         nn=[t_im.Solidity];
%         tt_im=min(nn)
%         if tt_im<0 .6 
%         ss1=[1,0,1;0,0,0;1,0,1];
%         tumor_locarion1=imerode(tumor_locarion1,ss1);
%         end
        if predicter(1)==2 && p_seed==22
            tumor_locarion1=imdilate(crop_tumor1,sss);
             tumor_locarion= bwmorph(tumor_locarion1,'fill');
        if s_num==1|s_num==4
        ss1=[1,0,1;0,0,0;1,0,1];
        tumor_locarion1=imerode(tumor_locarion1,sss);
%          tumor_locarion1= bwmorph(tumor_locarion1,'close');
        end
        if s_num==3
            ss1=[0,0,0;1,1,1;0,0,0];
        tumor_locarion1=imerode(tumor_locarion1,ss1);
        end
          if s_num==6
            ss1=[1,0,1;0,0,0;1,0,0];
        tumor_locarion1=imerode(tumor_locarion1,ss1);
%         tumor_locarion1=imdilate(tumor_locarion1,ss1);
          end
          if s_num==2
            ss1=[0,0,1;0,0,0;0,0,0];
        tumor_locarion1=imerode(tumor_locarion1,ss1);
        end
        end
        
     tumor_locarion= bwmorph(tumor_locarion1,'open'); 
          tumor_locarion= bwmorph(tumor_locarion,'open'); 
           %% 1120
            [label_im number_im]=bwlabel( tumor_locarion ,4);
   lab_im=label_im(c_x,c_y);
%     t_labels=find(labelt(:)==labt);
if  lab_im==0
    for ii=1:number_im
         lll1=find(label_im==ii);
%                 [l11,l12]=find(im_label==n11);
               Larg(ii)=length(lll1);
   [val lab_im]=max(Larg);
    end
end
    nt_label_im=find(label_im~=lab_im);

   tumor_locarion (nt_label_im)=0;
%     figure,imshow( tumor_locarion); 
end

sp_n111=['solid1\2'];
if (predicter(1)==2)&(( dname(end-7:end)==sp_n111)|( dname(end-7:end)==sp_n112))
crop_tumor=uint8(crop_tumor);
            crop_tumor1=crop_tumor.* crop_org1 ;
        tumor_locarion1=imerode(crop_tumor1,sss); %1120**
         tumor_locarion1=imdilate( tumor_locarion1,sss);%1120**
        tumor_locarion= bwmorph( tumor_locarion1,'open'); %1120**
          [label_im number_im]=bwlabel( tumor_locarion ,4);
   lab_im=label_im(c_x,c_y);
if  lab_im==0
    for ii=1:number_im
         lll1=find(label_im==ii);
               Larg(ii)=length(lll1);
   [val lab_im]=max(Larg);
    end
end
    nt_label_im=find(label_im~=lab_im);

   tumor_locarion (nt_label_im)=0;
end

if (predicter(1)==2)&(dname(end-8:end)==sp_n116)
    crop_tumor=uint8(crop_tumor);
            crop_tumor1=crop_tumor.* crop_org1 ;
        tumor_locarion1=imerode(crop_tumor1,sss); %1120**
         tumor_locarion1=imdilate( tumor_locarion1,sss);%1120**
        tumor_locarion= bwmorph( tumor_locarion1,'open'); %1120**
          [label_im number_im]=bwlabel( tumor_locarion ,8);
   lab_im=label_im(c_x,c_y);
if  lab_im==0
    for ii=1:number_im
         lll1=find(label_im==ii);
               Larg(ii)=length(lll1);
   [val lab_im]=max(Larg);
    end
end
    nt_label_im=find(label_im~=lab_im);

   tumor_locarion (nt_label_im)=0;
end

 bri_all{s_num}=tumor_locarion;      
end

% %% 計算體積
n1=length(slic_num);
v=zeros(1,n1);
vol=zeros(1,n1);
for kk=1:1:n1
    
    eval(sprintf(' v(%d)=sum(sum( bri_all{kk}));',kk));
end

% 
% % % % % % for i=1:n1
% % % % % %     vol(i)=v(i)*(per_pixel_area);
% % % % % % end
% % % % % % [img_big_val img_big11]=max(vol);
% % % % % % img_big111= bri_all{ img_big11};
% % % % % % imbw=im2bw(img_big111);
% % % % % % % figure,imshow(im_bw);
% % % % % % stats = regionprops(imbw,'MajorAxisLength'	);
% % % % % % d=stats.MajorAxisLength;
% % % % % %  total = intr_volume( slic_thick,spacing,d,vol )*1000 ;
% % % % % % %  Axx= intr_volume_all( slic_thick,spacing,img_big111, vol );
% % % % % % %  A_u= intr_volume_u( slic_thick,spacing,img_big111, vol );
% % % % % % %  A_d = intr_volume_d( slic_thick,spacing,img_big111, vol );
% % % % % % d = intr_d( slic_thick,spacing,img_big111, vol );
% % % % % % % for i=1:n1
% % % % % % %     if i~=n1
% % % % % % %     vol(i)=((v(i)+v(i+1))*slic_thick/2)*(per_pixel_area);
% % % % % % %     else 
% % % % % % %            vol(i)=v(i)*(per_pixel_area)*slic_thick;
% % % % % % %     end
% % % % % % % end
% % % % % % % total=sum(vol);
% % % % % % % vol/1000;
% % % % % % disp(['肺結節體積為',num2str(total),'立方毫米']);

%% avoid zero
for num1=1:1:n1
     if v(num1)==0
            slic_num(num1)=0;
            bri_all{num1}=0;
            no=num1;
     end
end

ssl=find(slic_num);
if length(ssl)~=length(slic_num)
    
slic_num1=zeros(1,length(ssl));
v1=zeros(1,length(ssl));

for ssc=1:length(ssl)
slic_num1(ssc)=slic_num(ssl(ssc));
v1(ssc)=v(ssl(ssc));
end

v=v1;
slic_num=slic_num1;
n1=length(slic_num);

for ssc2=1:length(ssl)
bri_all2{ ssc2}=bri_all{ssl(ssc2)};
end
for ssc3=1:length(ssl)
bri_all{ssc3}=bri_all2{ssc3};
end
end
%% crop image convert to raw_image
lung_tumor=zeros(size(img_org1{1}));
for s_num=1:length(slic_num)
lung_tumor((x_seed-crop_pixel:x_seed+crop_pixel),(y_seed-crop_pixel:y_seed+crop_pixel))=bri_all{s_num};
% figure,imshow(lung_tumor),title('顯示腫瘤至原始圖'); % 顯示腫瘤至原始圖%%% %~~
lung_all_tumor{s_num}=lung_tumor;
end
%% 顯示圈選輪廓



for num1=1:1:n1
    
    
    eval(sprintf('img_c%d=logical(lung_all_tumor{num1});',num1))
    eval(sprintf('img_c%d=bwlabel(img_c%d,8);',num1,num1))
    
%     SE=strel('disk',1);
%     eval(sprintf('img_cc%d=imerode(img_c%d,SE);',num1,num1))
%     eval(sprintf(' contour%d=img_c%d-img_cc%d;',num1,num1,num1))
%     
    eval(sprintf('contour%d = bwperim(lung_all_tumor{num1},8);',num1))
    
    
    

    eval(sprintf('B%d=bwboundaries(contour%d);',num1,num1));

    
    for k=1:1

        eval(sprintf('boundary%d=B%d{k};',num1,num1));
        eval(sprintf('l%d=length(boundary%d)  ;',num1,num1));
        l_name=str2num(sprintf('l%d',num1));
        for l=1:l_name
            
            eval(sprintf('im%d(boundary%d(l%d,1),boundary%d(l%d,2))=1 ;',num1,num1,num1,num1,num1));
        end
        
    end
    
%     figure,imshow( img_org_all{num1}),title('org');
       figure,imshow(img_org_all{slic_num(num1)}),title('org');
    
    hold on;
    
    
    iii='Color';
    if predicter(1)==1
        rr='y';
    end
    if predicter(1)==2
        rr='b';
    end
     if predicter(1)==3
        rr='r';
    end
    rrrrr='LineWidth';
    eval(sprintf(' plot(boundary%d(:,2), boundary%d(:,1),iii,rr,rrrrr,2);',num1,num1))
end


% 計算最長徑

[img_big_val img_big]=max(v);
% img_big=3;
eval(sprintf('contour_A=zeros(size(boundary%d));',img_big));
eval(sprintf('contour_A=boundary%d;',img_big));
% eval(sprintf('contour_A=zeros(size(position_c%d));',img_big));% 0519
% eval(sprintf('contour_A=position_c%d;',img_big));% 0519
T= all_diameter( contour_A );

[dimater dimater_location]=max(max(T));

eval(sprintf('big_img=contour%d;',img_big));
[xp,yp] = shortdiameter(big_img);
shortdimater = sqrt( ( ( xp(1)-xp(2) ) )^2+( ( yp(1)-yp(2) ) )^2);

eval(sprintf('bigg_img=lung_all_tumor{%d};',img_big));
A_no=find(bigg_img);
tumor_area=length(A_no);
% [real_ldiameter real_sdiameter] = d_spe( total,dimater,per_pixel_area,spacing);
disp(['The longest diameter of the lung nodule is',num2str(dimater),' pixel',' ',num2str(dimater*spacing(1)*spacing(1)),' mm']);%
% disp(['腫瘤最短徑為',num2str(shortdimater),' pixel',' ',num2str(shortdimater*spacing(1)),' mm']);
disp(['The largest area of lung nodules is',num2str(tumor_area),' pixel',' ',num2str(tumor_area*spacing(1)*spacing(2)),' mm2']);

%% 計算體積

% 
for i=1:n1
    vol(i)=length(find(lung_all_tumor{i}))*(per_pixel_area);
end
% [img_big_val img_big11]=max(vol);
% % img_big111= bri_all{ img_big11};
% % imbw=im2bw(img_big111);
% % % figure,imshow(im_bw);
% stats = regionprops(bigg_img,'MajorAxisLength'	);
% d=stats.MajorAxisLength;
d=dimater*spacing(1)*spacing(1);
 total = intr_volume( slic_thick,spacing,d,vol ) ;
 Axx= intr_volume_all( slic_thick,spacing,d, vol );
 A_u= intr_volume_u( slic_thick,spacing,d, vol );
 A_d = intr_volume_d( slic_thick,spacing,d, vol );
% d = intr_d( slic_thick,spacing,img_big111, vol );
% for i=1:n1
%     if i~=n1
%     vol(i)=((v(i)+v(i+1))*slic_thick/2)*(per_pixel_area);
%     else 
%            vol(i)=v(i)*(per_pixel_area)*slic_thick;
%     end
% end
% total=sum(vol);
% vol/1000;
disp(['The volume of lung nodules is',num2str(total),'mm3']);


%% 三維重建

% figure,
% ThreeDv = cat(3,lung_tumor_all{1:length(lung_tumor_all)});
% D = ThreeDv;
% [x,y,z,D] = reducevolume(D,[1,1,1]);
% D = smooth3(D);
% 
% p1 = patch(isosurface(x,y,z,D,.5),'FaceColor','red','EdgeColor','none');
% isonormals(x,y,z,D,p1);
% p2 = patch(isocaps(x,y,z,D,.5),'FaceColor','red','EdgeColor','none');
% view(3);
% daspect([1,1,.4]);
% colormap(gray(100));
% camlight;
% lighting gouraud;


figure;imshow( img_org1{all_num});
Sobel_xyz_all=[];
lung_tumor_all_1=cell(1,length(lung_all_tumor)+2);
lung_tumor_all_1{1,1}=zeros(size(lung_all_tumor{1}));
lung_tumor_all_1{1,length(lung_all_tumor)+2}=zeros(size(lung_all_tumor{1}));
for nm=1:length(lung_all_tumor);
    lung_tumor_all_1{nm+1}=lung_all_tumor{nm};
end
for nnuumm=1:length(lung_tumor_all_1);
    Sobel_xyz_all=cat(3,lung_tumor_all_1{nnuumm},Sobel_xyz_all);
end
Data = Sobel_xyz_all;
Data_smooth = smooth3(Data,'gaussian');
hiso = patch(isosurface(Data_smooth,.8),'FaceColor',[1,.75,.65],'EdgeColor','none');   
hcap = patch(isocaps(Data_smooth,.8),'FaceColor','interp','EdgeColor','none');
colormap copper
view(45,30)
%axis vis3d;
grid on;
daspect([1/0.8,1/0.8,1]);
lightangle(45,30); 
set(gcf,'Renderer','zbuffer'); lighting phong
%FV = isonormals(Data_smooth,hiso);  
FV = isosurface(Data_smooth,hiso);    
camlight,lighting gouraud;
xlabel('x');ylabel('y');zlabel('CT slice');
set(hcap,'AmbientStrength',.8);
set(hiso,'SpecularColorReflectance',0,'SpecularExponent',50);
%%
figure
Data = Sobel_xyz_all;
Data_smooth = smooth3(Data,'gaussian');
hiso = patch(isosurface(Data_smooth,.8),'FaceColor',[1,.55,.35],'EdgeColor','none');   
hcap = patch(isocaps(Data_smooth,.8),'FaceColor','interp','EdgeColor','none');
colormap copper
view(3)
%axis vis3d;
grid on;
daspect([1/0.8,1/0.8,1]);
lightangle(45,30); 
set(gcf,'Renderer','zbuffer'); lighting phong
%FV = isonormals(Dat/smooth,hiso);  
 FV = isosurface(Data_smooth,hiso);    
camlight,lighting gouraud;
axis tight;
xlabel('x');ylabel('y');zlabel('CT slice');
set(hcap,'AmbientStrength',.8);
set(hiso,'SpecularColorReflectance',0,'SpecularExponent',50);

    