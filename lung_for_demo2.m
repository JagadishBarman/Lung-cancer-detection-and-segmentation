clc,clear,close all

[filename,pathname] = uigetfile({'*.jpg';'*.*'},'please load one image');

if isequal([filename,pathname],[0,0]);
    msgbox('請重新選擇影像。');
    return;
end
 
im_org_name=[pathname,filename]; % 讀入原圖路徑名
im_org=imread(im_org_name);  % 讀入原圖
% figure,imshow(im_org),title('original');  % 顯示原圖 %%%
% dinfo = dicominfo([(im_org_name)]); % 讀入原始dicome檔
% im_org= dicomread(dinfo);

img_org=im_org; % 要點選腫瘤的圖

% dinfo = dicominfo([(im_org_name(1:end-3)),'dcm']); % 讀入原始dicome檔
% [Y map] = dicomread(dinfo);
% figure
% imshow(Y,'DisplayRange',[ ]);
%  spacing = dinfo.PixelSpacing;
% per_pixel_area =spacing(1) * spacing(2); % pixel 單位轉換
% slic_thick= dinfo.SliceThickness; % 切片厚度
spacing=0.8;
per_pixel_area =spacing *spacing;
slic_thick= 1;


img_org = convert2gray( im_org ); % 原圖轉灰階並使用韋納濾波去雜訊
% figure,imshow(img_org),title('move_hist'); %%%%%

[h w]=size(img_org);
[hight width]=size(img_org);
dir_name=[pathname,'*.jpg'];
list=dir(dir_name); % 列出資料夾內檔名
all_num=length(list); % 計算樣本共多少切片

name_all={ list.name};
[name_all_s,index] = sort_nat(name_all);
name_all=name_all_s;



im3=uint8(zeros(h,w));
im_bw3_all=[];
img_orginal=[];

for num=1:all_num
    
    im_name=[pathname,name_all_s{num}]; % 讀入同樣本所有切片之路徑檔名
    img=imread(im_name); % 讀入同樣本所有切片影像
% dinfo1 = dicominfo(im_name); % 讀入原始dicome檔
% img= dicomread(dinfo1 );
%  img= uint8(255 * mat2gray(img));
%     tic
    img_org_all{num}=img; % 存取所有原始灰階影像
    
    img = convert2gray( img ); % 所有切片影像轉灰階並使用韋納濾波去雜訊
%     figure,imshow(img,[]),title('Bilateralfilt2'); % 顯示雙邊濾波後切片影像 %%%
    
    img_orginal{num}=img;
    
    
    img_org1{num}=img; % 存取所有原始灰階影像
    %
    
    img=imadjust(img);
%     figure,imshow(img),title('imadjust'); % 顯示加強對比後切片影像 %%%%%%
    
    Names(num)={im_name}; % 存取所有切片影像之路徑檔名
    img_adj{num}=img; % 存取所有加強對比後之切片影像
    
    
%     img=adapthisteq(img);
%     figure,imshow(img),title('adapthisteq'); % 顯示自適應質方圖增強後切片影像 %%%%%%%%%%%%
    
    bw_img  = my_2Dotsu(img);
%     figure,imshow(bw_img),title('bw_img'); %%% %%%%
    
    %/*****第一次標記**********/
    [im_label,n1] = bwlabel(bw_img,8); % 標記show_X_fill內的面積數
    for n11=1:n1
        l1=find(im_label==n11);
        %         [l11,l12]=find(im_label==n11);
        X1(n11)=length(l1);
    end
    [number lable]=max(X1);
    im_label(im_label~=lable)=0;
    bw_img=logical(im_label);
    figure,imshow(bw_img),title('label_1'); % 顯示第一次標記結果 %%%
    
    
    im_fill=imfill(bw_img,'holes');
%     figure,imshow(im_fill),title('im_fill');% 顯示肺部全輪廓 %%%
    
    in_lung=uint8(im_fill).*img_adj{num};
%     figure,imshow(in_lung),title('in_lung'); %%%
    
    bw=in_lung;
    in_lung(in_lung==0)=[];
    [in_lung threshold]= my_2Dotsu( in_lung);%     level=graythresh(in_lung); %     bw=im2bw(bw,level);
    bw=my_bw(bw,threshold);
%     figure;imshow(bw),title('bw'); %%%
    
    im_bw2 = bwareaopen(bw, 10000,8); % 去除過小點(內部節節或腫瘤)
%     figure,imshow(im_bw2),title('im_bw2');  % %%%
    out_lung=im_bw2;
%     figure,imshow(out_lung),title('out_lung');  % %%%
    
    out_lung_asq=uint8(logical(out_lung)).*img; % 原始外圍之影像
        
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
    
    for i=1:M
        for j=1:N
            im_bw3(i,j)=1-im_bw2(i,j);  % 將肺部內部(有腫瘤區域>左肺與右肺)提取，貼入新圖，並反白
        end
    end
%     figure,imshow(im_bw3),title('im_bw3');   % 顯示左肺與右肺遮罩 %%%
    
    im_bwf=im_bw3.*bw;  % 使用點乘，抓出肺內部的點
%     figure,imshow(im_bwf),title('img_bwf');  % %%%
          
    im_grayd=double(img);
    img_bwf3= uint8(im_bw3.*im_grayd);  % 2塊遮罩乘上原始灰階圖，顯示腫瘤原本灰階值
%     figure,imshow(img_bwf3),title('img_bwf3'); %%%顯示肺內部資訊

    %% /******************************** 標準差二值化**********************************************/
    pixel_labels2 =zeros(size(img_bwf3)); % 建立存放K-means結果圖矩陣
    img_hist=zeros(size(img_bwf3)); % 建立存放直方圖平移後結果圖矩陣
    
    [rowss colss]=find(img_bwf3); % 找出圖內非0數值之原圖座標(x軸及y軸)
    non0=find(img_bwf3); % 找出圖內非0之原圖位置
    XX=img_bwf3(non0);% 找出圖內非0數值，以利後做直方圖平移
%     figure,imhist(img_bwf3(non0)),xlabel('灰階值'), ylabel('次數');
    
    mean_inlung=mean(XX);
    std_inlung=std2(XX);
    mean_std{num,1}=mean_inlung;
    mean_std{num,2}=std_inlung;
    
    % /***直方圖平移法***/
    % (只針對肺內部的左右2塊)
    
    mean_img=mean(XX); % 計算平均值
    for LL=1:length(non0)
        
        img_hist(non0(LL))=(img_bwf3(non0(LL))+(90-mean_img)); % 進行直方圖平移
        
    end
    img_hist=uint8(img_hist);
%     figure,imshow(img_hist),title('move_hist'); % 顯示直方圖平移結果
%     figure,imhist(img_hist(non0)),xlabel('灰階值'), ylabel('次數');
    
%     img_hist_inout=out_lung_asq+img_hist;
% %     figure,imshow( img_hist_inout),title('完整強化之肺部');
img_hist_inout=img_hist;
    img_hist_all{num}=img_hist_inout ; % 存取所有直方圖平移結果圖至cell
    
    non0=find(img_hist); % 找出圖內非0之原圖位置
    XX=img_hist(non0);
    
    mean_inlung2=mean(XX);
    std_inlung2=std2(XX);
    mean_std2{num,1}=mean_inlung2;
    mean_std2{num,2}=std_inlung2;
    threshold=255-(mean_inlung2+(std_inlung2*2));
    img_bw_std= my_bw( img_hist,threshold );
%     figure,imshow(img_bw_std),title('標準差之二值化');
    BW2 = bwareaopen(img_bw_std,10);
%     BW2_all{num}=BW2;
% figure,imshow(BW2)
  
    im_bw3_all{num}=BW2; % 存取所有二值化圖至cell
    im_bw3_all2{num}=double(BW2);
end

%% 三維重建


for num1=1:1:all_num
    llll=['C:\Users\User\Desktop\論文程式碼\final_code\sample\3d\5\mask\mask',num2str(num1),'.jpg'];

% imm{num1}=roipoly( im_bw3_all{num1});
% imm{num1}= im_bw3_all{num1};
% imwrite(imm{num1}, llll);

mask_d{num1}=imread(llll);
eval(sprintf('mask_all{num1}=double(mask_d{num1}).*im_bw3_all2{num1};',num1));

end

figure;imshow( img_org1{all_num});
Sobel_xyz_all=[];
mask_all_1=cell(1,length(mask_all)+2);
mask_all_1{1,1}=zeros(size(mask_all{2}));
mask_all_1{1,2}=zeros(size(mask_all{1}));
mask_all_1{1,length(mask_all)+3}=zeros(size(mask_all{1}));
for nm=1:length(mask_all);
    mask_all_1{nm+1}=mask_all{nm};
end
for nnuumm=1:length(mask_all_1);
    Sobel_xyz_all=cat(3,mask_all_1{nnuumm},Sobel_xyz_all);
end
Data = Sobel_xyz_all;
Data_smooth = smooth3(Data,'gaussian');
hiso = patch(isosurface(Data_smooth,.8),'FaceColor',[.88,.89,.1],'EdgeColor','none');   
hcap = patch(isocaps(Data_smooth,.8),'FaceColor',[0,.25,.0],'EdgeColor','none');
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
hiso = patch(isosurface(Data_smooth,.8),'FaceColor',[.98,.8,.8],'EdgeColor','none');   
hcap = patch(isocaps(Data_smooth,.8),'FaceColor',[.7,.5,.5],'EdgeColor','none');
colormap copper
view(3)
%axis vis3d;
grid on;
daspect([1/0.8,1/0.8,1]);
lightangle(45,20); 
set(gcf,'Renderer','zbuffer'); lighting phong
%FV = isonormals(Dat/smooth,hiso);  
 FV = isosurface(Data_smooth,hiso);    
camlight,lighting gouraud;
axis tight;
xlabel('x');ylabel('y');zlabel('CT slice');
set(hcap,'AmbientStrength',.8);
set(hiso,'SpecularColorReflectance',0,'SpecularExponent',50);
toc