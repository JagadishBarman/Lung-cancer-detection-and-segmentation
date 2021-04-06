% function [  p_seed,  x_seed,  y_seed ] = detect(dname, AA,spacing,slic_thick  )

%% test
clc;clear;close all;
[filename,pathname] = uigetfile({'*.jpg';'*.*'},'please load one image');
AA=1;
spacing1=0.8;
spacing=spacing1;
slic_thick= 1;
dname =[pathname(1:length(pathname)-1)];


%% 載入訓練模組
load ('Mod.mat');
% load svmStruct;
load('ann_PD_net')
load('ann_PD_maxI')
load('ann_PD_minI')

%% 設定初值
imcount=0;
count=0;
count1=0;
im_sum=0;
x_num1=0;

imi=repmat(uint8(0),512,512,300);
lung3=repmat(logical(0),512,512,300);
thr_d=repmat(uint8(0),512,512,300);

imlocation=[];
se1=strel('disk',1);
se20=strel('disk',15);

% tic
for num=1:300
    
    %% 讀取影像
    imcount=1+imcount;
    im_n=int2str(imcount);
    im_name=[dname '\' im_n '.jpg'];
    
    % 有無檔案，無檔案停止
    exc=exist(im_name,'file');
    
    if isequal([im_name],[0]);
        msgbox('請重新選擇影像。');
        return;
    end
    
    if exc~=0
        im=imread(im_name);
        im=im(:,:,1);
        [yy,xx]=size(im);
        if yy==666
            im=im(78:589,:);
        end
        
%                   figure,imshow(im),title('原圖');
        img_org{num}=im;
        
        %% 影像前處理
        im= medfilt2(im);
%         im= wiener2(im);
%                 figure,imshow(im),title('中值濾波');
%         img_filter{num}=im;
        
        im=imadjust(im);
%                figure,imshow(im),title('gamma轉換');
        img_adjust{num}=im;
        
        im=imsharpen(im,'Amount',2);
        %         figure,imshow(im),title('銳化'); % 顯示銳化後切片影像 %
%         img_sharp{num}=im;
        
        %% 影像分割
        yy=im;
        thr_d(:,:,num)=yy; % 三維
        
        
           [ n1, im_bw3, im_bwf,bw] = lung_for_demo( im,img_adjust,num);
        bw_all{num}=bw;
        imbwf_all{num}=im_bwf;
        imbw3{num}=im_bw3;
        
        %        [bw_img, n1,lung,im_bwf]= lung_segmentation( im,img_adjust,num);
        [  n1,lung] = lung_segmentation1( im,img_adjust,num);
        
     
        
        
        if n1~=0
            lung3(:,:,num)=lung;
            
            img_lung=yy.*uint8(logical(lung));
            imi(:,:,num)=img_lung;
            
            imii=img_lung;
            imii(lung==0)=[];
            im_sum1=sum(imii);
            [m_x,m_y]=size(imii);
            im_sum=im_sum+im_sum1;
            x_num1=x_num1+m_y;
        else
            break
        end
    else
        imcount=imcount-1;
        break
    end
end

%三維肺部輪廓
lung3=lung3(:,:,1:imcount);
imi=imi(:,:,1:imcount);
L1 = bwconncomp(lung3);
for number=1:L1.NumObjects;
    no1(number)=max(size(L1.PixelIdxList{number}));
end
[nm,loca1]=max(no1);
no1(loca1)=0;
[nm,loca2]=max(no1);
lung3 = bwlabeln(lung3,26);
lung3(lung3==loca1)=-1;
lung3(lung3==loca2)=-1;
lung3(lung3~=-1)=0;
lung3=logical(lung3);

%% 直方圖平移
lung_m=round(im_sum/x_num1);
refine_value=lung_m-42;
image=imi-refine_value;
image=uint8(lung3).*image;
% figure,imshow(image),title('直方圖平移'); % 顯示直方圖平移後切片影像 %

clear lung3 imi

%% 偵測系統
img_bw3d=repmat(logical(0),512,512,imcount);
img_r=repmat(logical(0),512,512,imcount);
for num2=2:imcount-1
    clear R ratio weight_diff weight_diffp ;
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
    img_bw3d(:,:,num2)=img2;
    new=bwareaopen(new,10);
    im_bwf=uint8(new).*img_lung; % 使用點乘，抓出肺內部的點
    % figure,imshow(im_bwf),title('肺內部資訊影像')
        im_bwf_all1{num2}=im_bwf;
    [train_data,n2]=bwlabel(im_bwf);
    count=n2+count;
    if n2==0
        train_data=zeros(512);
        %         reme=cat(3,reme,trainda);
        img_r(:,:,num2)=train_data;
        
    end

    %% 特徵擷取1
    data=regionprops(train_data,'Perimeter','Area','Centroid','MajorAxisLength','MinorAxisLength');
    weight1=regionprops(train_data,img_lung1,'WeightedCentroid');
    weight2=regionprops(train_data,im_bwf,'WeightedCentroid');
    weight3=regionprops(train_data,img_lung3,'WeightedCentroid');
    loading=[];
    for object=1:n2
        
        area=data(object).Area;
        perimeter=data(object).Perimeter;
        R(object)=area/perimeter;
        
        ml=data(object).MajorAxisLength;
        ms=data(object).MinorAxisLength;
        if area>150 && ms/ml>0.6;
            loading=cat(2,loading,object);
        end
        ratio(object)=ms./ml;
        
        centroid=data(object).Centroid;
        cw_1=weight1(object).WeightedCentroid;
        cw_2=weight2(object).WeightedCentroid;
        cw_3=weight3(object).WeightedCentroid;
        wx_1=(cw_1(1)-centroid(1))^2;
        wy_1=(cw_1(2)-centroid(2))^2;
        D1=wx_1+wy_1;
        wx_2=(cw_2(1)-centroid(1))^2;
        wy_2=(cw_2(2)-centroid(2))^2;
        D2=wx_2+wy_2;
        wx_3=(cw_3(1)-centroid(1))^2;
        wy_3=(cw_3(2)-centroid(2))^2;
        D3=wx_3+wy_3;
        weight_diff(object)=(D1+D2+D3)/slic_thick;
        weight_diffp(object)=((D1+D2+D3)/area)/slic_thick;
    end
    
    if n2>0
        testingsample=[R;weight_diffp;ratio;weight_diff];
        
        % 提取特徵值歸一化
        test_input =  tramnmx ( testingsample, ann_PD_minI, ann_PD_maxI ) ;
        Y = sim( ann_PD_net, test_input ) ;
        
        % 進行標籤
        [s1 , s2] = size( Y ) ;
        hitNum = 0 ;
        OutLabel1=zeros(1,s2);
        for i = 1 : s2
            [m , Index] = max( Y( : ,  i ) ) ;
            
            [m ,  OutLabel1(i)] = max( Y( : ,  i ) ) ;
        end
        
        OutLabel1=OutLabel1';
        extrect=find(OutLabel1==1);
        extrect=cat(1,extrect,(loading)');
        [e_size,QQ]=size(extrect);
        for nn=1:e_size
            train_data(train_data==extrect(nn))=-1;
        end
        if e_size>0
            train_data(train_data~=-1)=0;
            train_data(train_data==-1)=1;
        else
            train_data=zeros(512);
        end
        train_data=bwareaopen(train_data,10);
        img_r(:,:,num2)=train_data;
        train_data_all{num2}=train_data;
    end
end

%% 特徵擷取2
im_end=0;
img_r=cat(3,img_r,zeros(512));
img_bw3d=cat(3,img_bw3d,zeros(512));
obj_3d=bwconncomp(img_r);
lab_3d=bwlabeln(img_bw3d);
b=regionprops(obj_3d,'BoundingBox');
v=[];

clear img_r
% toc

for object=1:obj_3d.NumObjects
    clear  cgstd area_per area_diff meanin mean_diff
    per=0;

    c=b(object).BoundingBox;
    c(1)=floor(c(1));
    c(2)=floor(c(2));
    c(3)=ceil(c(3));
    rat=c(4)/c(5);
    if rat<2 && rat>0.5
        im_block=image(c(2):c(2)+c(5)+1,c(1):c(1)+c(4)+1,c(3):c(3)+c(6)-1);
        [bx,by,bz]=size(im_block);
        if (bx/by<2 )&& (bx/by>0.5 )&& (bx*by>15)
            x_range=[c(2):c(2)+c(5)+1];
            y_range=[c(1):c(1)+c(4)+1];
            im_block2=lab_3d( x_range, y_range,c(3)).*img_bw3d( x_range, y_range,c(3));
            num_block=im_block2(round(c(5)/2),round(c(4)/2));
            
            [block,im_block] = before_ann(num_block,c,image,lab_3d,im_block,img_bw3d );
            [xx,yy,z]=size(block);
            %  for nn=1:z
            %   figure;imshow(block(:,:,nn))
            %   end
            
            zz=(z+1)*slic_thick/spacing;
            diff=[xx,yy,zz];
            diff=max(diff)/min(diff);
            if (diff<3) && (diff>0.3) && (z>2) &&(xx<40/spacing) && (yy<40/spacing) && (zz<40/spacing)
                clear z1 
                z1=-1;
                lab_v=bwlabeln(block);
                
                % 顯示唯一值
                if yy>6 && xx>6
                    im=unique(unique(lab_v(round(xx/2)-2:round(xx/2)+2,round(yy/2)-3:round(yy/2)+1,round(z/2))));
                else
                    im=unique(unique(lab_v(round(xx/2)-2:round(xx/2)-1,round(yy/2)-2:round(yy/2)-1,round(z/2))));
                end
                
                im(im==0)=[];
                 if ~isempty(im)
                    lab_v(lab_v~=im)=0;
                    v=logical(lab_v);
                 end
                
                clear me_x me_y me_z v_n2
                if ~isempty(v) && ~isempty(im)
                    %  figure;imshow(image(:,:,c(3)));hold on;
                    %  plot(c(1)+round(c(4)/2),c(2)+round(c(5)/2),'*');
                    
                    count1=count1+1;
                    for n_x=1:xx
                        me_x(n_x)=mean(mean(v(n_x,:,:)));
                    end
                    for n_y=1:yy
                        me_y(n_y)=mean(mean(v(:,n_y,:)));
                    end
                    for n_z=1:z
                        me_z(n_z)=mean(mean(v(:,:,n_z)));
                        area_per(n_z)=length(v(find(v(:,:,n_z)==1)));
                        block_per=double(im_block(:,:,n_z)).*v(:,:,n_z);
                        block_per(block_per==0)=[];
                        block_per=block_per(:)';
                        meanin(n_z)=mean(block_per);
                        if per~=0
                            area_diff(n_z-1)=abs(area_per(n_z)-area_per(n_z-1));
                            mean_diff(n_z-1)=abs(meanin(n_z)-meanin(n_z-1));
                        end
                        per=1;
                    end
                    
                    [max_val,locate]=max(area_per);
                    [v_label,v_n]=bwlabel(v(:,:,locate));
                    
                    if v_n~=1
                        for in=1:v_n
                            v_n2(in)=length(v_label(v_label==in));
                        end
                        [newone,newonelocation]=max(v_n2);
                        v_label(v_label~=newonelocation)=0;
                    end
                    v_label=logical(v_label);
                    % figure;imshow(v_label)
                    
                    sts1=regionprops(v_label,'MajorAxisLength','MinorAxisLength');
                    ml=sts1.MajorAxisLength;
                    ms=sts1.MinorAxisLength;
                    circle_area=max_val/(ms*ml*pi/4);
                                        
                    block_3d=v.*double(im_block);
                    block_3d(block_3d==0)=[];
                    block_3d=block_3d(:)';
                    
                    block_m=mean(block_3d);
                    block_std=std(block_3d);
                    
                    object=length(v(find(v==1)));
                    vr=object/(yy*xx*z);
                    
                    maxin=max(block_3d);
                   mean_area_diff=mean(area_diff);
                    mean_diff(isnan(mean_diff)) = [];
                    m_mean_diff=mean(mean_diff);

                    page=z;
                    
                    std_mx=std(me_x);
                    std_my=std(me_y);
                    std_mz=std(me_z);
                    std_xy=std_mx+std_my+std_mz;

                    max_val=max(meanin);
                    min_val=min(meanin);

                    TMID=m_mean_diff;
                    object_vol=object;

                    testingsample2=[block_m;block_std;page;vr;max_val;min_val;mean_area_diff; m_mean_diff;object_vol;circle_area;std_xy]';
                    
                    o_lab3=Mod10.predict(testingsample2);
                    
                    clear find_nodule
                    find_nodule=cellfun(@isempty,regexp(o_lab3,'no'));
                    
                    %% 找出有結節之肺部切片
                    if find_nodule==1
                        slic_page=round((c(3)+c(3)+c(6)-1)/2);
                        p_seed1(object)={slic_page};
                        x_seed1(object)={c(1)+round(c(4)/2)};
                        y_seed1(object)={c(2)+round(c(5)/2)};
                                
                        im_end=im_end+1;
                        if im_end==50
                            im_end=0;
                        end
                        
                    end
                end
            end
        end
    end
end
% toc

p_seed=cell2mat(p_seed1);
x_seed=cell2mat(x_seed1);
y_seed=cell2mat(y_seed1);

% test
% figure;imshow(thr_d(:,:,slic_page));hold on;
% figure;imshow(thr_d(:,:,p_seed(length(p_seed))));hold on;
% plot(x_seed(length(x_seed)),y_seed(length(y_seed)),'ro','MarkerSize',20);
% disp(['第' num2str(p_seed(length(p_seed))) '張,' 'X' num2str(x_seed(length(x_seed))) ',' 'Y' num2str(y_seed(length(y_seed)))])


% imwrite(thr_d(:,:,slic_page),'detect_noudle.png');
% end

