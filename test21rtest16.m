%%3.6
% clc;clear all;close all;
% nooo=input('幾張');
% for noooo=1:2:nooo   
% ggg=0;
% lun=0;
% gg=0;
%     for noo=noooo:1:noooo+2
%         str1=int2str(noo);
%         str2=[str1 '.jpg'];
%         A=imread(str2);
%         x=A(:,:,1);
%         x=wiener2(x);
%         if noo==noooo+1
%             final=x;
%         end
%         [yy,xx]=size(x);
%         x=imadjust(x); 
%         level=graythresh(x);
%         bi=im2bw(x,level);
%         bi=bwareaopen(bi,500);
%         [label1,m]=bwlabel(bi);
%         max0=0;
%         for n=1:m
%                 f=find(label1==n);
%             if max0<length(f)
%                 max0=length(f);
%                 label=n;
%             end
%         end
%         label1(find(label1~=label))=0;
%         label1(find(label1~=0))=1;
%         logi=logical(label1);
%         se=strel('diamond',3);
%         logi=imdilate(logi,se);
%         fil=logical(imfill(label1,'holes'));
%         lung=uint8(fil-logi);
%         lun=logical(lung)+lun;
%         lungg=x.*lung;
%         lunggg=lungg;
%         if gg==0
%             gg=lungg;
%         else
%             gg(lungg>=gg)=0;
%             lungg(gg>lungg)=0;
%             gg=gg+lungg;    
%         end                      
%         ggg=ggg+lunggg;
%     end  
%     new=ggg;
%     new1=ggg;
%     lun(find(lun~=3))=0;
%     lun=logical(lun);
%     ggg=uint8(lun).*ggg;
%     ggg(find(ggg==0))=[];
%     level=graythresh(ggg);
%     new=im2bw(new,level);
%     se=strel('diamond',1);
%     new=imerode(new,se);
%     ggg15=bwareaopen(new,15);
%     new=ggg15;
%      new=imdilate(new,se);
%      if max(max(new))~=0
%         figure;imshow(final);hold on;
%         [imLL,nll]=bwlabel(new);
%         cen = regionprops(imLL,'centroid');
%         for coun = 1: numel(cen)         
%             circle(cen(coun).Centroid(1),cen(coun).Centroid(2),15); hold on;
%         end
%     end
% end

        
%%maybe OK (fast3.05)
% clc;clear all;close all;
% nooo=input('幾張');
% for noooo=1:2:nooo   
% ggg=0;
% lun=0;
% gg=0;
%     for noo=noooo:1:noooo+2
%         str1=int2str(noo);
%         str2=[str1 '.jpg'];
%         A=imread(str2);
%         x=A(:,:,1);
%         x=wiener2(x);
%         if noo==noooo+1
%             final=x;
%         end
%         [yy,xx]=size(x);
%         x=imadjust(x); 
%         level=graythresh(x);
%         bi=im2bw(x,level);
%         bi=bwareaopen(bi,500);
%         [label1,m]=bwlabel(bi);
%         max0=0;
%         for n=1:m
%                 f=find(label1==n);
%             if max0<length(f)
%                 max0=length(f);
%                 label=n;
%             end
%         end
%         label1(find(label1~=label))=0;
%         logi=logical(label1);
%         logi=imcomplement(logi);
%         bla=bwlabel(logi);
%         bla(bla==1)=0;
%         bound=[bla(1,:),bla(end,:),bla(:,1)',bla(:,end)'];
%         bound=unique(bound);
%         [tx,ty]=size(bound);
%         for ttt=1:ty
%             bla(bla==bound(1,ttt))=0;
%         end
%         bla=logical(bla);
%         se=strel('disk',1);
%         lung=imerode(bla,se);
%         lun=logical(lung)+lun;
%         lungg=x.*uint8(lung);
%         lunggg=lungg;
%         if gg==0
%             gg=lungg;
%         else
%             gg(lungg>=gg)=0;
%             lungg(gg>lungg)=0;
%             gg=gg+lungg;    
%         end                      
%         ggg=ggg+lunggg;
%     end  
%     new=ggg;
%     new1=ggg;
%     lun(find(lun~=3))=0;
%     lun=logical(lun);
%     ggg=uint8(lun).*ggg;
%     ggg(find(ggg==0))=[];
%     level=graythresh(ggg);
%     new=im2bw(new,level);
%     se=strel('diamond',1);
%     new=imerode(new,se);
%     ggg15=bwareaopen(new,15);
%     new=ggg15;
%      new=imdilate(new,se);
%      if max(max(new))~=0
%         figure;imshow(final);hold on;
%         [imLL,nll]=bwlabel(new);
%         cen = regionprops(imLL,'centroid');
%         for coun = 1: numel(cen)         
%             circle(cen(coun).Centroid(1),cen(coun).Centroid(2),15); hold on;
%         end
%     end
% end

%%3.5
% clc;clear all;close all;
% nooo=input('幾張');
% for noooo=1:2:nooo   
% ggg=0;
% lun=0;
% gg=0;
%     for noo=noooo:1:noooo+2
%         str1=int2str(noo);
%         str2=[str1 '.jpg'];
%         A=imread(str2);
%         x=A(:,:,1);
%         x=wiener2(x);
%         if noo==noooo+1
%             final=x;
%         end
%         x=imadjust(x); 
%         level=graythresh(x);
%         bi=im2bw(x,level);
%         bi=bwareaopen(bi,500);
%         [label1,m]=bwlabel(bi);
%         max0=0;
%         for n=1:m
%                 f=find(label1==n);
%             if max0<length(f)
%                 max0=length(f);
%                 label=n;
%             end
%         end
%         label1(find(label1~=label))=0;
%         logi=logical(label1);
%         fil=logical(imfill(label1,'holes'));
%         lung=logical(fil-logi);
%         se=strel('disk',1);
%         lung=imerode(lung,se);  
%         lun=logical(lung)+lun;
%         lungg=x.*uint8(lung);
%         lunggg=lungg;
%         if gg==0
%             gg=lungg;
%         else
%             gg(lungg>=gg)=0;
%             lungg(gg>lungg)=0;
%             gg=gg+lungg;    
%         end                      
%         ggg=ggg+lunggg;
%     end  
%     new=ggg;
%     new1=ggg;
%     lun(find(lun~=3))=0;
%     lun=logical(lun);
%     ggg=uint8(lun).*ggg;
%     ggg(find(ggg==0))=[];
%     level=graythresh(ggg);
%     new=im2bw(new,level);
%     se=strel('diamond',1);
%     new=imerode(new,se);
%     ggg15=bwareaopen(new,15);
%     new=ggg15;
%      new=imdilate(new,se);
%      if max(max(new))~=0
%         figure;imshow(final);hold on;
%         [imLL,nll]=bwlabel(new);
%         cen = regionprops(imLL,'centroid');
%         for coun = 1: numel(cen)         
%             circle(cen(coun).Centroid(1),cen(coun).Centroid(2),15); hold on;
%         end
%     end
% end

%%較慢但肺部擷取較完整
% clc;clear all;close all;
% nooo=input('幾張');
% for noooo=1:2:nooo  
% ggg=0;
% lun=0;
% gg=0;
%     for noo=noooo:1:noooo+2
%         str1=int2str(noo);
%         str2=[str1 '.jpg'];
%         A=imread(str2);
%         x=A(:,:,1);
%         x=wiener2(x);
%         if noo==noooo+1
%             final=x;
%         end
%         x=imadjust(x); 
%         level=graythresh(x);
%         bi=im2bw(x,level);
%         bi=bwareaopen(bi,500);
%         [label1,m]=bwlabel(bi);
%         max0=0;
%         for n=1:m
%                 f=find(label1==n);
%             if max0<length(f)
%                 max0=length(f);
%                 label=n;
%             end
%         end
%         label1(find(label1~=label))=0;
%         logi=logical(label1);       
%         logi=imcomplement(logi);
%         bla=bwlabel(logi);
%         bla(bla==1)=0;
%         bound=[bla(1,:),bla(end,:),bla(:,1)',bla(:,end)'];
%         bound=unique(bound);
%         [tx,ty]=size(bound);
%         for ttt=1:ty
%             bla(bla==bound(1,ttt))=0;
%         end
%         lung=logical(bla);
%         se=strel('disk',1);
%         lung=imerode(lung,se);
%         se=strel('disk',15);
%         lung=imclose(lung,se);
%         lun=logical(lung)+lun;
%         lungg=x.*uint8(logical(lung));       
%         lunggg=lungg;
%         if gg==0
%             gg=lungg;
%         else
%             gg(lungg>=gg)=0;
%             lungg(gg>lungg)=0;
%             gg=gg+lungg;    
%         end                      
%         ggg=ggg+lunggg;
%     end  
%     new=ggg;
%     new1=ggg;
%     lun(find(lun~=3))=0;
%     lun=logical(lun);
%     ggg=uint8(lun).*ggg;
%     ggg(find(ggg==0))=[];
%     level=graythresh(ggg);
%     new=im2bw(new,level);
%     se=strel('diamond',1);
%     new=imerode(new,se);
%     ggg15=bwareaopen(new,15);
%     new=ggg15;
%      if max(max(new))~=0
%         figure;imshow(final);hold on;
%         [imLL,nll]=bwlabel(new);
%         cen = regionprops(imLL,'centroid');
%         for coun = 1: numel(cen)         
%             circle(cen(coun).Centroid(1),cen(coun).Centroid(2),15); hold on;
%         end
%     end
% end


%%不做肺遮罩直接提取GGO
% clc;clear all;close all;
% nooo=input('幾張');
% for noooo=1:2:nooo   
% ggg=0;
% lun=0;
% gg=0;
%     for noo=noooo:1:noooo+2
%         str1=int2str(noo);
%         str2=[str1 '.jpg'];
%         A=imread(str2);
%         x=A(:,:,1);
%         x=wiener2(x);
%         if noo==noooo+1
%             final=x;
%         end
%         x=imadjust(x);
%         lunggg=x;
%         if gg==0
%             gg=x;
%         else
%             gg(x>=gg)=0;
%             x(gg>x)=0;
%             gg=gg+x;    
%          end                      
%         ggg=ggg+lunggg;
%     end  
%     new=ggg;
%     ggg(find(ggg==0))=[];
%     level=graythresh(ggg);
%     new=im2bw(new,level);
%     se=strel('diamond',1);
%     new=imerode(new,se);
%     ggg15=bwareaopen(new,15);
%     new=ggg15;
%      if max(max(new))~=0
%         figure;imshow(final);hold on;
%         [imLL,nll]=bwlabel(new);
%         cen = regionprops(imLL,'centroid');
%         for coun = 1: numel(cen)         
%             circle(cen(coun).Centroid(1),cen(coun).Centroid(2),15); hold on;
%         end
%     end
% end

clc;clear all;close all;
N=xlsread('ggosvmdata');
TrainingSample=N';
[N,TrainingLabel]=xlsread('ggosvmlabel');
svmStruct=svmtrain(TrainingSample,TrainingLabel);
nooo=input('幾張');
for noooo=2:2:nooo 
        str1=int2str(noooo);
        str2=[str1 '.jpg'];
        x=imread(str2);
        x=x(:,:,1);
        x=wiener2(x);
        x=imadjust(x); 
        level=graythresh(x);
        bi=im2bw(x,level);
        bi=bwareaopen(bi,500);
        [label1,m]=bwlabel(bi);
        max0=0;
        for n=1:m
                f=find(label1==n);
            if max0<length(f)
                max0=length(f);
                label=n;
            end
        end
        label1(find(label1~=label))=0;
        logi=logical(label1);       
        logi=imcomplement(logi);
        bla=bwlabel(logi);
        bla(bla==1)=0;
        bound=[bla(1,:),bla(end,:),bla(:,1)',bla(:,end)'];
        bound=unique(bound);
        [tx,ty]=size(bound);
        for ttt=1:ty
            bla(bla==bound(1,ttt))=0;
        end
        lung=logical(bla);
        se=strel('disk',1);
        lung=imerode(lung,se);
        lungg=x.*uint8(logical(lung));
        
        str1=int2str(noooo-1);
        str2=[str1 '.jpg'];
        x1=imread(str2);
        x1=x1(:,:,1);
        x1=wiener2(x1);
        x1=imadjust(x1); 
        str1=int2str(noooo+1);
        str2=[str1 '.jpg'];
        x3=imread(str2);
        x3=x3(:,:,1);
        x3=wiener2(x3);
        x3=imadjust(x3); 
        lungg1=x1.*uint8(logical(lung));
        lungg3=x3.*uint8(logical(lung));
        ggg=lungg+lungg1+lungg3;
               
            lungg(lungg1>=lungg)=0;
            lungg1(lungg>lungg1)=0;
            gg=lungg1+lungg;
            gg(lungg3>=gg)=0;
            lungg3(gg>lungg3)=0;
            gg=gg+lungg3;
                     
    new=ggg;
    ggg(find(ggg==0))=[];
    level=graythresh(ggg);
    new=im2bw(new,level);
    se=strel('diamond',1);
    new=imerode(new,se);
    ggg15=bwareaopen(new,15);
    new=ggg15;
    neww=uint8(new).*gg;
    
    [trainda,cou]=bwlabel(neww);
    long=regionprops(trainda,'Perimeter');
    Are=regionprops(trainda,'Area');
    major=regionprops(trainda,'Eccentricity');
    meann=regionprops(trainda,neww,'MeanIntensity');
    %%Eccentricity可能不需要使用，用面積除周長即可
    for k=1:cou
        P=long(k).Perimeter;
        A=Are(k).Area;
        M(k)=major(k).Eccentricity;
        Ratioo(k)=A/P;
        Meanintensiti(k)=meann(k).MeanIntensity;
    end
    
    figure;imshow(x)
    figure;imshow(neww)
end