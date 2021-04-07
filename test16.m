%%3張影像相加
clc;clear all;close all;
nooo=input('幾張');
for noooo=1:2:nooo    
ggg=0;
lun=0;
gg=0;
    for noo=noooo:1:noooo+2
        str1=int2str(noo);
        str2=[str1 '.jpg'];
        A=imread(str2);
        x=rgb2gray(A);
        x=wiener2(x);
        if noo==noooo+1
            final=x;
        end
        [yy,xx]=size(x);
        if min(x)~=0
            mini=min(min(x));
            x=uint8(double(x-mini*uint8(ones(yy,xx)))*double(255)/double((255-mini)));
        end 
        level=graythresh(x);
        bi=im2bw(x,level);
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
        label1(find(label1~=0))=1;
        logi=logical(label1);
        se=strel('diamond',3);
        logi=imdilate(logi,se);
        fil=logical(imfill(label1,'holes'));
        lung=uint8(fil-logi);
        lun=logical(lung)+lun;
        lungg=x.*lung;

        if gg==0
            gg=lungg;
        else
            for yyy=1:yy
                for xxx=1:xx                   
                    if  gg(yyy,xxx)<lungg(yyy,xxx)
                        gg(yyy,xxx)=lungg(yyy,xxx);
                    end    
                end
            end    
        end
                       
        ggg=ggg+lungg;
    end  
    new=ggg;
    new1=ggg;
    lun(find(lun~=3))=0;
    lun=logical(lun);
    ggg=uint8(lun).*ggg;
    ggg(find(ggg==0))=[];
    level=graythresh(ggg);
    new=im2bw(new,level);
%          new=bwareaopen(new,10);
    se=strel('diamond',1);
    new=imerode(new,se);
    ggg15=bwareaopen(new,15);
    new=ggg15;
     new=imdilate(new,se);
%     
%     rrr=uint8(logical(new)).*gg;
%     figure;imshow(rrr)
    
%     [new,m]=bwlabel(new);
%     major=regionprops(new,'Eccentricity');
%     majorr=cell2mat(struct2cell(major));
%     for n=1:m
%         if majorr(n) > 0.7
%         new(find(new==n))=0;
%         end
%     end
%     
%      %計算群內灰階平均
%     [imL,nl] = bwlabel(new);
%     if nl~=0       
%         for n=1:nl
%             objvals = gg(imL==n);
%             objave(n) = mean(objvals(:));
%             if  objave(n) > 170
%                 new(imL==n)=0;
%             end  
%         end
%     end
    if max(max(new))~=0
        figure;imshow(final);hold on;
        [imLL,nll]=bwlabel(new);
        cen = regionprops(imLL,'centroid');
        for coun = 1: numel(cen)         
            circle(cen(coun).Centroid(1),cen(coun).Centroid(2),15); hold on;
        end
    end
end