function [ out_img threshold ] = my_2Dotsu( img )
% 二維Otsu法，將空間訊息加入一維otsu法的概念。
% 製作二維值方圖，以鄰域平均值及灰階值出現機率製作。
% 將灰階影像(0~255)分割出前景(1)，背景(0)之二值化影像。
% img為輸入圖
% out_img為輸出圖
% 使用my_bw函式將影像變二值化

%/*****teat*****/
% clc,clear,close all
% % img=imread('data\1145566\1145566-before\1145566 _20050923_CT_3546_11.jpg');
% % img=imread('data\2284072\2284072-before\2284072 _20090123_CT_1005_47_10.jpg');
% img=imread('data\2916451\2916451-before\2916451 _20170131_CT_6_30.jpg');
% figure,imshow(img),title('original');

% img=rgb2gray(img);
% figure,imshow(img),title('gray');
% img=wiener2(img);
% img=adapthisteq(img);
% img=255-img;
% figure,imshow(img),title('c');
%/***************/


[m n]=size(img); % 讀取影像長寬
% hist=zeros(256,1); % 存放灰階影像中每個像素(1~256)出限次數
r=1;    %鄰域半徑

imgn=zeros(m+2*r+1,n+2*r+1);
imgn(r+1:m+r,r+1:n+r)=img;

imgn(1:r,r+1:n+r)=img(1:r,1:n);                 %上
imgn(1:m+r,n+r+1:n+2*r+1)=imgn(1:m+r,n:n+r);    %右
imgn(m+r+1:m+2*r+1,r+1:n+2*r+1)=imgn(m:m+r,r+1:n+2*r+1);    %下
imgn(1:m+2*r+1,1:r)=imgn(1:m+2*r+1,r+1:2*r);       %左

hist=zeros(256,256);
for i=1+r:r+m
    for j=1+r:r+n
        pix1=uint8(imgn(i,j));
        pix2=uint8(mean2(imgn(i-r:i+r,j-r:j+r)));
        hist(pix1+1,pix2+1)=hist(pix1+1,pix2+1)+1;           
    end
end
% figure,
% b1=mesh(double(hist));
% set(b1,'FaceColor','b','EdgeColor','w');
h1=zeros(256,1);
for c=1:256
    h1(c)=hist(c,c);
end
h=h1./(m*n); % 計算每個像素出現機率

var=zeros(256,1); % 存放變異數之矩陣
for t=1:256
    w0=sum(h(1:t)); % 計算第一類（背景）在影像中出現機率 
    w1=sum(h(t+1:end));% 計算第二類（前景）在影像中出現機率 
    u0=sum(h(1:t).*(1:t)')/w0;% 計算第一類平均值 
    u1=sum(h(t+1:end).*(t+1:256)')/w1;% 計算第二類平均值 
    var(t)=w0*w1*((u1-u0).^2); %計算兩類間變異數 
end

[value threshold]=max(var(:)); % 計算兩類間最大變異數求出最大閥值

out_img=my_bw(img,threshold);
% figure,imshow(out_img),title('bw');

end




