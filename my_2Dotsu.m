function [ out_img threshold ] = my_2Dotsu( img )
% �G��Otsu�k�A�N�Ŷ��T���[�J�@��otsu�k�������C
% �s�@�G���Ȥ�ϡA�H�F�쥭���ȤΦǶ��ȥX�{���v�s�@�C
% �N�Ƕ��v��(0~255)���ΥX�e��(1)�A�I��(0)���G�ȤƼv���C
% img����J��
% out_img����X��
% �ϥ�my_bw�禡�N�v���ܤG�Ȥ�

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


[m n]=size(img); % Ū���v�����e
% hist=zeros(256,1); % �s��Ƕ��v�����C�ӹ���(1~256)�X������
r=1;    %�F��b�|

imgn=zeros(m+2*r+1,n+2*r+1);
imgn(r+1:m+r,r+1:n+r)=img;

imgn(1:r,r+1:n+r)=img(1:r,1:n);                 %�W
imgn(1:m+r,n+r+1:n+2*r+1)=imgn(1:m+r,n:n+r);    %�k
imgn(m+r+1:m+2*r+1,r+1:n+2*r+1)=imgn(m:m+r,r+1:n+2*r+1);    %�U
imgn(1:m+2*r+1,1:r)=imgn(1:m+2*r+1,r+1:2*r);       %��

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
h=h1./(m*n); % �p��C�ӹ����X�{���v

var=zeros(256,1); % �s���ܲ��Ƥ��x�}
for t=1:256
    w0=sum(h(1:t)); % �p��Ĥ@���]�I���^�b�v�����X�{���v 
    w1=sum(h(t+1:end));% �p��ĤG���]�e���^�b�v�����X�{���v 
    u0=sum(h(1:t).*(1:t)')/w0;% �p��Ĥ@�������� 
    u1=sum(h(t+1:end).*(t+1:256)')/w1;% �p��ĤG�������� 
    var(t)=w0*w1*((u1-u0).^2); %�p��������ܲ��� 
end

[value threshold]=max(var(:)); % �p��������̤j�ܲ��ƨD�X�̤j�֭�

out_img=my_bw(img,threshold);
% figure,imshow(out_img),title('bw');

end




