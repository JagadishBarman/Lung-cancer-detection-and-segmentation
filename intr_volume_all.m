function [ Axx ] = intr_volume_all( slic_thick,spacing,d, vol )
% 內差法+曲線擬和>>誤差:0.35%

% %%test
% clc,clear all ,close all
% 
% 
% % dinfo = dicominfo('C:\Users\User\Desktop\論文程式碼\check\實心\1071130_20181130_CT_6_1_37.dcm');
% dinfo = dicominfo('F:\論文程式碼\check\實心\1071130_20181130_CT_6_1_37.dcm');
% slic_thick = dinfo.SliceThickness;
% spacing=dinfo.PixelSpacing(1);
% %*dinfo.PixelSpacing(2);
% % img_big=imread('C:\Users\User\Desktop\論文程式碼\img_b.jpg');
% img_big=imread('F:\論文程式碼\1080101\img_b.jpg');

% % A=[80.8790170486500,123.079710198825,165.483616959675,206.126339094675,234.508506718950,256.726528152750,273.119092747200,284.973220036575,292.966288723125,286.124763830400,255.845935839825,217.032136200900,227.260554604875,199.691241423300,166.431947142825,120.912098351625,77.1534341862750,36.9848771428500,8.19628229722500];
% A=[113.079017048650,155.279710198825,197.683616959675,238.326339094675,266.708506718950,288.926528152750,305.319092747200,317.173220036575,325.166288723125,318.324763830400,288.045935839825,249.232136200900,259.460554604875,231.891241423300,198.631947142825,153.112098351625,109.353434186275,69.1848771428500,40.3962822972250];
% % A=[1,4,2];

%%
% im_bw=im2bw(img_big);
% % figure,imshow(im_bw);
% stats = regionprops(im_bw,'MajorAxisLength'	);
% 
% d=stats.MajorAxisLength;
% im_r=((d+32.2)*slic_thick)/2;
im_r=(d)/2;
A= vol;
n=length(find(A));
[img_big_val img_big_n]=max(A);
% loss_im_u=length(A(1: img_big_n))*slic_thick;
loss_im_u= img_big_n*slic_thick;
l_uh=(im_r-loss_im_u);

if l_uh<0 | l_uh>=slic_thick
    l_uh=0;
end
loss_im_d=(n-img_big_n)*slic_thick;
l_dh=(im_r-loss_im_d);

if l_dh<0 | l_dh>=slic_thick
    l_dh=0;
end

% n=length(A);
% m=100000000; %>>0.48%
m=6;%>>0.35%
Axx=zeros(1,(n-1)*(m)+1);
for i=1:n-1
for a=1:m-1
    Ac(a)=(A(i+1)*a+A(i)*(m-a))/m;
    
end
Acc(i)=sum(Ac);
ou=i*m-(m-1);   
%  ou1=fix(ou/m)+1;
Axx(ou)=A(i);
    Axx(ou+1:ou+m-1)=Ac;
% 
end

An=((sum(A(1:n-1))*(slic_thick/m))+(sum(Acc(:))*(slic_thick/m))+(A(n)*slic_thick));
Axx(end)=A(end);
end