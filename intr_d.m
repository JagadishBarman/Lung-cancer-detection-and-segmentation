function [ d ] = intr_d( slic_thick,spacing,img_big,img_big_n, vol )
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
im_bw=im2bw(img_big);
% figure,imshow(im_bw);
stats = regionprops(im_bw,'MajorAxisLength'	);

d=stats.MajorAxisLength;
% im_r=((d+32.2)*slic_thick)/2;
im_r=(d*spacing)/2;
A= vol;
[img_big_val img_big_n]=max(A);
loss_im_u=length(A(1: img_big_n))*slic_thick;
l_uh=(im_r-loss_im_u);

if l_uh<0 | l_uh>=slic_thick
    l_uh=0;
end
loss_im_d=length(A( img_big_n:end))*slic_thick;
l_dh=(im_r-loss_im_d);

if l_dh<0 | l_dh>=slic_thick
    l_dh=0;
end

n=length(A);
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

An=((sum(A(1:n-1))*(slic_thick/m))+(sum(Acc(:))*(slic_thick/m))+(A(n)*slic_thick))/1000;
Axx(end)=A(end);
% An=sum(Axx)/1000;
% 
%     Ac22=A(1)*1/100;
% 
% Ann=An+(Ac22)*l_uh+(Ac22)*l_dh;
A_org=sum(A)/1000;
A_actrul=1*1*1*(4/3)*pi;
%for i=1:n-1
%     for a=1:m
%         A_b(i)=((a*A(i)+(m-a)*A(i+1))/m);
%     end
%     Vm(i)=(A(i)+A_b(i))*(slic_thick)/2;
% end
% V_a=sum(V_m)/1000

% for i=1:n-1
%        for a=1:m
%     V1=A(i)+A(i+1)+(a*A(i)+(m-a+1)*A(i+1))/m)*(slic_thick/m));
%        end
% end

% Ax=zeros(1,n);
% Ax(1:n-1)=Acc(1:end);
% Ax(end)=A(end);

%%
x = 1:length(A); 
y = A; 
[p,S] = polyfit(x,y,5); 

[y_fit,delta] = polyval(p,x,S);
% plot(x,y,'bo')
% hold on
% plot(x,y_fit,'r-')
% % plot(x,y_fit+2*delta,'m--',x,y_fit-2*delta,'m--')
% title('Linear Fit of Data ')
% % legend('Data','Linear Fit','95% Prediction Interval')
% xlabel('Slice No.'), ylabel('Area')%,'FontWeight','bold'
% xx=(1/m)*l_uh;
% xx2=n+(1/m)*l_dh;
xx=l_uh;
xx2=n+l_dh;

A_u=p(1)*xx^5+p(2)*xx^4+p(3)*xx^3+p(4)*xx^2+p(5)*xx+p(6);
A_d=p(1)*(xx2)^5+p(2)*(xx2)^4+p(3)*(xx2)^3+p(4)*(xx2)^2+p(5)*(xx2)+p(6);

% A_u=p(1)*xx^3+p(2)*xx^2+p(3)*xx+p(4);
% A_d=p(1)*(xx2)^3+p(2)*(xx2)^2+p(3)*(xx2)+p(4);

Ann2=(An*1000+A_u*l_uh+A_d*l_dh)/1000;

error=(A_actrul-Ann2)/A_actrul;



for aa=1:m-1
    Acx(aa)=(A_u*(m-aa)+A(1)*(aa))/m;
    
end
A_up=A_u+sum(Acx);

for aa=1:m-1
    Acx2(aa)=(A(end)*aa+A_d*(m-aa))/m;
    
end
A_down=A_d+sum(Acx2);

if l_uh==0 | l_uh<0
   A_up=0;
end

if l_dh==0 | l_dh<0
   A_down=0;
end
% Ann3=(An*1000+A_up*(l_uh/m)+A_down*(l_dh/m))/1000;
Ann3=((sum(Axx)*(1/m))+A_up*(1/m)+A_down*(1/m))/1000;

error2=(A_actrul-Ann3)/A_actrul;

end

