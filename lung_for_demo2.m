clc,clear,close all

[filename,pathname] = uigetfile({'*.jpg';'*.*'},'please load one image');

if isequal([filename,pathname],[0,0]);
    msgbox('�Э��s��ܼv���C');
    return;
end
 
im_org_name=[pathname,filename]; % Ū�J��ϸ��|�W
im_org=imread(im_org_name);  % Ū�J���
% figure,imshow(im_org),title('original');  % ��ܭ�� %%%
% dinfo = dicominfo([(im_org_name)]); % Ū�J��ldicome��
% im_org= dicomread(dinfo);

img_org=im_org; % �n�I��~�F����

% dinfo = dicominfo([(im_org_name(1:end-3)),'dcm']); % Ū�J��ldicome��
% [Y map] = dicomread(dinfo);
% figure
% imshow(Y,'DisplayRange',[ ]);
%  spacing = dinfo.PixelSpacing;
% per_pixel_area =spacing(1) * spacing(2); % pixel ����ഫ
% slic_thick= dinfo.SliceThickness; % �����p��
spacing=0.8;
per_pixel_area =spacing *spacing;
slic_thick= 1;


img_org = convert2gray( im_org ); % �����Ƕ��èϥέ����o�i�h���T
% figure,imshow(img_org),title('move_hist'); %%%%%

[h w]=size(img_org);
[hight width]=size(img_org);
dir_name=[pathname,'*.jpg'];
list=dir(dir_name); % �C�X��Ƨ����ɦW
all_num=length(list); % �p��˥��@�h�֤���

name_all={ list.name};
[name_all_s,index] = sort_nat(name_all);
name_all=name_all_s;



im3=uint8(zeros(h,w));
im_bw3_all=[];
img_orginal=[];

for num=1:all_num
    
    im_name=[pathname,name_all_s{num}]; % Ū�J�P�˥��Ҧ����������|�ɦW
    img=imread(im_name); % Ū�J�P�˥��Ҧ������v��
% dinfo1 = dicominfo(im_name); % Ū�J��ldicome��
% img= dicomread(dinfo1 );
%  img= uint8(255 * mat2gray(img));
%     tic
    img_org_all{num}=img; % �s���Ҧ���l�Ƕ��v��
    
    img = convert2gray( img ); % �Ҧ������v����Ƕ��èϥέ����o�i�h���T
%     figure,imshow(img,[]),title('Bilateralfilt2'); % ��������o�i������v�� %%%
    
    img_orginal{num}=img;
    
    
    img_org1{num}=img; % �s���Ҧ���l�Ƕ��v��
    %
    
    img=imadjust(img);
%     figure,imshow(img),title('imadjust'); % ��ܥ[�j��������v�� %%%%%%
    
    Names(num)={im_name}; % �s���Ҧ������v�������|�ɦW
    img_adj{num}=img; % �s���Ҧ��[�j���ᤧ�����v��
    
    
%     img=adapthisteq(img);
%     figure,imshow(img),title('adapthisteq'); % ��ܦ۾A�����ϼW�j������v�� %%%%%%%%%%%%
    
    bw_img  = my_2Dotsu(img);
%     figure,imshow(bw_img),title('bw_img'); %%% %%%%
    
    %/*****�Ĥ@���аO**********/
    [im_label,n1] = bwlabel(bw_img,8); % �аOshow_X_fill�������n��
    for n11=1:n1
        l1=find(im_label==n11);
        %         [l11,l12]=find(im_label==n11);
        X1(n11)=length(l1);
    end
    [number lable]=max(X1);
    im_label(im_label~=lable)=0;
    bw_img=logical(im_label);
    figure,imshow(bw_img),title('label_1'); % ��ܲĤ@���аO���G %%%
    
    
    im_fill=imfill(bw_img,'holes');
%     figure,imshow(im_fill),title('im_fill');% ��ܪͳ������� %%%
    
    in_lung=uint8(im_fill).*img_adj{num};
%     figure,imshow(in_lung),title('in_lung'); %%%
    
    bw=in_lung;
    in_lung(in_lung==0)=[];
    [in_lung threshold]= my_2Dotsu( in_lung);%     level=graythresh(in_lung); %     bw=im2bw(bw,level);
    bw=my_bw(bw,threshold);
%     figure;imshow(bw),title('bw'); %%%
    
    im_bw2 = bwareaopen(bw, 10000,8); % �h���L�p�I(�����`�`�θ~�F)
%     figure,imshow(im_bw2),title('im_bw2');  % %%%
    out_lung=im_bw2;
%     figure,imshow(out_lung),title('out_lung');  % %%%
    
    out_lung_asq=uint8(logical(out_lung)).*img; % ��l�~�򤧼v��
        
    % �N�ͳ��ϰ�ϥեH������������
    im_fill=imfill(im_bw2,'holes'); % �ϥθɬ}�k
    im_bw2(repmat(~im_fill,1))=1; % ���Ӫͳ���(imfill)��0(��)�A
    % ��X�e���G�ȤƼv���᪺�ͳ��~����1�N���ܥ�(1)�C
    % �Y�N�I�������]��1�A�ҥHim_bw2�����`�`�B�ܬ�0(��)�A��L��1(��)
    %figure,imshow(im_fill); % 5,title('im_fill')
    % % figure,imshow(~im_fill),title('~1');
    % figure,imshow(~im_bw2),title('~2');
    % figure,imshow(im_bw2); % 6,title('~3')
    
    
    
    [M N]=size(im_bw2);
    
    im_bw3=zeros(M,N);
    
    for i=1:M
        for j=1:N
            im_bw3(i,j)=1-im_bw2(i,j);  % �N�ͳ�����(���~�F�ϰ�>���ͻP�k��)�����A�K�J�s�ϡA�äϥ�
        end
    end
%     figure,imshow(im_bw3),title('im_bw3');   % ��ܥ��ͻP�k�;B�n %%%
    
    im_bwf=im_bw3.*bw;  % �ϥ��I���A��X�ͤ������I
%     figure,imshow(im_bwf),title('img_bwf');  % %%%
          
    im_grayd=double(img);
    img_bwf3= uint8(im_bw3.*im_grayd);  % 2���B�n���W��l�Ƕ��ϡA��ܸ~�F�쥻�Ƕ���
%     figure,imshow(img_bwf3),title('img_bwf3'); %%%��ܪͤ�����T

    %% /******************************** �зǮt�G�Ȥ�**********************************************/
    pixel_labels2 =zeros(size(img_bwf3)); % �إߦs��K-means���G�ϯx�}
    img_hist=zeros(size(img_bwf3)); % �إߦs�񪽤�ϥ����ᵲ�G�ϯx�}
    
    [rowss colss]=find(img_bwf3); % ��X�Ϥ��D0�ƭȤ���Ϯy��(x�b��y�b)
    non0=find(img_bwf3); % ��X�Ϥ��D0����Ϧ�m
    XX=img_bwf3(non0);% ��X�Ϥ��D0�ƭȡA�H�Q�ᰵ����ϥ���
%     figure,imhist(img_bwf3(non0)),xlabel('�Ƕ���'), ylabel('����');
    
    mean_inlung=mean(XX);
    std_inlung=std2(XX);
    mean_std{num,1}=mean_inlung;
    mean_std{num,2}=std_inlung;
    
    % /***����ϥ����k***/
    % (�u�w��ͤ��������k2��)
    
    mean_img=mean(XX); % �p�⥭����
    for LL=1:length(non0)
        
        img_hist(non0(LL))=(img_bwf3(non0(LL))+(90-mean_img)); % �i�檽��ϥ���
        
    end
    img_hist=uint8(img_hist);
%     figure,imshow(img_hist),title('move_hist'); % ��ܪ���ϥ������G
%     figure,imhist(img_hist(non0)),xlabel('�Ƕ���'), ylabel('����');
    
%     img_hist_inout=out_lung_asq+img_hist;
% %     figure,imshow( img_hist_inout),title('����j�Ƥ��ͳ�');
img_hist_inout=img_hist;
    img_hist_all{num}=img_hist_inout ; % �s���Ҧ�����ϥ������G�Ϧ�cell
    
    non0=find(img_hist); % ��X�Ϥ��D0����Ϧ�m
    XX=img_hist(non0);
    
    mean_inlung2=mean(XX);
    std_inlung2=std2(XX);
    mean_std2{num,1}=mean_inlung2;
    mean_std2{num,2}=std_inlung2;
    threshold=255-(mean_inlung2+(std_inlung2*2));
    img_bw_std= my_bw( img_hist,threshold );
%     figure,imshow(img_bw_std),title('�зǮt���G�Ȥ�');
    BW2 = bwareaopen(img_bw_std,10);
%     BW2_all{num}=BW2;
% figure,imshow(BW2)
  
    im_bw3_all{num}=BW2; % �s���Ҧ��G�ȤƹϦ�cell
    im_bw3_all2{num}=double(BW2);
end

%% �T������


for num1=1:1:all_num
    llll=['C:\Users\User\Desktop\�פ�{���X\final_code\sample\3d\5\mask\mask',num2str(num1),'.jpg'];

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