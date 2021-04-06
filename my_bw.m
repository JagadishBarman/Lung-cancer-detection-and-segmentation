function [ out_img  ] = my_bw( img,threshold )
% �ۻs²���G�Ȥ�
% img����J���Ƕ��v��
% threshold���֭ȡA�i��graythreshold�禡�۰ʧ�(Otsu)�Φۦ��ʵ�
% out_img����X���G�ȤƼv��(����:logical)

% % test%
% img=imread('data\1145566\1145566-before\1145566 _20050923_CT_3546_11.jpg');
% img=imread('G:\code0323\data\1080322\2891298\2891298-BEFORE/2891298 _20170426_CT_401_81_08.jpg');
% figure,imshow(img),title('origin');
% 
% img=rgb2gray(img);

% img=(img-25)*3-40;
% figure,imshow(img),title('contrast enhence_before bw');
% % test %


% �֭� (0~255)
% threshold=150; % �ۦ��ʵ����֭�
% threshold=graythresh(img)*255;

[M N]=size(img);
out_img=zeros(M,N);

for i=1:1:M
    for j=1:1:N

        if img(i,j)>threshold
            img(i,j)=255;

        else
              img(i,j)=0;
        end

    end
end
out_img=logical(img);
% figure,imshow(out_img),title('bw');

end

