function [ position ] =bwposition( img )
% �ۻsŪ���G�ȤƼv���զ��I���y��
% out_position�� n*2 ���s��y�Яx�}
% img��Ū�J���G�ȤƼv��

%% test%%%
% clc,clear,close all
% [filename,pathname] = uigetfile({'*.jpg';'*.*'},'please load one image');
% 
% if isequal([filename,pathname],[0,0]);
%     msgbox('�Э��s��ܼv���C');
%     return;
% end
% 
% im_org_name=[pathname,filename]; % Ū�J��ϸ��|�W
% img=logical(imread(im_org_name));  % Ū�J���
% figure,imshow(img),title('original');  % ��ܭ�� 


%%

[nr nc]=size(img);
p_l=sum(sum(img));
position=zeros(p_l,2);

% for p_l1=1:p_l
%     for p_x=1:nr
%         for p_y=1:nc
            
            [p_x1,p_y1]=find(img);
            
            position(:,1)=p_x1;
             position(:,2)=p_y1;
%         end
%     end
% end

end
