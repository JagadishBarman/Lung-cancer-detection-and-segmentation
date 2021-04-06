function [ ves_mask, pixel_labels3] = remove_vessel( pixel_labels2,diameters,diameter_th,position,x_seed,y_seed )
% �h����ީιL�쪺�禡
% pixel_labels�����G�ȥB����޳s���ιL�줧�v���C
% detect=stats.Solidity�Adetect���p��0.9�C
% diameters ���v�����Ҧ����|
% diameter_th ����޳s�����֭�
% ves_mask ����޳s�����B�n
% pixel_labels3 ����X���h����ީιL�쪺�v��


        [ convex_p1 convex_p2]=find(diameters>  diameter_th); % ��X�j��֭Ȫ̤��y��
        
        img_thin1=pixel_labels2;
        
        % �h���j��֭Ȫ̤��y�С]�N���ܢ��^�A�H�o��~�F�P��޳s�����B�n
        for iiiii=1:length(convex_p1)
            
            img_thin1(position(convex_p1(iiiii),1),position(convex_p1(iiiii),2))=0;
            img_thin1(position(convex_p2(iiiii),1),position(convex_p2(iiiii),2))=0;
        end
%         figure,imshow( img_thin1),title(' img thin1');
        
        img_dilate=zeros(size(img_thin1));

         img_dilate=img_thin1;
        ves_mask=img_dilate.*pixel_labels2; % �o��~�F�P��޳s�����B�n
        figure,imshow( ves_mask),title('  ves_mask');
        
        tumor_d=pixel_labels2- ves_mask; % �h����޳s�������A�~�F�P��ޤ���
        tumor_locarion=zeros(size( tumor_d));
        pixel_labels3=zeros(size( tumor_d));

        [label3 number3]=bwlabel( tumor_d,4);
        
 % �аO�X�~�F�A�o��L��޳s�����~�F
        for lab3=1:number3
            
            if label3(x_seed,y_seed)==lab3
                p_labels3=find(label3(:)==lab3);

pixel_labels3( p_labels3)=1;
            end
        end
  figure,imshow(pixel_labels3),title(' pixel_labels3');

end



