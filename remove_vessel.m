function [ ves_mask, pixel_labels3] = remove_vessel( pixel_labels2,diameters,diameter_th,position,x_seed,y_seed )
% 去除血管或過抓的函式
% pixel_labels２為二值且有血管連接或過抓之影像。
% detect=stats.Solidity，detect須小於0.9。
% diameters 為影像中所有長徑
% diameter_th 為血管連接之閥值
% ves_mask 為血管連接之遮罩
% pixel_labels3 為輸出之去除血管或過抓的影像


        [ convex_p1 convex_p2]=find(diameters>  diameter_th); % 找出大於閥值者之座標
        
        img_thin1=pixel_labels2;
        
        % 去除大於閥值者之座標（將其變０），以得到腫瘤與血管連接之遮罩
        for iiiii=1:length(convex_p1)
            
            img_thin1(position(convex_p1(iiiii),1),position(convex_p1(iiiii),2))=0;
            img_thin1(position(convex_p2(iiiii),1),position(convex_p2(iiiii),2))=0;
        end
%         figure,imshow( img_thin1),title(' img thin1');
        
        img_dilate=zeros(size(img_thin1));

         img_dilate=img_thin1;
        ves_mask=img_dilate.*pixel_labels2; % 得到腫瘤與血管連接之遮罩
        figure,imshow( ves_mask),title('  ves_mask');
        
        tumor_d=pixel_labels2- ves_mask; % 去除血管連接部分，腫瘤與血管分離
        tumor_locarion=zeros(size( tumor_d));
        pixel_labels3=zeros(size( tumor_d));

        [label3 number3]=bwlabel( tumor_d,4);
        
 % 標記出腫瘤，得到無血管連接之腫瘤
        for lab3=1:number3
            
            if label3(x_seed,y_seed)==lab3
                p_labels3=find(label3(:)==lab3);

pixel_labels3( p_labels3)=1;
            end
        end
  figure,imshow(pixel_labels3),title(' pixel_labels3');

end



