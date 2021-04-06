function [ output_img ] = convert2gray( im_org )
% 檢測影像為灰階或彩色，並直接韋納濾波去雜訊

[h w d] = size(im_org);
    
    if d > 1
    % It's a true color RGB image.  We need to convert to gray scale.
output_img=wiener2(rgb2gray(im_org));
else
    % It's already gray scale.  No need to convert.
output_img=wiener2(im_org);
% output_img= BilateralFilt2(im_org,1,0.3);
    end
end

