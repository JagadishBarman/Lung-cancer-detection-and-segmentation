function [ diameters ] = all_diameter( position )
% 計算二值影像上所有長度
%  position 為二值影像上白點的所有座標矩陣
% diameters 為存放所有長度之矩陣


l=length(position);

for tt=1:1:l
for nn=1:1:l
 
diameters(tt,nn)=(sqrt(((position(nn,1)-position(tt,1)).^2)+((position(nn,2)-position(tt,2)).^2)));

end
end


end

