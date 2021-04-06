function [xp,yp] = shortdiameter(img )
 
% IC =edge(img,'Canny');
% IC = bwareaopen(IC,5);
IC=img;
IC = im2uint8(IC);
[y,x] = find(IC>1);
%  
    maxDistance = -inf;
    for k = 1 : length(x)
        distances = sqrt( (x(k) - x) .^ 2 + (y(k) - y) .^ 2 );
        [thisMaxDistance, indexOfMaxDistance] = max(distances);
        if thisMaxDistance > maxDistance
            maxDistance = thisMaxDistance;
            index1 = k;
            index2 = indexOfMaxDistance;
        end
    end

    longSlope = (y(index1) - y(index2)) / (x(index1) - x(index2));
    perpendicularSlope = -1/longSlope;
    
    for w = 1 : length(x)
        Slope =   (y(w) - y) ./ (x(w) - x);
        %Correspond = find(Slope == perpendicularSlope);
        [~, Correspond] = min(abs(Slope-perpendicularSlope));
         
        if  isempty(Correspond)
            continue;
             
        elseif  length(Correspond)>1
            Sp(w) = w;
            Z = sqrt(( x(w) - x(Correspond) ) .^2 + ( y(w) - y(Correspond) ) .^2);
            Cp(w) = Correspond(find(max(sqrt(( x(w) - x(Correspond) ) .^2 + ( y(w) - y(Correspond) ) .^2))));
             
        else
            Sp(w) = w;
            Cp(w) = Correspond;
             
        end
    end
     
Sp = Sp(find(Sp>0));
Cp = Cp(find(Cp>0));
Pd = sqrt((x(Sp)-x(Cp)).^2 + (y(Sp)-y(Cp)).^2);
%ShortAxis = max(Pd);
l = find(Pd==max(Pd));
Index1 = Sp(l(1));
Index2 = Cp(l(1));

% figure(MaxDistanceL)
% line([x(index1), x(index2)], [y(index1), y(index2)], 'Color', 'g', 'LineWidth', 1);
%  
% figure(MaxDistanceL)
% line([x(Index1), x(Index2)], [y(Index1), y(Index2)], 'Color', 'y', 'LineWidth', 1);

xp = [x(Index1) x(Index2)];
yp = [y(Index1) y(Index2)];