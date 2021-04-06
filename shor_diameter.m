function sd = shor_diameter(LR,ps,RL)

c = regionprops(LR,'Centroid');
c = c.Centroid;
% polar args
t = linspace(0,2*pi,361);
t(end) = [];
r = 0:ceil(sqrt(numel(LR)/4));
[tg,rg] = meshgrid(t,r);
[xg,yg] = pol2cart(tg,rg);
xoff = xg + c(1);
yoff = yg + c(2);
% polar image
pbw = interp2(double(LR),xoff,yoff,'nearest') == 1;
[~,radlen] = min(pbw,[],1);
radlen(radlen == 1) = max(r);
n = numel(radlen);
% add two edges of line to form diameter
diamlen = radlen(1:n/2) + radlen(n/2+1:n);
% find min diameter
[mindiam,tminidx1] = min(diamlen);
tmin = t(tminidx1);
rmin = radlen(tminidx1);
tminidx2 = tminidx1 + n/2;
xx = [xoff(radlen(tminidx1),tminidx1) xoff(radlen(tminidx2),tminidx2)];
yy = [yoff(radlen(tminidx1),tminidx1) yoff(radlen(tminidx2),tminidx2)];
% viz
figure(find(RL==max(RL)));
hold on
plot(xx,yy,'g')

sd = sqrt(((xx(1)-xx(2))*ps(1))^2+((yy(1)-yy(2))*ps(2))^2);