function win_data = MAP_Value(H, W, map_fname)
% 在HxW的局部窗口内展示权值，并且给出局部窗口叠加在全图上的效果
% 地图文件名默认为 occupancy_map_0.txt
if nargin<1, H = 31; end
if nargin<2, W = 31; end
if nargin<3
    map_fname = 'occupancy_map_0.txt';
end

win_X = zeros(H,W);
win_Y = zeros(H,W);
win_data = zeros(H,W);

cx = fix(H/2); cy = fix(W/2);

sigma_1 = 3;  % 局部窗

ind_i = 1:H;  ind_j = 1:W;
for i = ind_i
    for j = ind_j
        x = i-cx;
        y = j-cy;
        a = x/abs(x);
        b = y/abs(y);
        L = norm([x,y]);
        theta = atan(x/y);
        if (a<=0 && b<=0) || (a>=0 && b<=0) || (x==0 && y<0)
            win_X(i,j) = -exp(-L/sigma_1)*cos(theta);
            win_Y(i,j) = -exp(-L/sigma_1)*sin(theta);            
        else
            win_X(i,j) = exp(-L/sigma_1)*cos(theta);
            win_Y(i,j) = exp(-L/sigma_1)*sin(theta);
        end
        win_data(i,j) = norm([win_X(i,j), win_Y(i,j)]);
    end
end

win_X(cx,cy) = 0; win_Y(cx,cy) = 0;


figure(1), imshow(mat2gray(win_data))

[x,y] = meshgrid(ind_i-cx,ind_j-cy);
figure(2), quiver(x,y, win_X, win_Y), axis([-cx,cx,-cy,cy]);
title('局部窗口向量场')

Map_data = load(map_fname);
[L1, L2]=size(Map_data);

Map_X = conv2(Map_data,win_X,'same');
Map_Y = conv2(Map_data,win_Y,'same');

figure(30),imshow(mat2gray(sqrt(Map_X.^2 + Map_Y.^2))),title('向量场的大小')

[XX, YY] = meshgrid(1:L1,1:L2);
figure(20), quiver(XX, YY, Map_X, Map_Y),title('向量场') 
figure(11),imshow(mat2gray(Map_X)),title('X分量')
figure(12),imshow(mat2gray(Map_Y)),title('Y分量')




