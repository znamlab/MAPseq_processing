
%crop images at lines where there is bad stitching into separate images
close all

lcm = imread('tubes/s008_tube56.jpg');
new = 'tubes/s008_tube56.jpg'
imshow(lcm)
%crop region that you're interested in for lcm, and another region that
%will help with alignment
a = im2gray(imcrop());
b = im2gray(imcrop());

%select points on each image corresponding to where the continous features
%are
imshow(a)
pointa = drawpoint()
positiona = pointa.Position

imshow(b)
pointb = drawpoint()
positionb = pointb.Position

%restitch based on points in centre
sizea = size(a)
sizeb = size(b)
background = zeros(2*(size(a)+size(b)));
bsize = size(background);
centrex = bsize(2)/2;
centrey = bsize(1)/2;
xmina= centrex-positiona(1)
xmaxa= sizea(2)-positiona(1)+centrex
ymina= centrey-positiona(2)
ymaxa= sizea(1)-positiona(2)+centrey
xminb= centrex-positionb(1)
xmaxb= sizeb(2)-positionb(1)+centrex
yminb= centrey-positionb(2)
ymaxb= sizeb(1)-positionb(2)+centrey


figure
ax = imshow(background);

hold on
image(a, 'XData', [xmina xmaxa],'YData', [ymina ymaxa]);
image(b, 'XData', [xminb xmaxb],'YData', [yminb ymaxb]);
saveas(gcf,new)


