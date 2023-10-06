
%align lcm brain images to brain images used for registration

files = dir('tubes/s001_*.jpg')
rois = dir ('rois/')

for k = 1(files)
    
original = imread('sections/brain1_s001.jpg');
original = rgb2gray(original)

toread = files(k).name;
lcm1 = strcat('tubes/', toread)
lcm = imread(lcm1);
lcm = im2gray(lcm)

ptsOriginal  = detectSURFFeatures(original);
ptsLcm = detectSURFFeatures(lcm);

[featuresOriginal,  validPtsOriginal]  = extractFeatures(original,  ptsOriginal);
[featuresLcm, validPtsLcm] = extractFeatures(lcm, ptsLcm);

indexPairs = matchFeatures(featuresOriginal, featuresLcm);

matchedOriginal  = validPtsOriginal(indexPairs(:,1));
matchedLcm = validPtsLcm(indexPairs(:,2));

figure;
showMatchedFeatures(original,lcm,matchedOriginal,matchedLcm);

[tform, inlierIdx] = estimateGeometricTransform2D(...
    matchedLcm, matchedOriginal, 'similarity');
inlierLcm = matchedLcm(inlierIdx, :);
inlierOriginal  = matchedOriginal(inlierIdx, :);

figure;
showMatchedFeatures(original,lcm,inlierOriginal,inlierLcm);
title('Matching points (inliers only)');
legend('ptsOriginal','ptsLcm');

Tinv  = tform.invert.T;

ss = Tinv(2,1);
sc = Tinv(1,1);
scaleRecovered = sqrt(ss*ss + sc*sc)
thetaRecovered = atan2(ss,sc)*180/pi


outputView = imref2d(size(original));
recovered  = imwarp(lcm,tform,'OutputView',outputView);
[filepath,name,ext] = fileparts(toread)
savelcm = strcat('registered_lcm/', name, '.jpg')
imwrite (recovered, savelcm)

figure;
imshowpair(original,recovered,'montage')

%load image for ROI selection for tube
C = imfuse(original,recovered,'falsecolor','Scaling','joint','ColorChannels',[2 1 0]);


imshow(C);
roi = drawpolygon('Color','r');
BW = createMask(roi);
imshow(BW)

tosave = strcat('rois/', name, '.png')
imwrite (BW, tosave)
recovered1 = recovered
end
%now do for loop using previous registered LCM image as comparison for ROI
%selection
for k = 2: length(files)
reg = dir ('registered_lcm/s001_*')    
original = imread('sections/brain1_s001.jpg');
original = rgb2gray(original)

toread = files(k).name;
lcm1 = strcat('tubes/', toread)
lcm = imread(lcm1);
lcm = im2gray(lcm)

ptsOriginal  = detectSURFFeatures(original);
ptsLcm = detectSURFFeatures(lcm);

[featuresOriginal,  validPtsOriginal]  = extractFeatures(original,  ptsOriginal);
[featuresLcm, validPtsLcm] = extractFeatures(lcm, ptsLcm);

indexPairs = matchFeatures(featuresOriginal, featuresLcm);

matchedOriginal  = validPtsOriginal(indexPairs(:,1));
matchedLcm = validPtsLcm(indexPairs(:,2));

figure;
showMatchedFeatures(original,lcm,matchedOriginal,matchedLcm);

[tform, inlierIdx] = estimateGeometricTransform2D(...
    matchedLcm, matchedOriginal, 'similarity');
inlierLcm = matchedLcm(inlierIdx, :);
inlierOriginal  = matchedOriginal(inlierIdx, :);

figure;
showMatchedFeatures(original,lcm,inlierOriginal,inlierLcm);
title('Matching points (inliers only)');
legend('ptsOriginal','ptsLcm');

Tinv  = tform.invert.T;

ss = Tinv(2,1);
sc = Tinv(1,1);
scaleRecovered = sqrt(ss*ss + sc*sc)
thetaRecovered = atan2(ss,sc)*180/pi


outputView = imref2d(size(original));
recovered  = imwarp(lcm,tform,'OutputView',outputView);
[filepath,name,ext] = fileparts(toread)
savelcm = strcat('registered_lcm/', name, '.jpg')
imwrite (recovered, savelcm)

figure;
imshowpair(original,recovered,'montage')


%load image for ROI selection for tube
C = imfuse(recovered1,recovered,'falsecolor','Scaling','joint','ColorChannels',[2 1 0]);


imshow(C);
roi = drawpolygon('Color','r');
BW = createMask(roi);
imshow(BW)

tosave = strcat('rois/', name, '.png')
imwrite (BW, tosave)
recovered1 = recovered
end

