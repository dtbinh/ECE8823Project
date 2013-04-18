function []=displayFace(y,h,w, fig)
if nargin ==3
    fig = 100;
end

y = reshape(y,h,w);
figure(fig);clf;
imagesc(y)
colormap gray