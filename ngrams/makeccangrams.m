clear all;
close all;
more off;

addpath('../');
prefix='data/';

randn('seed',8675309);
rand('seed',90210);

tic
instruct=struct('NumHeaderLines',0, ...
                'NumColumns',3, ...
                'Format', '%f %f %f', ...
                'InfoLevel', 1);
leftraw=txt2mat(strcat(prefix,'left'),instruct);
left=spconvert(leftraw);
clear leftraw;
left=left';
rightraw=txt2mat(strcat(prefix,'right'),instruct);
right=spconvert(rightraw);
clear rightraw;
right=right';
weightraw=txt2mat(strcat(prefix,'weight'),instruct);
weight=full(spconvert(weightraw));
clear weightraw;
toc

tic
cca=alscca(left,sqrt(weight),right,200,...
           struct('tmax',10,'verbose',true,'bs',1e+7,'innerloop',4));
toc
tic
[d,~]=size(right);
megaproj=bsxfun(@times,cca.projecty(speye(d)),sqrt(cca.sigma));
megaprojnorm=arrayfun(@(z) norm(megaproj(z,:)), 1:size(megaproj,1));
megaproj=single(bsxfun(@rdivide,megaproj,megaprojnorm'));
save('megaproj.mat','-v7.3','megaproj');
toc
