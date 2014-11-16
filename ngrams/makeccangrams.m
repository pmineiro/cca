clear all;
close all;
more off;

addpath('..');
prefix='data/';

randn('seed',8675309);
rand('seed',90210);

tic
fprintf('loading data (takes about 10 minutes)...\n');
instruct=struct('NumHeaderLines',0, ...
                'NumColumns',3, ...
                'Format', '%f %f %f', ...
                'InfoLevel', 0);
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
fprintf('finished. ');
toc

tic
fprintf('building embedding using ALS cca (takes about 60 minutes)...\n');
cca=alscca(left,sqrt(weight),right,200,...
           struct('verbose',true,'tmax',5,'p',100,'innerloop',6));
fprintf('finished. ');
toc
tic
fprintf('saving projection to megaproj.mat ...\n');
[d,~]=size(right);
megaproj=bsxfun(@times,cca.projecty(speye(d)),sqrt(cca.sigma));
megaprojnorm=arrayfun(@(z) norm(megaproj(z,:)), 1:size(megaproj,1));
megaproj=single(bsxfun(@rdivide,megaproj,megaprojnorm'));
save('megaproj.mat','-v7.3','megaproj');
fprintf('finished. ');
toc
