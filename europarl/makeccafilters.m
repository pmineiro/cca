clear all;
close all;
more off;

addpath('..');

randn('seed',8675309);
rand('seed',90210);

tic
fprintf('loading data...');
c=2;
data=cell(1,c);
instruct=struct('NumHeaderLines',0, ...
                'NumColumns',3, ...
                'Format', '%f %f %f', ...
                'InfoLevel', 0);
europarl_v7_el_en_el=txt2mat('europarl-v7.el-en.el.mat',instruct);
data{1}=spconvert(europarl_v7_el_en_el);
clear europarl_v7_el_en_el;
europarl_v7_el_en_en=txt2mat('europarl-v7.el-en.en.mat',instruct);
data{2}=spconvert(europarl_v7_el_en_en);
clear europarl_v7_el_en_en;
fprintf(' finished. ');
toc

perm=randperm(size(data{1},1));
split=ceil(0.9*length(perm));
train=sort(perm(1:split));
test=sort(perm(split+1:end));

tic
fprintf('building multilingual embedding with randomized cca...\n');
cca = rcca(data{1}(train,:)',ones(1,length(train)),data{2}(train,:)',60, ...
           struct('compress',true,'tmax',3,'p',2000,'kbs',100,'lambda',0.01));
fprintf('finished. ');
toc
%disp(sum(cca.sigma))
trainprox=cca.projectx(data{1}(train,:));
trainproy=cca.projecty(data{2}(train,:));
trainsigma=trainprox'*trainproy/length(train);
testprox=cca.projectx(data{1}(test,:));
testproy=cca.projecty(data{2}(test,:));
testsigma=(testprox'*testproy)/length(test);
fprintf('training sum of canonical correlations: %g\n',sum(diag(trainsigma)));
fprintf('test sum of canonical correlations: %g\n',sum(diag(testsigma)));
