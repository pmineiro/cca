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

% for tmax=2:2
%   lambdas=[0.3594];
%   %lambdas=logspace(log(0.01)/log(10),log(1)/log(10),20);
%   deltat=[];
%   sumsigma=[];
%   sumtrainsigma=[];
%   sumtestsigma=[];
%   for lambda=lambdas
%     randn('seed',8675309);
%     rand('seed',90210);
%     start=tic;
%     [x,xb,y,yb,sigma] = rcca(data{1}(train,:),data{2}(train,:),60, ...
%                              struct('tmax',4,'innerloop',3,'init','rcca2','lambda',lambda,'rcca2p',1000,'rcca2lambda',0.00001,'verbose',true));
% 
% %                             struct('tmax',15,'innerloop',3,'init','randn','lambda',lambda));
% 
% %    [x,xb,y,yb,sigma] = rcca2(data{1}(train,:),data{2}(train,:),60,struct('tmax',tmax,'p',2000,'lambda',lambda));
%     deltat=[deltat toc(start)];
%     sumsigma=[sumsigma sum(sigma)];
%     trainprox=bsxfun(@minus,data{1}(train,:)*x,xb);
%     trainproy=bsxfun(@minus,data{2}(train,:)*y,yb);
%     trainsigma=trainprox'*trainproy;
%     testprox=bsxfun(@minus,data{1}(test,:)*x,xb);
%     testproy=bsxfun(@minus,data{2}(test,:)*y,yb);
%     testsigma=(length(train)/length(test))*(testprox'*testproy);
%     sumtrainsigma=[sumtrainsigma sum(diag(trainsigma))];
%     sumtestsigma=[sumtestsigma sum(diag(testsigma))];
%   end
%   disp(lambdas)
%   disp(deltat)
%   disp(sumsigma)
%   disp(sumtestsigma)
% end
% %save('lambdarcca.mat','lambdas','deltat','sumsigma','sumtestsigma');
% return

% for tmax=0:3
%   %ps=ceil(logspace(log(100)/log(10),log(2000)/log(10),20));
%   ps=[910 2000];
%   deltat=[];
%   sumsigma=[];
%   sumtrainsigma=[];
%   sumtestsigma=[];
%   for p=ps
%     randn('seed',8675309);
%     rand('seed',90210);
%     start=tic;
%     %[x,xb,y,yb,sigma] = rcca(data{1},data{2},60, ...
%     %                         struct('tmax',1,'init','rcca2','verbose',true));
%     [x,xb,y,yb,sigma]= rcca2(data{1}(train,:),data{2}(train,:),60,struct('tmax',tmax,'p',p,'lambda',1));
%     deltat=[deltat toc(start)];
%     sumsigma=[sumsigma sum(sigma)];
%     trainprox=bsxfun(@minus,data{1}(train,:)*x,xb);
%     trainproy=bsxfun(@minus,data{2}(train,:)*y,yb);
%     trainsigma=trainprox'*trainproy;
%     testprox=bsxfun(@minus,data{1}(test,:)*x,xb);
%     testproy=bsxfun(@minus,data{2}(test,:)*y,yb);
%     testsigma=(length(train)/length(test))*(testprox'*testproy);
%     sumtrainsigma=[sumtrainsigma sum(diag(trainsigma))];
%     sumtestsigma=[sumtestsigma sum(diag(testsigma))];
%   end
%   disp(ps)
%   disp(deltat)
%   disp(sumsigma)
%   disp(sumtestsigma)
%   save(sprintf('results%u.mat',tmax),'ps','deltat','sumsigma',...
%        'sumtrainsigma','sumtestsigma');
% end
%return

% prox=bsxfun(@minus,data{1}*x,xb);
% proy=bsxfun(@minus,data{2}*y,yb);
% empsigma=prox'*proy;
% empnormx=prox'*prox;
% empnormy=proy'*proy;
% disp(empsigma(1:10,1:10))
% disp(norm(empnormx-eye(60)))
% disp(norm(empnormy-eye(60)))
