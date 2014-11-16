function retval = alscca(Ltic, W, Rtic, k, varargin)
%ALSCCA alternating least squares CCA
%
% retval = alscca(Ltic, W, Rtic, k) attempts to optimize
% the following constrained optimization problem
%
%    maximize x'*L'*diag(W)*R*y
%
%    subject to x'*(L'*diag(W)*L+cl*I)*x = sum(W)*I
%               y'*(R'*diag(W)*R+cr*I)*y = sum(W)*I
%
% using alternating least squares. The two views are represented
% by the matrices Ltic and Rtic.  Ltic and Rtic are (sparse) 
% design matrices whose rows are features and whose columns 
% are examples.  W is a column vector of importance weights.
%
% The return value is a struct containing the following fields:
%
%    projectx: function pointer to project left view 
%    projecty: function pointer to project right view 
%    x: projection matrix for left view
%    xb: mean shift vector for left view
%    y: projection matrix for right view
%    yb: mean shift vector for right view
%    sigma: vector of canonical correlations
%    usedL: if compression enabled, list of left view features
%    usedR: if compression enabled, list of right view features
%
% Use the projectx and projecty function pointers to project data
% and you get the mean shift and compression as appropriate.
%
% retval = alscca(Ltic, W, Rtic, k, opts) takes an additional struct
% with extra options.
%
%   opts.verbose: if true, display progress information.
%   opts.lambda: regularization parameter.  default is 1. 
%   opts.innerloop: number of passes in the inner loop of ALS.  
%                   increasing this might improve results, but
%                   will slow running time. default is 3.
%   opts.tmax: number of passes in the outer loop of ALS.
%              increasing this might improve results, but 
%              will slow running time.  default is 10.
%   opts.init: indicate initialization strategy.
%        'randn': random Gaussian matrix (the default).
%        'rcca': randomized CCA.  
%                in this case, opts.rccaopts is passed to rcca.
%   opts.pre: ALS preconditioner.  possible values are:
%        'diag': use diagonal preconditioner (the default).
%        'identity': use no preconditioner.
%        @(Z,d,s): any function handle.  Z is the matrix to precondition.
%                  d is a (vector) diagonal of the view covariance.
%                  s is '1' for left view and '2' for right view.
%   opts.p: oversampling parameter.  default is 10.  larger 
%           values might improve results, but will increase memory usage.
%   opts.compress: detect and ignore unused features.  default is false.
%                  when using hashing to generate features, this can save
%                  substantial memory.
%   opts.kbs: column block size for MEX-accelerated matrix operations.
%             smaller block sizes save memory but run slower.  default
%             block size is number of latent dimensions.
%   opts.bs: example block size for matrix operations.  smaller 
%            block sizes save memory but run slower.  default block
%            size is number of examples.
%
% opts.innerloop, opts.tmax, and opt.pre default values are tuned for 
% text problems.  For non-text data the first thing you should try is 
% a different preconditioner and/or increasing innerloop.

    start=clock;

    if exist('sparsequad','file') == 3 && ...
       exist('dmsm','file') == 3 && ...
       exist('sparseweightedsum','file') == 3
      havemex=true;
    else
      havemex=false;
      warning('alscca:nomex', 'MEX acceleration not available, have you compiled the mex?');
    end

    [dl,nl]=size(Ltic);
    [dr,nr]=size(Rtic);
    [~,nw]=size(W);
    sumw=sum(W);

    if (nl ~= nr || nr ~= nw)
      error('alscca:shapeChk', 'arguments have incompatible shape');
    end

    k=min([dl;dr;k]);
    [lambda,p,tmax,innerloop,bs,kbs,compress]=parseArgs(nw,k,varargin{:});

    if (compress)
      usedL=find(any(Ltic,2));
      [udl,~]=size(usedL);
      if (udl < dl)
        L=Ltic'; clear Ltic; Ltic=L(:,usedL)'; clear L;
        dl=udl;
      end
      usedR=find(any(Rtic,2));
      [udr,~]=size(usedR);
      if (udr < dr)
        R=Rtic'; clear Rtic; Rtic=R(:,usedR)'; clear R;
        dr=udr;
      end
      k=min([dl;dr;k]);
    end

    kp=min([dl;dr;k+p]);

    if (havemex && issparse(Ltic))
      mL=sparseweightedsum(Ltic,W,1)/sumw;
      dLL=sparseweightedsum(Ltic,W,2)-sumw*mL.*mL;
    else
      mL=full(sum(bsxfun(@times,Ltic,W),2)')/sumw;
      dLL=sum(bsxfun(@times,Ltic.*Ltic,W),2)'-sumw*mL.*mL;
    end  
    
    if (havemex && issparse(Rtic))
      mR=sparseweightedsum(Rtic,W,1)/sumw;
      dRR=sparseweightedsum(Rtic,W,2)-sumw*mR.*mR;
    else
      mR=full(sum(bsxfun(@times,Rtic,W),2)')/sumw;
      dRR=sum(bsxfun(@times,Rtic.*Rtic,W),2)'-sumw*mR.*mR;
    end

    cl=lambda*sum(dLL)/dl;
    cr=lambda*sum(dRR)/dr;

    dLL=dLL+cl;
    dRR=dRR+cr;
    
    LticR=@(Z) LticRimpl(Z,bs,kbs,Ltic,mL,Rtic,mR,W,sumw,havemex);
    RticL=@(Z) LticRimpl(Z,bs,kbs,Rtic,mR,Ltic,mL,W,sumw,havemex);
    LticL=@(Z) LticRimpl(Z,bs,kbs,Ltic,mL,Ltic,mL,W,sumw,havemex);
    RticR=@(Z) LticRimpl(Z,bs,kbs,Rtic,mR,Rtic,mR,W,sumw,havemex);
    
    [preleft,preright]=parsePreconditioner(dLL,dRR,varargin{:});
    
    [QL,QR]=initialize(Ltic,W,Rtic,LticL,dl,cl,RticR,dr,cr,kp,varargin{:});
    if (size(varargin,1) == 1 && isfield(varargin{1}, 'project'))
        QL=varargin{1}.project(QL, 1);
        QR=varargin{1}.project(QR, 2);
    end
    
    for j=1:tmax
      if (size(varargin,1) == 1 && isfield(varargin{1},'verbose') && varargin{1}.verbose)
        [flass,~,~]=subspaceopt(RticL,QL,QR,k);
        disp(struct('iteration',(j-1),'sumsigma',sum(flass(1:k)),...
                    'topsigma',flass(1:min(k,8)),'deltat',etime(clock,start)));
      end
        
      % innerloop data passes
      YL=cheesypcg(@(Z) LticL(Z)+cl*Z, preleft, QL, LticR(QR), innerloop);  
      % 1 data pass
      QL=modgs(YL,@(Z) LticL(Z)*Z'+cl*(Z*Z')); clear YL;
      if (size(varargin,1) == 1 && isfield(varargin{1}, 'project'))
        QL=varargin{1}.project(QL, 1);
      end
  
      % innerloop data passes
      YR=cheesypcg(@(Z) RticR(Z)+cr*Z, preright, QR, RticL(QL), innerloop);
      % 1 data pass
      QR=modgs(YR,@(Z) RticR(Z)*Z'+cr*(Z*Z'));
      if (size(varargin,1) == 1 && isfield(varargin{1}, 'project'))
        QR=varargin{1}.project(QR, 2);
      end
    end
    [sigma,x,y]=subspaceopt(RticL,QL,QR,k);
    if (size(varargin,1) == 1 && isfield(varargin{1},'verbose') && varargin{1}.verbose)
      disp(struct('iteration',tmax,'sumsigma',sum(sigma(1:k)),...
                  'topsigma',sigma(1:min(k,8)),'deltat',etime(clock,start)));
    end
    x=sqrt(sumw)*x;
    xb=mL*x;
    y=sqrt(sumw)*y;
    yb=mR*y;
    
    if (compress)
      retval=struct('x',x,'xb',xb,'y',y,'yb',yb,'sigma',sigma,...
                    'projectx',@(Z) project(Z,x,xb,usedL), ...
                    'projecty',@(Z) project(Z,y,yb,usedR), ...
                    'usedL',usedL,'usedR',usedR);
    else
      retval=struct('x',x,'xb',xb,'y',y,'yb',yb,'sigma',sigma, ...
                    'projectx',@(Z) project(Z,x,xb), ...
                    'projecty',@(Z) project(Z,y,yb));
    end
end

function [lambda,p,tmax,innerloop,bs,kbs,compress] = parseArgs(n,k,varargin)
  lambda=1;
  if (size(varargin,1) == 1 && isfield(varargin{1},'lambda'))
    lambda=varargin{1}.lambda;
  end   
  p=10;
  if (size(varargin,1) == 1 && isfield(varargin{1},'p'))
    p=varargin{1}.p;
  end
  tmax=10;
  if (size(varargin,1) == 1 && isfield(varargin{1},'tmax'))
    tmax=varargin{1}.tmax;
  end
  innerloop=3;
  if (size(varargin,1) == 1 && isfield(varargin{1},'innerloop'))
    innerloop=varargin{1}.innerloop;
  end
  bs=n;
  if (size(varargin,1) == 1 && isfield(varargin{1},'bs'))
    bs=varargin{1}.bs;
  end
  kbs=k+p;
  if (size(varargin,1) == 1 && isfield(varargin{1},'kbs'))
    kbs=varargin{1}.kbs;
  end
  compress=false;
  if (size(varargin,1) == 1 && isfield(varargin{1},'compress'))
    compress=varargin{1}.compress;
  end
end

function [preleft,preright] = parsePreconditioner(dLL,dRR,varargin)
  if (size(varargin,1) == 0 || ~isfield(varargin{1}, 'pre') || strcmp(varargin{1}.pre, 'diag'))
    preleft=@(z) bsxfun(@rdivide,z,dLL);
    preright=@(z) bsxfun(@rdivide,z,dRR);
  elseif (strcmp(varargin{1}.pre, 'identity'))
    preleft=@(z) z;
    preright=@(z) z;
  else
    preleft=@(z) varargin{1}.pre(z, dLL, 1);
    preright=@(z) varargin{1}.pre(z, dRR, 2);
  end  
end
  
function [QL,QR] = initialize(Ltic,W,Rtic,LticL,dl,cl,RticR,dr,cr,kp,varargin)
  if (size(varargin,1) == 0 || ~isfield(varargin{1}, 'init') || ...
      strcmp(varargin{1}.init, 'randn'))
    QL=modgs(randn(kp,dl),@(Z) LticL(Z)*Z'+cl*(Z*Z'));
    QR=modgs(randn(kp,dr),@(Z) RticR(Z)*Z'+cr*(Z*Z'));
  elseif (strcmp(varargin{1}.init, 'rcca'))
    mystruct=struct();
    if (isfield(varargin{1},'rccaopts'))
      mystruct=varargin{1}.rccaopts;
    end
    cca=rcca(Ltic,W,Rtic,kp,mystruct);
    sumw=sum(W);
    QL=cca.x'/sqrt(sumw);
    QR=cca.y'/sqrt(sumw);
  else
    error('bad initialization spec');
  end
end

function Y = LticRimpl(Z,bs,kbs,Ltic,mL,Rtic,mR,W,sumw,havemex)
  [~,n]=size(Ltic);
  [k,~]=size(Z);
  Y=-sumw*((Z*mR')*mL);
  if (bs >= n)
    if havemex && issparse(Rtic) && issparse(Ltic)
      if (kbs >= k)
        Y=Y+sparsequad(Ltic,W,Rtic,Z);
      else
        for koff=1:kbs:k
          koffend=min(k,koff+kbs-1);
          Y(koff:koffend,:)=Y(koff:koffend,:)+sparsequad(Ltic,W,Rtic,Z(koff:koffend,:));
        end
      end
    elseif havemex && issparse(Rtic)      
      Y=Y+dmsm(Z,Rtic,W)*Ltic';
    elseif havemex && issparse(Ltic)
      Y=Y+dmsm(bsxfun(@times,Z*Rtic,W),Ltic');
    else
      Y=Y+bsxfun(@times,Z*Rtic,W)*Ltic';
    end
  else
    if havemex && issparse(Rtic) && issparse(Ltic)
      if (kbs >= k)
        Y=Y+sparsequad(Ltic,W,Rtic,Z,kbs);
      else
        for koff=1:kbs:k
          koffend=min(k,koff+kbs-1);
          Y(koff:koffend,:)=Y(koff:koffend,:)+sparsequad(Ltic,W,Rtic,Z(koff:koffend,:));
        end
      end       
    elseif havemex && issparse(Rtic)      
      for off=1:bs:n
        offend=min(n,off+bs-1);
        Y=Y+dmsm(Z,Rtic,W,off,offend)*Ltic(:,off:offend)';
      end
    elseif havemex && issparse(Ltic)
      for off=1:bs:n
        offend=min(n,off+bs-1);
        Y=Y+dmsm(bsxfun(@times,Z*Rtic(:,off:offend),W(off:offend)),Ltic(:,off:offend)');
      end
    else
      for off=1:bs:n
        offend=min(n,off+bs-1);
        Y=Y+bsxfun(@times,Z*Rtic(:,off:offend),W(off:offend))*Ltic(:,off:offend)';
      end
    end
  end
end

function [sigma,x,y] = subspaceopt(RticL,QL,QR,k)
  T=QR*RticL(QL)';
  [U,S,V]=svd(T);
  sigma=diag(S(1:k,1:k))';
  x=QL'*V(:,1:k);
  y=QR'*U(:,1:k);
end

% objective is \| A Y - B \|^2
function Y = cheesypcg(Afunc,preAfunc,Y,b,iter)
  tol=1e-6;
  
  r=b-Afunc(Y);
  z=preAfunc(r);
  p=z;
  rho=diag(r*z');
  initsumzz=sum(sum(z.*z));
  
  for ii=1:iter
    Ap=Afunc(p);
    alpha=rho./diag(p*Ap');
    Y=Y+bsxfun(@times,p,alpha);
    r=r-bsxfun(@times,Ap,alpha);
    z=preAfunc(r);
    newsumzz=sum(sum(z.*z));     
    if (newsumzz<tol*initsumzz)
        break;
    end
    rho1=rho;
    rho=diag(r*z');
    beta=rho./rho1;
    p=z+bsxfun(@times,p,beta);
  end
end

function Y = modgs( Y, W )
  if (isfloat(W))
    %W is a symmetric matrix
    C=Y*(W*Y');   
  else
    assert(isa(W,'function_handle'));
    %W is a function that computes Y'*(W*Y)
    C=W(Y);
  end
  [V,D]=eig(0.5*(C+C'));
  Y=pinv(sqrt(max(D,0)))*V'*Y;
end

function P = project(Z,x,xb,varargin)
  if (nargin == 4)
    Z=Z(:,varargin{1});
  end
  [~,dl]=size(Z);
  [dlx,~]=size(x);
  if (dl ~= dlx)
    error('alscca.project:shapeChk','argument has incompatible shape');
  end
  P=bsxfun(@minus,Z*x,xb);
end
