function retval = rcca(Ltic, W, Rtic, k, varargin)
%RCCA randomized CCA
%
% retval = rcca(Ltic, W, Rtic, k) attempts to optimize
% the following constrained optimization problem
%
%    maximize x'*L'*diag(W)*R*y
%
%    subject to x'*(L'*diag(W)*L+cl*I)*x = sum(W)*I
%               y'*(R'*diag(W)*R+cr*I)*y = sum(W)*I
%
% using randomized methods. The two views are represented
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
% retval = rcca(Ltic, W, Rtic, k, opts) takes an additional struct
% with extra options.
%
%   opts.lambda: regularization parameter.  default is 1. 
%   opts.p: oversampling parameter.  default is 500.  larger gives 
%           better results, but uses more memory and takes longer.
%   opts.tmax: number of passes for the range finder.  default is 1.
%              2 is sometimes better.  more than 2 rarely helps.
%   opts.compress: detect and ignore unused features.  default is false.
%                  when using hashing to generate features, this can save
%                  substantial memory.
%   opts.kbs: column block size for MEX-accelerated matrix operations.
%             smaller block sizes save memory but run slower.  default
%             block size is number of latent dimensions.
%   opts.bs: example block size for matrix operations.  smaller 
%            block sizes save memory but run slower.  default block
%            size is number of examples.

    if exist('sparsequad','file') == 3 && ...
       exist('dmsm','file') == 3 && ...
       exist('sparseweightedsum','file') == 3
      havemex=true;
    else
      havemex=false;
      warning('rcca:nomex', 'MEX acceleration not available, have you compiled the mex?');
    end

    [dl,nl]=size(Ltic);
    [dr,nr]=size(Rtic);
    [~,nw]=size(W);
    sumw=sum(W);

    if (nl ~= nr || nr ~= nw)
      error('rcca:shapeChk', 'arguments have incompatible shape');
    end

    k=min([dl;dr;k]);
    [lambda,p,tmax,bs,kbs,compress]=parseArgs(nw,k,varargin{:});

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
    
    LticR=@(Z) LticRimpl(Z,bs,kbs,Ltic,mL,Rtic,mR,W,sumw,havemex);
    RticL=@(Z) LticRimpl(Z,bs,kbs,Rtic,mR,Ltic,mL,W,sumw,havemex);
    LticL=@(Z) LticRimpl(Z,bs,kbs,Ltic,mL,Ltic,mL,W,sumw,havemex);
    RticR=@(Z) LticRimpl(Z,bs,kbs,Rtic,mR,Rtic,mR,W,sumw,havemex);

    % 1. randomized top range finder for L'*diag(W)*R
    QL=randn(kp,dl);
    QR=randn(kp,dr);
    
    for ii=1:tmax
      YL=LticR(QR); clear QR;
      YR=RticL(QL); clear QL;
      if (ii<tmax)
        QL=modgs(YL,@(Z) Z*Z'); clear YL;
        QR=modgs(YR,@(Z) Z*Z'); clear YR;
      else
        QL=YL; clear YL;
        QR=YR; clear YR;
      end
    end
    
    % 2. compute feasible basis for top range
    %    alternate interpretation (?): QL <- QL*(L'*L)^(-1/2)
    QL=modgs(QL,@(Z) LticL(Z)*Z'+cl*(Z*Z'));
    QR=modgs(QR,@(Z) RticR(Z)*Z'+cr*(Z*Z'));
    
    % 3. final optimization over feasible basis
    
    % F = QL'*(L'*R)*QR' = U*S*V'
    % x = QL*U => x'*L'*L*x = U'*QL'*L'*L*QL*U = I
    % y = QR*V => y'*R'*R*y = V'*QR'*R'*R*QR*V = I
    % x'*L'*R*y = U'*QL'*L'*R*QR*V
    %           = U'*U*S*V'*V
    %           = S

    F=QL*LticR(QR)';
    [U,S,V]=svd(F);
    x=sqrt(sumw)*(QL'*U(:,1:k));
    xb=mL*x;
    y=sqrt(sumw)*(QR'*V(:,1:k));
    yb=mR*y;
    sigma=diag(S(1:k,1:k))';

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

function [lambda,p,tmax,bs,kbs,compress] = parseArgs(n,k,varargin)
  lambda=1;
  if (size(varargin,1) == 1 && isfield(varargin{1},'lambda'))
    lambda=varargin{1}.lambda;
  end   
  p=500;
  if (size(varargin,1) == 1 && isfield(varargin{1},'p'))
    p=varargin{1}.p;
  end
  tmax=1;
  if (size(varargin,1) == 1 && isfield(varargin{1},'tmax'))
    tmax=varargin{1}.tmax;
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
    error('rcca.project:shapeChk','argument has incompatible shape');
  end
  P=bsxfun(@minus,Z*x,xb);
end
