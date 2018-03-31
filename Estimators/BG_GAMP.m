% -------------------------------------------------------------------------
% Author:   Matthew L Kokshoorn 
% Group:    The University of Sydney 
% Role:     PhD Candidate.
% Date:     18/05/2017.  
% email:    matthew.kokshoorn@ieee.com
%
% -------------------------------------------------------------------------
% Light-weight Implementation of BG-GAMP algorithm in:
%
%  J. Vila and P. Schniter, “Expectation-maximization bernoulli-gaussian
%  approximate message passing,” in IEEE Conf. on Signals, Systems and
%  Computers, 2011, pp. 799–803.
%
% Lines of code in this fucntion are labelled to correspond to the 
% lines in the paper's algorithm e.g.,  (R1),(R2) etc.
%
% -------------------------------------------------------------------------
function [ v, est_stats ] = BG_GAMP(A,y,N_0,P_R,v_lambda,P_v_APP,varargin) 
%  Solves for v in a compressed sensing equation of the form
%
%                       y = Av + n 
%  where:
%  @MATH: v: is a (N x 1) sparse vector follwing an i.i.d., bernoulli-gaussian
%  (BG) distribution, charecterized by having an expected number of
%  non-zero entries as v_lambda*N, and with each non-zero value following
%  a circularly symmetric, zero mean, complex, AWGN process with variance P_R.
%  @MATH: n: is a length-N i.i.d. complex AWGN noise vector with variance, N_0.
%
%  @INPUT y: Is an observation vector of size (M x 1)
%  @INPUT v_lambda: is the probabillity of a non-zero element in v (prior).
%  @INPUT N_0 - variance of Complex AWGN noise process.
%  @INPUT P_R - variance of non-zero  R.V. elements in v.
%  @INPUT A: is a M x N compressed sensing matrix which relates the 
%  observations in y, to the unknown vector, v.
%
%  @INPUT P_v_APP: is an (N x 1) vector of a posteriori probabilities (APP)
%  That me be used applied to enhance estiamtion (default to all 0.5). 
%
%  @OUTPUT v:         -  Final estimate of v.
%  @OUTPUT est_stats: -  Struct containing convergence results.
%
%  Optional Inputs:
%
%  @INPUT It_min - optional number of minimum iterations. Default 10.
%  @INPUT It_max - optional number of maximum iterations. Default 50.
%
% Notes:
%
% Sparsity - provide A as a SPARSE matrix to increase performance, 
% if appropriate.
%
% Numerical Instability - Some lines are added to provide numerical
% stabillity and can be removed for norminal SNRs. May also reduce
% performance in some SNR regimes.
%
% Todo:
% Optional Warm start   - could be given when v is being updated..
% v covariance - could be considered in line labelled as (8).
%
% -------------------------------------------------------------------------

    %Saturate Noise (To pevent numerical instabillity). 
    %N_0=max(N_0,10^-2);
    r_var_max=10^6;
    
    %If optional arguments are not provided, use defaults.
    if(~(length(varargin)==2))
        It_min=20;          
        It_max=20;
    else
        It_min=varargin{1};
        It_max=varargin{2};
    end
    
    %Deterine Size of sensing matrix.
    [M,N]=size(A);
    error_msg='Number of rows in A must equal length of y!';
    assert(M==length(y),error_msg);
    
    % Convert sparsity to log.
    log_lambda_term= log((1-v_lambda)/v_lambda); 
    
    % Init passing vectors.
    s_est = complex(v_lambda.*zeros(M,1));
    v_prev = complex(zeros(N,1));
    v_var_prev  = complex(zeros(N,1));
    
    % Convert APP vector to log. 
    if(any(not(P_v_APP==1)))
        LLR_EXT =-log(P_v_APP);
    else
        LLR_EXT=0;
    end
    
    % Precompute conjugate transpose and abs-square.
    A_herm= A';
    A_sq = abs(A).^2;

    for It=1:It_max

        % Output Linear Step. Scalar Variances. ---------------------------
        z_est =  A*v_prev;                           % (R1)    
        z_var =  A_sq*(v_var_prev);                  % (R2)
        z_var = min(abs((z_est-y)./s_est),z_var);    % Numerical Stability.
        p_est =  z_est - z_var.*s_est;               % (R3)

        % Output non-Linear Step. Complex AWGN Output Channel -------------
        s_est = (y-p_est)./( z_var + N_0);           % (R4)
        s_var  = 1./(z_var+N_0);                     % (R5)
        
        % Input Linear Step. ----------------------------------------------
        r_var = (1./(A_sq'*s_var));                  % (R6)
        r_var(r_var==Inf)=r_var_max;                 % Numerical Stability.
        r_est= v_prev + r_var.*(A_herm*s_est);       % (R7)
        
        % Input Non-Linear Step. BG --------------------------------------
        r_est_sq=abs(r_est).^2;
        r_var_inv=1./r_var;
        
        % LLR of bernoulli process in (8).
        temp=log(1+P_R.*r_var_inv)- P_R*r_est_sq.*r_var_inv./(r_var + P_R); 
        v_LLR= LLR_EXT +log_lambda_term + 0.5*(temp); 
        pi_f=1./(1+exp(v_LLR));

        % Gaussian Process - Correspond to functions in (9) and (10).
        v_f = 1./( r_var_inv + 1/P_R );                          %(10)        
        gamma_f = ( r_est.*r_var_inv  ).*v_f;                    %(9 )
        v   =    pi_f.* gamma_f;                                 %(R8)
        v_var   =    pi_f.*(v_f + (1 - pi_f).*abs(gamma_f).^2);  %(R9)   

        % Check Convergence -----------------------------------------------     
        SE = sum(abs(v- v_prev).^2);
        if((SE<10^-6 && It>=It_min) || It==It_max )
            break;
        else
            v_prev=v;
            v_var_prev=v_var;
        end
        
    end
    
    est_stats.SE=SE;  
    est_stats.pi_f=pi_f;
    est_stats.its=It;
    est_stats.v_var=v_var;

end


