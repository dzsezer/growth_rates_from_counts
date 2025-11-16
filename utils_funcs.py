import torch

from collections import deque

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec

from matplotlib.ticker import MaxNLocator

import numpy as np

# DYNAMICS of growth

def saturationK(model, K_N0, ys):
    if model == 'logistic':
        return 1 - torch.exp(ys-K_N0).sum(dim=0)  # sum over K or [K,M] → shape: [] or [M]
    elif model == 'gompertz':
        return -torch.logsumexp(ys-K_N0, dim=0)   # shape: [] or [M]
    else:
        raise ValueError(f"Unknown model type: {model}")
    

def dynamics(model, K_N0, times_fine,lams,dels):
    '''
    Inputs:
        lams, dels: shape [K] or [K, M]
    Outputs:
        ys_fine: shape [K, T] or [K, M, T]
    '''
    pntT = len(times_fine)
    ys_fine = [dels]

    for i in range(1, pntT):
        dt = times_fine[i] - times_fine[i - 1]
        prev_ys = ys_fine[-1]

        if model == 'exponential':
            # calculate analytically
            new_ys = lams*times_fine[i] + dels
        else:
            satK = saturationK(model, K_N0, prev_ys)  # shape: [] or [M]

            # Expand satK to match prev_ys shape
            while satK.ndim < prev_ys.ndim:
                satK = satK.unsqueeze(0)  # unsqueeze to match leading K dim

            new_ys = prev_ys + dt * lams * satK  # shape: [K] or [K, M]

        ys_fine.append(new_ys)

    ys_fine = torch.stack(ys_fine, dim=-1)  # shape: [K, T] or [K, M, T]
    return ys_fine


def dynamics_selected_times(model, K_N0,times_fine,selected_indices,lams,dels):
    '''
    Inputs:
        lams,dels: [K] or [K,M]
    Outputs:
        ys: [K,T] or [K,M,T]
    Where:
        K: number of variants (barcodes); dim=0
        T: number of experimental time points; dim=-1
        M: number of random samples for stochastic gradient descent
    '''
    if model == 'exponential':
        # calculate analytically
        times = times_fine[selected_indices]
        # Adjust reads for broadcasting if M exists
        if lams.ndim == 2:
            ys = lams[:,:,None]*times[None,None,:] + dels[:,:,None] # [K, M, T]
        else:
            ys = lams[:,None]*times[None,:] + dels[:,None] # [K, T]
    else:
        # other models integrate dynamics numerically
        ys_fine = dynamics(model,K_N0,times_fine, lams, dels)
        ys = ys_fine[...,selected_indices] # shape [K,T] or [K, M, T]
    return ys


# test if functions work with both shapes [K] and [K,M] of the inputs

def check_dynamics(models, K_N0, selection_bools, noise_bools, params_true,params_time):

    growth_rates_true,log_abundances_true = params_true
    times_fine,selected_indices = params_time

    K = growth_rates_true.size(0)

    for mod in models:    
        print(">>",mod)
        for noise in noise_bools:
            if noise:
                print("     with noise")
                M = 8
                eps = torch.randn(K,M)
                sigmas = 0.03*torch.ones(K,) # K x 1
                lams = growth_rates_true[:,None] + sigmas[:,None]*eps
                dels = log_abundances_true[:,None] + sigmas[:,None]*eps
            else:
                print("     without noise")
                lams = growth_rates_true
                dels = log_abundances_true

        
            for select in selection_bools:
                if select:
                    #test logistic selected times
                    ys_coarse = dynamics_selected_times(mod,K_N0,times_fine,selected_indices,lams,dels)
                    fracs_coarse = torch.softmax(ys_coarse,dim=0)
                    print("ys_fine:",ys_coarse.shape)
                else:
                    #test logistic
                    ys_fine = dynamics(mod,K_N0,times_fine,lams,dels)
                    print("ys_fine:",ys_fine.shape)


# LOG LIKELIHOOD

def calc_loglikelihood(model,K_N0, reads, times_fine, selected_indices, lams, dels):
    '''
    Inputs:
        reads: [K, T]
        lams, dels: [K] or [K, M]
    Output:
        loglikelihood: [] or [M]
    '''
    #K, T = reads.shape
    
    ys = dynamics_selected_times(model,K_N0, times_fine, selected_indices, lams, dels)  # [K, T] or [K, M, T]

    # Adjust reads for broadcasting if M exists
    if ys.ndim == 3:
        # ys: [K, M, T], reads: [K, T] → make reads [K, 1, T]
        reads_ = reads[:,None,:] # [K, 1, T]
    else:
        reads_ = reads  # [K, T]

    # First term: sum over K, then T
    first = (reads_ * ys).sum(dim=0).sum(dim=-1)  # [] or [M]

    # Second term: 
    # sum over K
    log_term = torch.logsumexp(ys, dim=0)  # [T] or [M, T]
    # sum reads over K
    reads_sum = reads_.sum(dim=0)  # [T] or [1,T]
    # sum over T
    second = (reads_sum * log_term).sum(dim=-1)  # [] or [M]

    loglikelihood = first - second  # [] or [M]
    return loglikelihood.mean() # the mean is over M
  

# ELBO for ALL time points

def calc_elbo_stoch(model,K_N0,reads,times_fine,selected_indices, muLs,log_sigLs,muDs,log_sigDs,K,M):
    epsL = torch.randn(K, M)
    sigLs = torch.exp(log_sigLs)
    lams = muLs[:,None] + sigLs[:,None] * epsL # [K,M]
    
    epsD = torch.randn(K, M)
    sigDs = torch.exp(log_sigDs)
    dels = muDs[:,None] + sigDs[:,None] * epsD # [K,M]

    loglike = calc_loglikelihood(model,K_N0, reads, times_fine, selected_indices, lams, dels)
    return loglike + log_sigLs.sum() + log_sigDs.sum()


def calc_elbo_bound(reads,times, muLs,log_sigLs,muDs,log_sigDs,K):
    '''
        EXPONENTIAL only !
    '''
    # First term: 
    means = muLs[:,None]*times[None,:] + muDs[:,None] # [K,T]
    # sum over K, then T
    first = (reads * means).sum(dim=0).sum(dim=-1) 

    # Second term: 
    sigLs2 = torch.exp(log_sigLs)**2
    sigDs2 = torch.exp(log_sigDs)**2
    sigs = sigLs2[:,None] * (times**2)[None,:] + sigDs2[:,None] # [K,T]
    # sum over K
    log_term = torch.logsumexp(means+0.5*sigs, dim=0)  # [T] 
    # sum reads over K
    reads_sum = reads.sum(dim=0)  # [T] 
    # sum over T
    second = (reads_sum * log_term).sum(dim=-1)  # [] 

    # combine
    return first - second + log_sigLs.sum() + log_sigDs.sum()


# ELBO for SINGLE time point

def single_point_elbo_stoch(reads,mus,log_sigs,K,M):
    # non-stochastic terms
    first = (reads * mus).sum()
    third = log_sigs.sum()
    # stochastic term
    sigs = torch.exp(log_sigs)    
    eps = torch.randn(K, M)    
    ys = mus[:,None] + sigs[:,None] * eps # (K,M)
    second = reads.sum() * torch.logsumexp(ys,dim=0).mean()
    # combine
    return first - second + third  


def single_point_elbo_bound(reads,mus,log_sigs):
    # non-stochastic terms
    first = (reads * mus).sum()
    third = log_sigs.sum()
    # APPROXIMATION to stochastic term
    sigs = torch.exp(log_sigs)    
    ys = mus + 0.5*sigs**2
    second = reads.sum() * torch.logsumexp(ys,dim=0).mean()
    # combine
    return first - second + third  


# Weighted mean-squared error

def wmsq_log(times,lams,dels,psi0s,psi1s,rho=0):
    #Err = (psi0s[k]-psi0s[rho] - (lams[k]-lams[rho])*times-(dels[k]-dels[rho]))**2/(psi1s[k]+psi1s[rho])
    num = (psi0s-psi0s[rho] - (lams-lams[rho])[:,None]*times[None,:]-(dels-dels[rho])[:,None])**2 # (K,T)
    denom = psi1s+psi1s[rho] # (K,T)
    Err = num/denom
    return Err.sum() # sum is over T _and_ K


def wmsq_frac(times,lams,dels,alphas,alphas_sum,rho=0):
    ys = (lams-lams[rho])[:,None]*times[None,:] + (dels-dels[rho])[:,None]
    phis = torch.softmax(ys,dim=0)
    mean_f = alphas/alphas_sum
    var_f = mean_f*(1-mean_f)/(alphas_sum+1)
    Err = (mean_f - phis)**2/var_f
    return Err.sum()


################################

# OPTIMIZATION

def print_params(label, lams, dels):
    lams_str = ', '.join(f"{v:.6f}" for v in lams.flatten().tolist())
    dels_str = ', '.join(f"{v:.6f}" for v in dels.flatten().tolist())
    print(f"{label}: lams = [{lams_str}], dels = [{dels_str}]]")


# WEIGHTED MEAN-SQUARED error

def optimize_WLSQ(reads,params_time,alpha_prior,rho,lr,num_iterations,error):

    times_fine,selected_indices = params_time
    times = times_fine[selected_indices]

    K, T = reads.size()

    alphas = reads + alpha_prior

    if error == 'log':
        # digamma and trigamma
        psi0s = torch.digamma(alphas)
        psi1s = torch.polygamma(1,alphas)
    elif error == 'frac':
        alphas_sum = alphas.sum(dim=0)
        # fraction mean and variance 
        #f_means = alphas/alphas_sum
        #f_vars = f_means*(1-f_means)/(alphas_sum+1)
        #f_stds = torch.sqrt(f_vars)

    # initial values for the optimization
    lams = torch.full((K,), 0.1, requires_grad=True)
    dels = torch.full((K,), torch.log(1./torch.tensor(K)), requires_grad=True) 


    optimizer = torch.optim.Adam([lams, dels], lr=lr)

    history = []

    prev_values = None
    prev_loss = None
    for i in range(num_iterations):

        #print(f"step {i}")
        optimizer.zero_grad()

        if error == 'log':
            loss = wmsq_log(times,lams,dels,psi0s,psi1s,rho)
        elif error == 'frac':
            loss = wmsq_frac(times,lams,dels,alphas,alphas_sum,rho)

        loss.backward()
        optimizer.step()

        history.append(loss.item())

        # Apply constraints
        with torch.no_grad():
            lams[rho] = 0.
            dels[rho] = 0.

    return lams.detach(), dels.detach(), history


# Log Likelihood

def optimize_LogLike(reads, params_time, model, K_N0,
                        lr=0.01, tol=1e-5, num_iterations=50000):

    times_fine,selected_indices = params_time
    
    K, T = reads.size()
    # initial values for the optimization
    lams = torch.full((K,), 0.1, requires_grad=True)
    # equal fractions to all variants
    dels = torch.full((K,), -torch.log(torch.tensor(K)), requires_grad=True) 

    print_params("Initial Params ", lams, dels)

    optimizer = torch.optim.Adam([lams, dels], lr=lr)

    history = []

    prev_values = None
    prev_loss = None
    for i in range(num_iterations):

        #print(f"step {i}")
        optimizer.zero_grad()

        loglike = calc_loglikelihood(model,K_N0,reads, times_fine, selected_indices, lams, dels)
        loss = -loglike/K
        loss.backward()
        optimizer.step()

        history.append(loss.item())

        # Apply constraints
        with torch.no_grad():
            # impose initial density on dels
            dels -= torch.logsumexp(dels,dim=0)
            if model == 'exponential':    
                # remove the mean of the growth rates            
                lams -= lams.mean()                

    
        values = torch.stack([p for p in [lams,dels]])

        if prev_values is not None:
            diff_val = (values - prev_values).abs().max()
            diff_loss = (loss - prev_loss).abs()

            if diff_val < tol[1] and diff_loss < tol[0]:
                print(f"Converged at iteration {i} with diff_loss {diff_loss.item():.14e} and diff_val {diff_val:.6e}")
                break

        prev_values = values
        prev_loss = loss

        if i % (num_iterations // 10) == 0:
            print_params(f"Iteration {i:5}", lams, dels)           

    print_params("Final grads  ",lams.grad, dels.grad)
    print_params("Final Params ", lams, dels)
    
    fitness = lams.detach().flatten()
    abundance = dels.detach().flatten()

    return fitness, abundance, history


# ELBO for ALL times

def optimize_ELBO(reads, params_time, model,K_N0,
    num_steps,lr,M,F, maxlike):

    times_fine,selected_indices = params_time
    growth_rates_ML,log_abundances_ML = maxlike
    #if model == 'exponential':
    times = times_fine[selected_indices]

    K, T = reads.size()
    
    # --- Variational Parameters (learnable) ---
    muLs = growth_rates_ML.clone().detach().requires_grad_(True)
    log_sigLs = (-0.5*torch.log(((times**2)[None,:]*reads).sum(dim=1))).requires_grad_(True)
    #
    muDs = log_abundances_ML.clone().detach().requires_grad_(True)
    log_sigDs = (-0.5*torch.log(reads.sum(dim=1))).requires_grad_(True)

    optimizer = torch.optim.AdamW([muLs, log_sigLs, muDs, log_sigDs], lr=lr)

    # --- Rolling buffers for parameter averaging ---
    muL_buffer = deque(maxlen=F)
    muD_buffer = deque(maxlen=F)
    log_sigL_buffer = deque(maxlen=F)
    log_sigD_buffer = deque(maxlen=F)

    history = []

    for i in range(num_steps):
        optimizer.zero_grad()
        if (model == 'exponential' and M == 0):
            elbo = calc_elbo_bound(reads,times,muLs,log_sigLs,muDs,log_sigDs,K)
        else:
            elbo = calc_elbo_stoch(model,K_N0,reads,times_fine,selected_indices,muLs,log_sigLs,muDs,log_sigDs,K,M)
        loss = -elbo/K
        loss.backward()
        optimizer.step()

        # Apply constraints
        with torch.no_grad():
            # impose initial density of 1 on dels
            shift = torch.logsumexp(muDs,dim=0) #- torch.log(N0_K)
            muDs -= shift

            # remove mean of growth rates
            if model == 'exponential':
                muLs -= muLs.mean()

        # Add current parameters to rolling buffers
        muL_buffer.append(muLs.detach().clone())
        log_sigL_buffer.append(log_sigLs.detach().clone())
        muD_buffer.append(muDs.detach().clone())
        log_sigD_buffer.append(log_sigDs.detach().clone())

        history.append(loss.item())

    # --- Final Rolling-Average Parameters ---
    muL_avg = torch.stack(list(muL_buffer)).mean(dim=0)
    muD_avg = torch.stack(list(muD_buffer)).mean(dim=0)
    # --- Get means and variances for log of sigma ---
    log_sigL_avg = torch.stack(list(log_sigL_buffer)).mean(dim=0)
    log_sigD_avg = torch.stack(list(log_sigD_buffer)).mean(dim=0)
    log_sigL_var = torch.stack(list(log_sigL_buffer)).var(dim=0, unbiased=False)
    log_sigD_var = torch.stack(list(log_sigD_buffer)).var(dim=0, unbiased=False)
    # --- Mean of sigma from log-normal distribution ---
    sigL_avg = torch.exp(log_sigL_avg + 0.5*log_sigL_var)
    sigD_avg = torch.exp(log_sigD_avg + 0.5*log_sigD_var)

    # impose initial density of 1 on mean deltas
    shift = torch.logsumexp(muD_avg,dim=0) #- torch.log(N0_K)
    muD_avg -= shift

    # remove mean of mean lambdas
    if model == 'exponential':        
        muL_avg -= muL_avg.mean()

    return muL_avg,sigL_avg,muD_avg,sigD_avg,history


# ELBO for SINGLE time


def first_point_maxELBO(reads,num_steps,lr,M,F,jensen=False):
    K, T = reads.size()

    reads1 = reads[:,0]

    # --- Variational Parameters (learnable) ---
    if jensen:
        # Initialize from Jensen lower bound
        log_reads1 = torch.log(reads1)
        log_sigs = (-0.5*log_reads1).requires_grad_(True)
        mus = (log_reads1-0.5/reads1).requires_grad_(True)
    else:
        # Initialize somehow
        mus = torch.zeros(K).requires_grad_(True)
        log_sigs = torch.zeros(K).requires_grad_(True)

    with torch.no_grad():
        # impose initial density of 1 on means
        mus -= torch.logsumexp(mus,dim=0)

    # --- Rolling buffers for parameter averaging ---
    mu_buffer = deque(maxlen=F)
    log_sig_buffer = deque(maxlen=F)

    # Add current parameters to rolling buffers
    mu_buffer.append(mus.detach().clone())
    log_sig_buffer.append(log_sigs.detach().clone())

    history = []
    

    if M: # M > 0
        optimizer = torch.optim.AdamW([mus, log_sigs], lr=lr)
        
        for i in range(num_steps):
            #print(f"step {i}")
            optimizer.zero_grad()

            elbo = single_point_elbo_stoch(reads1,mus,log_sigs,K,M)
                        
            loss = -elbo/K
            loss.backward()

            optimizer.step()

            # Apply constraints
            with torch.no_grad():
                # impose initial density of 1 on means
                mus -= torch.logsumexp(mus,dim=0) 

            # Add current parameters to rolling buffers
            mu_buffer.append(mus.detach().clone())
            log_sig_buffer.append(log_sigs.detach().clone())

            history.append(loss.item())

    else:
        elbo = single_point_elbo_bound(reads1,mus,log_sigs)
        history.append(-elbo.item())

    mus = mus.detach().clone()
    sigs = torch.exp(log_sigs).detach().clone()

    # --- Final Rolling-Average Parameters ---
    mu_avg = torch.stack(list(mu_buffer)).mean(dim=0)
    # impose initial density of 1 on means
    mu_avg -= torch.logsumexp(mu_avg,dim=0)
    
    # --- Get means and variances for log of sigma ---
    log_sig_avg = torch.stack(list(log_sig_buffer)).mean(dim=0)
    log_sig_var = torch.stack(list(log_sig_buffer)).var(dim=0, unbiased=False)
    # --- Mean of sigma from log-normal distribution ---
    sig_avg = torch.exp(log_sig_avg + 0.5*log_sig_var)    

    return mu_avg,sig_avg,mus,sigs,history



#############################

# PLOTS 

# Likelihood and ELBO maximization

def plot_History(history,model,inset_start,figsize,ylabel=None,ylim=None,title=None):

    steps = range(0,len(history))
    
    fig = plt.figure(figsize=figsize)

    if len(history) == 1:
        plt.plot(steps,history,'o')
    else:
        plt.plot(steps,history)

    plt.ticklabel_format(useOffset=False)

    plt.xlabel('Iteration Step', fontsize=12)
    if ylabel:
        plt.ylabel(ylabel, fontsize=12)
    else:
        plt.ylabel('Negative Log Likelihood', fontsize=12)
    if title:
        plt.title(title,fontsize=14)
    else:
        plt.title(f"{model.capitalize()} growth",fontsize=14)

    # ---- Inset ----
    if inset_start > -1:
        ax = plt.gca()  # get current axes
        # `bbox_to_anchor` shifts the inset inward from the corner
        inset_ax = inset_axes(
            ax, width="60%", height="60%", 
            loc='upper right', 
            bbox_to_anchor=(-0.05, -0.05, 1, 1),  # shifts left and down a bit
            bbox_transform=ax.transAxes,
            borderpad=0.5  # slight internal padding
        )
        start = inset_start
        inset_ax.plot(steps[start:],history[start:])
        inset_ax.tick_params(labelsize=8)
        inset_ax.ticklabel_format(useOffset=False)
        if ylim:
            inset_ax.set_ylim(133,135)

    return fig



# PLOTS INITIAL

def plot_read_counts(reads,times,model,colors,markers,figsize,legend=True,title=True,ylog=False):

    K,T = reads.size()

    fig = plt.figure(figsize=figsize)

    for k in range(K):
        color = colors[k % len(colors)]
        marker=markers[k % len(markers)]
        plt.plot(times, reads[k], ':', color=color,marker=marker, label='Var '+str(k+1))

    plt.xticks(times)
    if ylog:
        plt.yscale('log')

    plt.xlabel('Time',fontsize=12)
    plt.ylabel('Read Count',fontsize=12)
    #plt.legend(bbox_to_anchor=(1.04, 1),ncol=1, frameon=False)
    if legend:
        plt.legend(frameon=False)

    if title:
        plt.title(f"{model.capitalize()}",fontsize=12)
    plt.tight_layout()
    return fig


def plot_log_abundance(true,params_time,model, K_N0,colors,figsize,legend=True):

    growth_rates_true,log_abundances_true = true
    times_fine,selected_indices = params_time
    times = times_fine[selected_indices]

    ys_fine = dynamics(model,K_N0,times_fine,growth_rates_true,log_abundances_true)
    K,T = ys_fine.size()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111) 

    for k in range(K):
        color = colors[k % len(colors)]
        plt.plot(times_fine, ys_fine[k], '-', color=color, label=f'Var {k+1}')

    if model != 'exponential':
        x_min, x_max = ax.get_xlim()
        start = x_min + 1/2*(x_max - x_min)   # last third
        end   = x_max
        ax.hlines(y=K_N0, xmin=start, xmax=end, linestyle="--", color="gray")


    plt.xticks(times)
    if legend:
        plt.legend(frameon=False)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.xlabel('Time',fontsize=12)
    plt.ylabel('Log Abundance', fontsize=12)

    plt.title(f"{model.capitalize()}",fontsize=12)
    plt.tight_layout()
    return fig


def plot_empirical(reads,true,params_time,model,K_N0,colors,markers,figsize,legend=True):

    fracs_empir = reads/reads.sum(dim=0)

    growth_rates_true,log_abundances_true = true
    times_fine,selected_indices = params_time
    times = times_fine[selected_indices]

    ys_fine = dynamics(model,K_N0,times_fine,growth_rates_true,log_abundances_true)
    fracs_fine = torch.softmax(ys_fine,dim=0)

    K,T = fracs_empir.size()

    fig = plt.figure(figsize=figsize)    

    for k in range(K):
        color = colors[k % len(colors)]
        marker=markers[k % len(markers)]
        if k == 0:
            plt.plot(times_fine, fracs_fine[k], '--', color=color, label='True')
        else:
            plt.plot(times_fine, fracs_fine[k], '--', color=color)
            
        plt.plot(times, fracs_empir[k], ' ',color=color,marker=marker, label=f'Emp {k+1}')

    plt.xticks(times)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Variant Fraction', fontsize=12)

    if legend:
        plt.legend(ncol=1, frameon=False)
    plt.title(f"{model.capitalize()}",fontsize=12)

    plt.tight_layout()
    return fig


def plot_logAbd_readCnt_Emp(reads,true,params_time,model,K_N0,colors,markers,figsize):

    fracs_empir = reads/reads.sum(dim=0)

    growth_rates_true,log_abundances_true = true
    times_fine,selected_indices = params_time
    times = times_fine[selected_indices]

    ys_fine = dynamics(model,K_N0,times_fine,growth_rates_true,log_abundances_true)
    fracs_fine = torch.softmax(ys_fine,dim=0)

    K,T = reads.size()

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 1)

    # FIRST subplot 
    ax1 = plt.subplot(gs[0])

    for k in range(K):
        color = colors[k % len(colors)]
        ax1.plot(times_fine, ys_fine[k], '-', color=color, label=f'Var {k+1}')

    if model != 'exponential':
        x_min, x_max = ax1.get_xlim()
        start = x_min + 1/2*(x_max - x_min)   # last third
        end   = x_max
        ax1.hlines(y=K_N0, xmin=start, xmax=end, linestyle="--", color="gray")

    ax1.set_xticks(times)
    ax1.set_ylabel('Log Abundance', fontsize=12)
    ax1.set_title(f"{model.capitalize()} growth",fontsize=14)

    # Force y-axis ticks to integers
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax1.legend(frameon=False)


    # SECOND subplot 
    ax2 = plt.subplot(gs[1])

    for k in range(K):
        color = colors[k % len(colors)]
        marker=markers[k % len(markers)]
        ax2.plot(times, reads[k], ':', color=color,marker=marker, label='Var '+str(k+1))


    ax2.set_xticks(times)
    ax2.set_ylim(-20,435)
    ax2.set_ylabel('Read Counts',fontsize=12)
    #ax2.set_xlabel('Time',fontsize=12)

    #ymin, ymax = ax2.get_ylim()
    #print(ymin,ymax)

    ax2.legend(frameon=False)

    # THIRD subplot 
    ax3 = plt.subplot(gs[2])

    for k in range(K):
        color = colors[k % len(colors)]
        marker=markers[k % len(markers)]
        if k == 0:
            ax3.plot(times_fine, fracs_fine[k], '--', color=color, label='True')
        else:
            ax3.plot(times_fine, fracs_fine[k], '--', color=color)
            
        ax3.plot(times, fracs_empir[k],linestyle='None',color=color,marker=marker, label=f'Emp {k+1}')

    ax3.set_xticks(times)
    ax3.set_ylim(-0.05,1.05)
    ax3.set_ylabel('Variant Fractions', fontsize=12)
    ax3.set_xlabel('Time', fontsize=12)

    ax3.legend(ncol=1, frameon=False)

    #ymin, ymax = ax3.get_ylim()
    #print(ymin,ymax)

    fig.align_ylabels([ax1, ax2,ax3])


    plt.tight_layout()

    return fig


# Fractions from all-time analysis

def plot_fractions(reads,true,fit,labels,params_time,model,K_N0,
    colors,markers,figsize,ylabel=True,legend=True,title=None,
    sigmas=None,Mtrj=500):

    fracs_empir = reads/reads.sum(dim=0)

    growth_rates_true,log_abundances_true = true
    growth_rates_fit,log_abundances_fit = fit
    label_true, label_fit = labels

    times_fine,selected_indices = params_time
    
    times = times_fine[selected_indices]

    K = growth_rates_fit.size(0)
    t = times_fine.size(0)

    f_top    = torch.zeros(2,K,t)
    f_bottom = torch.zeros(2,K,t)

    if sigmas:
        sigmasL_VB,sigmasD_VB = sigmas
        epsL = torch.randn(K, Mtrj)
        epsD = torch.randn(K, Mtrj)

        for m,mult in enumerate([2,1]):
            epsL[epsL.abs() > mult] = 0
            lams = growth_rates_fit[:,None] + sigmasL_VB[:,None] * epsL # [K,M]
            epsD[epsD.abs() > mult] = 0
            dels = log_abundances_fit[:,None] + sigmasD_VB[:,None] * epsD # [K,M]
            ys = dynamics(model,K_N0, times_fine, lams, dels)  # [K, M, T]

            phis = torch.softmax(ys,dim=0) # [K,M,T]
            f_top[m], _ = torch.max(phis, dim=1) # [K,T]
            f_bottom[m], _ = torch.min(phis, dim=1) # [K,T]
    
    fracs_true =torch.softmax(dynamics(model,K_N0,times_fine,growth_rates_true,log_abundances_true),dim=0)
    fracs_fit =torch.softmax(dynamics(model,K_N0,times_fine,growth_rates_fit,log_abundances_fit),dim=0)


    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(2, 1)

    # Create the first subplot (linear)
    ax1 = plt.subplot(gs[0])

    #colors = plt.cm.tab10(np.linspace(0, 1, 10))
    #for k in range(4):
    for k in range(K):
    
        color = colors[k % len(colors)]
        marker=markers[k % len(markers)]


        #if k == K-1:
        if k == 0:
            ax1.plot(times_fine,fracs_fit[k],'-', color=color,label=label_fit)
            if sigmas:
                ax1.fill_between(times_fine, f_bottom[0,k], f_top[0,k], color=color, alpha=0.2,label=f'$\pm2\sigma$')
                ax1.fill_between(times_fine, f_bottom[1,k], f_top[1,k], color=color, alpha=0.4,label=f'$\pm1\sigma$')
            ax1.plot(times_fine,fracs_true[k], '--', color=color, label=label_true)
        else:
            ax1.plot(times_fine,fracs_fit[k], '-', color=color) # Fit
            if sigmas:
                ax1.fill_between(times_fine, f_bottom[0,k], f_top[0,k], color=color, alpha=0.2)
                ax1.fill_between(times_fine, f_bottom[1,k], f_top[1,k], color=color, alpha=0.4)
            ax1.plot(times_fine,fracs_true[k],'--',color=color) # True

        ax1.plot(times, fracs_empir[k], ' ',color=color,marker=marker,label=f'Emp {k+1}')

    #ax1.set_xlabel('Time', fontsize=12)
    ax1.set_xticks(times)
    if K < 20:
        ax1.set_ylim(-0.05,1.05) # K=50
    else:
        ax1.set_ylim(-0.03,0.24)
    ax1.set_ylabel('linear', fontsize=10)

    if legend:
        ax1.legend(ncol=2,frameon=False)

    if title:
        ax1.set_title(title,fontsize=14)
    else:
        ax1.set_title(f"{model.capitalize()} growth",fontsize=14)

    # Create the second subplot (logarithmic)
    ax2 = plt.subplot(gs[1])

    for k in range(K):
        color = colors[k % len(colors)]
        marker=markers[k % len(markers)]
    #for ik,k in enumerate(range(K-6,K-3)):
    #    color = colors[ik+6 % len(colors)]
    #    marker=markers[ik+5 % len(markers)]

        ax2.plot(times,fracs_empir[k],' ',marker=marker, color=color,label=f'Emp {k+1}')

        if k == K-4:
            ax2.plot(times_fine,fracs_fit[k],'-', color=color,label=label_fit)
            if sigmas:
                ax2.fill_between(times_fine, f_bottom[0,k], f_top[0,k], color=color, alpha=0.2,label=r'$\pm2\sigma$')
                ax2.fill_between(times_fine, f_bottom[1,k], f_top[1,k], color=color, alpha=0.4,label=r'$\pm1\sigma$')
            ax2.plot(times_fine,fracs_true[k],'--',color=color,label=label_true)
        else:
            ax2.plot(times_fine,fracs_fit[k],'-', color=color)
            if sigmas:
                ax2.fill_between(times_fine, f_bottom[0,k], f_top[0,k], color=color, alpha=0.2)
                ax2.fill_between(times_fine, f_bottom[1,k], f_top[1,k], color=color, alpha=0.4)
            ax2.plot(times_fine,fracs_true[k],'--',color=color)
     

    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_xticks(times)
    if K > 20:
        ax2.set_ylim(3e-7,6e-2) # K=50
    else:
        ax2.set_ylim(6e-4,1.5) # K=4
    
    ax2.set_ylabel('logarithmic', fontsize=10)
    ax2.set_yscale('log')

    #ax2.legend(ncol=1,frameon=False)

    fig.align_ylabels([ax1, ax2])

    if ylabel:
        fig.text(-0.03, 0.5, 'Variant Fractions', va='center', rotation='vertical',fontsize=12)

    #fig.suptitle(f"{model.capitalize()}", fontsize=14)
    plt.tight_layout()

    return fig


def plot_fractions_many(reads,true,fit,labels,params_time,model,K_N0,
    colors,markers,figsize,ylabel=True,legend=True,title=None,
    sigmas=None,Mtrj=500):

    colors = plt.cm.tab10(np.linspace(0, 1, 8))

    fracs_empir = reads/reads.sum(dim=0)

    growth_rates_true,log_abundances_true = true
    growth_rates_fit,log_abundances_fit = fit
    label_true, label_fit = labels

    times_fine,selected_indices = params_time
    
    times = times_fine[selected_indices]

    K = growth_rates_fit.size(0)
    t = times_fine.size(0)

    f_top    = torch.zeros(2,K,t)
    f_bottom = torch.zeros(2,K,t)

    if sigmas:
        sigmasL_VB,sigmasD_VB = sigmas
        epsL = torch.randn(K, Mtrj)
        epsD = torch.randn(K, Mtrj)

        for m,mult in enumerate([2,1]):
            epsL[epsL.abs() > mult] = 0
            lams = growth_rates_fit[:,None] + sigmasL_VB[:,None] * epsL # [K,M]
            epsD[epsD.abs() > mult] = 0
            dels = log_abundances_fit[:,None] + sigmasD_VB[:,None] * epsD # [K,M]
            ys = dynamics(model,K_N0, times_fine, lams, dels)  # [K, M, T]

            phis = torch.softmax(ys,dim=0) # [K,M,T]
            f_top[m], _ = torch.max(phis, dim=1) # [K,T]
            f_bottom[m], _ = torch.min(phis, dim=1) # [K,T]
    
    fracs_true =torch.softmax(dynamics(model,K_N0,times_fine,growth_rates_true,log_abundances_true),dim=0)
    fracs_fit =torch.softmax(dynamics(model,K_N0,times_fine,growth_rates_fit,log_abundances_fit),dim=0)


    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(2, 1)

    # Create the first subplot (linear)
    ax1 = plt.subplot(gs[0])

    #colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for k in range(4):
    #for k in range(K):
    
        color = colors[k+3 % len(colors)]
        marker=markers[k+3 % len(markers)]


        #if k == K-1:
        if k == 0:
            ax1.plot(times_fine,fracs_fit[k],'-', color=color)#,label=label_fit)
            if sigmas:
                ax1.fill_between(times_fine, f_bottom[0,k], f_top[0,k], color=color, alpha=0.3)#,label=f'$\pm2\sigma$')
            ax1.plot(times_fine,fracs_true[k], '--', color=color)#, label=label_true)
        else:
            ax1.plot(times_fine,fracs_fit[k], '-', color=color) # Fit
            if sigmas:
                ax1.fill_between(times_fine, f_bottom[0,k], f_top[0,k], color=color, alpha=0.3)
            ax1.plot(times_fine,fracs_true[k],'--',color=color) # True

        ax1.plot(times, fracs_empir[k], ' ',color=color,marker=marker,label=f'Emp {k+1}')

    #ax1.set_xlabel('Time', fontsize=12)
    ax1.set_xticks(times)
    if K < 20:
        ax1.set_ylim(-0.05,1.05) # K=50
    else:
        ax1.set_ylim(-0.03,0.24)
    ax1.set_ylabel('linear', fontsize=10)

    if legend:
        ax1.legend(ncol=1,frameon=False)

    if title:
        ax1.set_title(title,fontsize=14)

    # Create the second subplot (logarithmic)
    ax2 = plt.subplot(gs[1])

    #for k in range(K):
    #    color = colors[k % len(colors)]
    #    marker=markers[k % len(markers)]
    for ik,k in enumerate(range(K-6,K-3)):
        color = colors[ik % len(colors)]
        marker=markers[ik % len(markers)]

        ax2.plot(times,fracs_empir[k],' ',marker=marker, color=color,label=f'Emp {k+1}')

        if k == K-4:
            ax2.plot(times_fine,fracs_fit[k],'-', color=color,label=label_fit)
            if sigmas:
                ax2.fill_between(times_fine, f_bottom[0,k], f_top[0,k], color=color, alpha=0.3,label=r'$\pm2\sigma$')
            ax2.plot(times_fine,fracs_true[k],'--',color=color,label=label_true)
        else:
            ax2.plot(times_fine,fracs_fit[k],'-', color=color)
            if sigmas:
                ax2.fill_between(times_fine, f_bottom[0,k], f_top[0,k], color=color, alpha=0.3)
            ax2.plot(times_fine,fracs_true[k],'--',color=color)

        #ax2.plot(times,fracs_empir[k],' ',marker=marker, color=color,label=f'Emp {k+1}')

    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_xticks(times)
    if K > 20:
        ax2.set_ylim(3e-7,6e-2) # K=50
    else:
        ax2.set_ylim(6e-4,1.5) # K=4
    
    ax2.set_ylabel('logarithmic', fontsize=10)
    ax2.set_yscale('log')

    ax2.legend(ncol=1,frameon=False)

    fig.align_ylabels([ax1, ax2])

    if ylabel:
        fig.text(-0.03, 0.5, 'Variant Fractions', va='center', rotation='vertical',fontsize=12)

    #fig.suptitle(f"{model.capitalize()}", fontsize=14)
    plt.tight_layout()

    return fig

# Gaussian posteriors

def plot_Gaussians(true,maxlike,means,sigmas,model,colors,figsize,title=None,ML=True):

    K = means[0].size(0)

    growth_rates_true,log_abundances_true = true
    growth_rates_ML,log_abundances_ML = maxlike
    growth_rates_VB,log_abundances_VB = means
    sigmasL_VB, sigmasD_VB = sigmas

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1)

    # FIRST subplot 
    ax1 = plt.subplot(gs[0])

    for b in range(K):
        color = colors[b % len(colors)]
        
        mu = growth_rates_VB[b]
        sig = sigmasL_VB[b]

        yy = torch.linspace(mu-3*sig,mu+3*sig,100)
        xx = torch.exp(-0.5*((yy-mu)/sig)**2)

        y_vert = growth_rates_true[b]
        if model == 'exponential':
            y_vert = growth_rates_true[b] - growth_rates_true.mean()

        if b==K-1:        
            ax1.plot(0.8*xx + (b+1) - 0.3, yy, '-', color=color,label='Var Inf')
            if ML:
                ax1.plot([b+0.7,b+1.5],[growth_rates_ML[b],growth_rates_ML[b]],'--', color=color,label='Max Like') 
            ax1.plot([b+0.7,b+1.5],[y_vert,y_vert],'-k', label='True')
        else:
            ax1.plot(0.8*xx + (b+1) - 0.3, yy,'-', color=color)
            if ML:
                ax1.plot([b+0.7,b+1.5],[growth_rates_ML[b],growth_rates_ML[b]],'--', color=color)
            ax1.plot([b+0.7,b+1.5],[y_vert,y_vert],'-k')
    
    #if model != 'gompertz':
        #ax1.set_ylim(-0.59,0.39)  
                
    ax1.set_xlim(0.1,K+0.9)
    #ax1.set_xticks(range(1,K+1))
    ax1.set_ylabel("Growth Rate",fontsize=12)
    ax1.legend(loc='lower left',frameon=False)
    if title:
        ax1.set_title(title,fontsize=14)
    else:
        ax1.set_title(f"{model.capitalize()} growth",fontsize=14)

    # SECOND subplot
    ax2 = plt.subplot(gs[1], sharex=ax1)
    # Turn off y-axis labels for ax1 to avoid redundancy
    plt.setp(ax1.get_xticklabels(), visible=False)

    for b in range(K):
        color = colors[b % len(colors)]
        
        mu = log_abundances_VB[b]
        sig = sigmasD_VB[b]

        yy = torch.linspace(mu-3*sig,mu+3*sig,100)
        xx = torch.exp(-0.5*((yy-mu)/sig)**2)

        y_vert = log_abundances_true[b] 

        if b==K-1:        
            ax2.plot(0.8*xx + (b+1) - 0.3, yy, '-', color=color,label='Var Inf')
            if ML:
                ax2.plot([b+0.7,b+1.5],[log_abundances_ML[b],log_abundances_ML[b]],'--', color=color,label='Max Like')
            ax2.plot([b+0.7,b+1.4],[y_vert,y_vert],'-k', label='True')
            put_labels = False
        else:
            ax2.plot(0.8*xx + (b+1) - 0.3, yy,'-', color=color)
            if ML:
                ax2.plot([b+0.7,b+1.5],[log_abundances_ML[b],log_abundances_ML[b]],'--', color=color)
            ax2.plot([b+0.7,b+1.5],[y_vert,y_vert],'-k')

    #ax2.set_ylim(-2.4,-0.6) # K = 4
                
    ax2.set_xlim(0.1,K+0.9)
    if K > 20:
        ax2.set_xticks(range(0,K+1,int(K/20)))
    else:
        ax2.set_xticks(range(1,K+1))
    ax2.set_ylabel("Initial Log Abundance",fontsize=12)
    ax2.set_xlabel("Variant Number",fontsize=12)
    #ax2.legend(loc='upper right',frameon=False)

    fig.align_ylabels([ax1, ax2])

    plt.tight_layout()

    return fig



