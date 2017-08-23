#Reference: V.Viswanathan (2017) - PhD thesis

from skew_normal import random_skewnormal,skewnormal_parms,skewnormal_stats, pdf_skewnormal,rnd_skewnormal
import matplotlib.pyplot as plt
from numpy.random import *
import numpy as np
from scipy.signal import correlate

def oca_method(s,rejection_level,refl_std,bin_width):
    # bin_width=0.2
    n_bin=100/bin_width
    x,y,o=plt.hist(s,bins=n_bin,range=(-50,50),color='black',alpha=1,label="initial residuals")
    plt.close('all')
    
    
    a=np.array(x)
    # if np.size(correl_shape)==0:
    b=np.array([2,7,10,7,2])
    #   print "shape absent in interval"
    #   print "using correl_shape:",b
    # else:
        # b=array([2,7,10,7,2])
    

    cres1=correlate(a,b,mode='same')
    # print "Correl :",cres1
    # print "cor int:",y
    # show()

    indices = np.where(cres1 == cres1.max())
    corr_mean=cres1.mean()
    # print corr_mean
    # print len(cres1[cres1>corr_mean])
    if cres1.max()==0 :
        print "no normal points returned : correlation 0 :",filename
        normal_point0=np.array([0,0,0,0])
        normal_p=np.array([normal_point0])
        return normal_p

    resid_int=y[indices[0][0]]
    # x,y,o=plt.hist(cres1,bins=100,color='g',alpha=0.5,label="correlation")    
    # plt.show()
    # plt.close('all')

    resid_int_end=resid_int + bin_width 
    # print "resid intervals: ",resid_int,resid_int_end
    cond1=s>=resid_int
    # print cond1
    cond2=s<=resid_int_end
    # print cond2
    cond_al0=np.all([cond1,cond2],axis=0)

    # print "selected ref :",select_refl
    # if select_refl==103:
    #     std_cal=0.3
    # else :
    #     std_cal=0.2
    # print "std_cal:",std_cal

    residual_ok=s[cond_al0]


    avg_residual=np.mean(residual_ok)
    std_residual=np.std(residual_ok)

    k=len(residual_ok)
    sr=sum(residual_ok)
    sr2=sum(residual_ok*residual_ok)
    std_residual=np.sqrt((sr2-sr*sr/k)/k)
    # check_std=std(residual_ok)
    # print "first avg residual",avg_residual

    # print "first std residual",std_residual,check_std
    # print "first avg date",avg_date/3600

    check_len1=len(residual_ok)
    resid_int = avg_residual - (rejection_level)*(refl_std)
    resid_int_end = (avg_residual) + (rejection_level)*(refl_std)
    # residual_ok_1=data[0][(data[0]<=(resid_int_end)) & (data[0]>=resid_int)]
    # avg_residual=mean(data[0,cond_al2])
    
    # print "resid int"
    # print resid_int
    # print resid_int_end

    cond1=s>=resid_int
    cond2=s<=resid_int_end
    cond_al0=np.all([cond1,cond2],axis=0)
    residual_ok_1=s[cond_al0]
    cond_altest=cond_al0
    
    # print data[0]
    # print cond_al0
    avg_residual=np.mean(residual_ok_1)
    # print "residok :",residual_ok_1
    # print "mean:", avg_residual
    # print "std res :", std(residual_ok_1)
    
    # print data[0][cond_al3]
    resid_sub=s[cond_al0]

    avg_residual=np.mean(resid_sub)
    # sigma_i=np.std(resid_sub)
    sigma_i=np.sqrt(sum(np.square(s[cond_al0]-avg_residual))/len(residual_ok_1))
    return avg_residual,sigma_i
    # print avg_residual,"+/-",sigma_i

def fit_two_peaks_EM(sample, sigma, weights=False, p0=np.array([0.5,1.0,0.2,1.5,2.0,0.2,0.33]), max_iter=10000, tollerance=1e-10):
    
    if not weights: w = np.ones(sample.size)
    else: w = 1./(sigma**2)
    w *= 1.*w.size/w.sum() # renormalization so they sum to N
    
    # Initial guess of parameters and initializations
    mu  = np.array([p0[0], p0[3]])
    sig = np.array([p0[1], p0[4]])
    sk  = np.array([p0[2], p0[5]])
    pi_ = np.array([p0[6], (1-p0[6])])
    
    gamma, N_ = np.zeros((2, sample.size)), np.zeros(2)
    p_new = np.array(p0)
    #print ('Initial conditions :\n',p_new)
    N = sample.size
    
    # EM loop
    counter = 0
    converged, stop_iteration = False, False
    while not stop_iteration:
        p_old = p_new
        # Compute the responsibility func. and new parameters
        for k in [0,1]:
            #gamma[k,:] = w*pi_[k]*normpdf(sample, mu[k], sig[k])/pdf_model(sample, p_new) # SCHEME1
            locp1, scalep1, shapep1 = skewnormal_parms(mu[k],sig[k],sk[k])
            # print(pdf_model(sample,p_new,2))
            #print(pdf_skewnormal(sample,locp1,scalep1,shapep1))
            gamma[k,:] = pi_[k]*(pdf_skewnormal(sample,locp1,scalep1,shapep1)/pdf_model(sample,p_new,2))

            #gamma[k,:] = pi_[k]*normpdf(sample, mu[k], sig[k])/pdf_model(sample, p_new) # SCHEME2
            N_[k] = gamma[k,:].sum()
            
            sk[k] = sum(gamma[k]*skewnormal_stats(locp1,scalep1,shapep1)[2])/N_[k]# SCHEME1
            mu[k] = sum(gamma[k]*sample)/N_[k]

            #mu[k] = sum(w*gamma[k]*sample)/sum(w*gamma[k]) # SCHEME2
            sig[k] = np.sqrt(sum(gamma[k]*(sample-mu[k])**2)/N_[k])
            pi_[k] = 1.*N_[k]/N
        p_new = np.array([mu[0], sig[0], sk[0], mu[1], sig[1],sk[1], pi_[0]])
        
        #assert abs(N_.sum() - N)/float(N) < 1e-6 
        #assert abs(pi_.sum() - 1) < 1e-6
        
        # Convergence check
        counter += 1
        max_variation = max((p_new-p_old)/p_old)
        converged = True if max_variation < tollerance else False
        stop_iteration = converged or (counter >= max_iter)
    #print "Iterations:", counter
    if not converged: print ("WARNING: Not converged")
    return p_new

def fit_one_peak_EM(sample, sigma, weights=False, p0=np.array([0.5,0.5,0.5,1]), max_iter=10000, tollerance=1e-10):
    
    if not weights: w = np.ones(sample.size)
    else: w = 1./(sigma**2)
    w *= 1.*w.size/w.sum() # renormalization so they sum to N
    
    # Initial guess of parameters and initializations
    mu  = np.array([p0[0]])
    sig = np.array([p0[1]])
    sk  = np.array([p0[2]])
    pi_ = np.array([p0[3]])
    
    gamma, N_ = np.zeros((1, sample.size)), np.zeros(1)
    p_new = np.array(p0)
    #print ('Initial conditions :\n',p_new)
    N = sample.size
    
    # EM loop
    counter = 0
    converged, stop_iteration = False, False
    while not stop_iteration:
        p_old = p_new
        # Compute the responsibility func. and new parameters
        for k in [0]:
            #gamma[k,:] = w*pi_[k]*normpdf(sample, mu[k], sig[k])/pdf_model(sample, p_new) # SCHEME1
            locp1, scalep1, shapep1 = skewnormal_parms(mu[k],sig[k],sk[k])
            # print(pdf_model(sample,p_new,2))
            #print(pdf_skewnormal(sample,locp1,scalep1,shapep1))
            gamma[k,:] = pi_[k]*(pdf_skewnormal(sample,locp1,scalep1,shapep1)/pdf_model(sample,p_new,1))

            #gamma[k,:] = pi_[k]*normpdf(sample, mu[k], sig[k])/pdf_model(sample, p_new) # SCHEME2
            N_[k] = gamma[k,:].sum()
            
            sk[k] = sum(gamma[k]*skewnormal_stats(locp1,scalep1,shapep1)[2])/N_[k]# SCHEME1
            mu[k] = sum(gamma[k]*sample)/N_[k]

            #mu[k] = sum(w*gamma[k]*sample)/sum(w*gamma[k]) # SCHEME2
            sig[k] = np.sqrt(sum(gamma[k]*(sample-mu[k])**2)/N_[k])
            pi_[k] = 1.*N_[k]/N
        p_new = np.array([mu[0], sig[0], sk[0], pi_[0]])
        
        #assert abs(N_.sum() - N)/float(N) < 1e-6 
        #assert abs(pi_.sum() - 1) < 1e-6
        
        # Convergence check
        counter += 1
        max_variation = max((p_new-p_old)/p_old)
        converged = True if max_variation < tollerance else False
        stop_iteration = converged or (counter >= max_iter)
    #print "Iterations:", counter
    if not converged: print ("WARNING: Not converged")
    return p_new

def pdf_model(x, p, n_G):

    if(n_G==3):
        mu1, sig1,sk1, mu2, sig2,sk2, mu3, sig3,sk3, pi_1, pi_2= p
        locp1, scalep1, shapep1 = skewnormal_parms(mu1,sig1,sk1)
        locp2, scalep2, shapep2 = skewnormal_parms(mu2,sig2,sk2)
        locp3, scalep3, shapep3 = skewnormal_parms(mu3,sig3,sk3)
        out=pi_1*pdf_skewnormal(x,locp1,scalep1,shapep1) + pi_2*pdf_skewnormal(x,locp2,scalep2,shapep2) + (1-pi_1-pi_2)*pdf_skewnormal(x,locp3,scalep3,shapep3)
    if(n_G==2):
        #print('PDF MODEL for 2 Gaussian')
        mu1, sig1,sk1, mu2, sig2,sk2, pi_1= p
        locp1, scalep1, shapep1 = skewnormal_parms(mu1,sig1,sk1)
        locp2, scalep2, shapep2 = skewnormal_parms(mu2,sig2,sk2)
        out=pi_1*pdf_skewnormal(x,locp1,scalep1,shapep1) + (1-pi_1)*pdf_skewnormal(x,locp2,scalep2,shapep2)
    if(n_G==1):
        #print('PDF MODEL for 2 Gaussian')
        mu1, sig1,sk1, pi_1= p
        locp1, scalep1, shapep1 = skewnormal_parms(mu1,sig1,sk1)
        # locp2, scalep2, shapep2 = skewnormal_parms(mu2,sig2,sk2)
        out=pi_1*pdf_skewnormal(x,locp1,scalep1,shapep1)
    #print(out)
    return out

def nfilter(samples,m,scale,s):
    # m=np.mean(samples[samples!=0])
    # print "np.mean :",m
    l1=m + scale*s
    l2=m - scale*s

    s1=samples[samples<l1]
    s1=s1[s1>l2]

    sub_samples=s1

    return sub_samples

##################################
#SELECT METHOD## - 1
GM_method=2
###
OCA_method=0
#Set rejection level
rejection_level=2.5
##################################
#INTRODUCE SKEW, IF REQUIRED
skew_sim=0.2
##################################

#Set iterations
n_iter=100

#Set photon count
nsamples=200
#Set noise samples
noise_samples=10*nsamples


mu_sim_0=0.5 #ns
sig_sim_0=0.293 #ns

mu_noise=0.0
sig_noise=50
skew_noise=0

#Set bin width
bin_width=0.2
nbin=10/bin_width

refl_std=0.3

m1=[1,0.3,0.1] #guess signal
m2=[0.001,100,0.1] #guess noise


###################################
weights=False


i=0
mu_iter=np.zeros(n_iter)
sig_iter=np.zeros(n_iter)
mun_iter=np.zeros(n_iter)
sign_iter=np.zeros(n_iter)
mu_error=np.zeros(n_iter)
std_error=np.zeros(n_iter)

# np.random.seed(1054)

# np.random.seed(1054)
# np.random.seed(1054)


while i<n_iter:
    np.random.seed(1054+i)
    mu_sim=mu_sim_0 * np.random.uniform(low=1.1,high=1.2,size=1)
    sig_sim=sig_sim_0 * np.random.uniform(low=1.1,high=1.2,size=1)
    print mu_sim,sig_sim
    
    sim_noise = random_skewnormal(mu_noise,sig_noise,skew_noise,noise_samples)
    sim_dist1 = random_skewnormal(mu_sim,sig_sim,skew_sim,nsamples)
    # cond1=(sim_dist1<=0.8)
    # cond2=(sim_dist1>=0.6)
    # rm_cond=~np.all([cond1,cond2],axis=0)
    # sim_dist1=sim_dist1[rm_cond]
    sim_distnoise=np.concatenate([sim_dist1,sim_noise])
    

    if GM_method==1:
        print "1 Gaussian EM ",i*100/n_iter
        fig = plt.figure()
        ax = fig.add_subplot(111)
        n, bins, patches=ax.hist(sim_distnoise,bins=nbin,range=(-5,5), alpha=0.5, color='r', normed=False)
        plt.close('all')
        bin_max = np.where(n == n.max())
        mu_filt=bins[bin_max][0]
        sim_sub=nfilter(sim_distnoise,mu_filt,rejection_level,refl_std)
        p0 = np.array([m1[0],m1[1],m1[2],1]) #guess
        p1=fit_one_peak_EM(sim_sub,1,weights,p0,max_iter=10000, tollerance=0.001)
        mu1,sig1,sk1,pi_1 = p1
        mu_iter[i]=mu1
        sig_iter[i]=sig1
    if OCA_method==1:
        print "OCA ",i*100/n_iter
        mu_iter[i],sig_iter[i]=oca_method(sim_distnoise,rejection_level,refl_std,bin_width)
        mu_error[i]=(mu_sim-mu_iter[i])*100/mu_sim
        std_error[i]=(sig_sim-sig_iter[i])*100/sig_sim
        # print "Result    :","(",round(mu_iter[i],3),"+/-",round(sig_iter[i],3),")","  [ns]"

    if GM_method==2:
        print "2 Gaussian EM ",i*100/n_iter
        sim_sub=sim_distnoise
        p0 = np.array([m1[0],m1[1],m1[2],m2[0],m2[1],m2[2],0.2]) #guess
        p1=fit_two_peaks_EM(sim_sub,1,weights,p0,max_iter=10000, tollerance=0.001)
        mu1,sig1,sk1,mu2,sig2,sk2,pi_1 = p1
        mu_iter[i]=mu1
        sig_iter[i]=sig1
        mun_iter[i]=mu2
        sign_iter[i]=sig2
        mu_error[i]=(mu_sim-mu1)*100/mu_sim
        std_error[i]=(sig_sim-sig1)*100/sig_sim
        # plt.figure()
        # # plt.hist(sim_distnoise,bins=nbin,range=(-5,5), alpha=0.5, color='r', normed=False)
        # plt.hist(sim_dist1,bins=nbin,range=(-5,5), alpha=1, color='k', normed=False)
        # plt.show()

    # print(p1)
    
    i=i+1

# print mu_iter[:]
plt.figure()
plt.hist(sim_noise,bins=nbin,range=(-5,5), alpha=0.5, color='r', normed=False, label="Noise")
plt.hist(sim_dist1,bins=nbin,range=(-5,5), alpha=1, color='k', normed=False, label="Simulated observation")
plt.xlabel("ns (bin width: 0.2ns)")
plt.ylabel("photon count")
plt.legend(loc='upper left')
plt.show()
# plt.close('all')

# print "Sim       :","(",round(mu_sim,3),"+/-",round(sig_sim,3),")","  [ns]"

if n_iter>1:
    print "Iterations: ",n_iter
    # print "Result    :","(",round(np.mean(mu_iter),3),"+/-",round(np.std(mu_iter),3),")","+/-","(", round(np.mean(sig_iter),3),"+/-",round(np.std(sig_iter),3),")","  [ns]"
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.hist(mu_iter,normed=False)
    ax1.set_title('Mean')
    ax2.hist(sig_iter,normed=False)
    ax2.set_title('Sigma')
    plt.show()


else:
    print "Result    :","(",round(mu_iter,3),"+/-", round(sig_iter,3),")","  [ns]"
    # print "EMnois::","(",round(mun_iter,3),"+/-", round(sign_iter,3),")","  [ns]"


# print " % Err:","(",round((mu_sim-np.mean(mu_iter))*100/mu_sim,1)," ",round((sig_sim-np.mean(sig_iter))*100/sig_sim,1),")" 
print " % Err:",np.mean(mu_error), np.mean(std_error)

# ##################################
# GM_method=0
# # GM_method=2
# OCA_method=1

# mu_sim=0.5 #ns
# sig_sim=0.293 #ns
# skew_sim=0.0
# bin_width=0.2
# nbin=10/bin_width
# nsamples=500

# mu_noise=0.0
# sig_noise=25
# skew_noise=0
# noise_samples=nsamples*20

# rejection_level=3
# refl_std=0.3

# m1=[1,0.3,0.1] #guess signal
# m2=[0.001,100,0.1] #guess noise


# ###################################
