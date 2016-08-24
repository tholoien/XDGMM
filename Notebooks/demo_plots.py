import numpy as np
from matplotlib import pyplot as plt
from astroML.plotting.tools import draw_ellipse
from astroML.plotting import setup_text_plots
from sklearn.mixture import GMM as skl_GMM

def plot_bic(param_range,bics,lowest_comp):
    plt.clf()
    setup_text_plots(fontsize=16, usetex=True)
    fig = plt.figure(figsize=(12, 6))
    plt.bar(param_range-0.25,bics,color='blue',width=0.5)
    plt.text(lowest_comp, bics.min() * 0.97 + .03 * bics.max(), '*',
             fontsize=14, ha='center')

    plt.xticks(param_range)
    plt.ylim(bics.min() - 0.01 * (bics.max() - bics.min()),
             bics.max() + 0.01 * (bics.max() - bics.min()))
    plt.xlim(param_range.min() - 1, param_range.max() + 1)

    plt.xticks(param_range,fontsize=14)
    plt.yticks(fontsize=14)


    plt.xlabel('Number of components',fontsize=18)
    plt.ylabel('BIC score',fontsize=18)

    plt.show()

def plot_val_curve(param_range, train_mean, train_std, test_mean,
                   test_std):
    plt.clf()
    setup_text_plots(fontsize=16, usetex=True)
    fig=plt.figure(figsize=(12,8))
    plt.plot(param_range, train_mean, label="Training score",
             color="red")
    plt.fill_between(param_range, train_mean - train_std,
                     train_mean + train_std, alpha=0.2, color="red")
    plt.plot(param_range, test_mean,label="Cross-validation score",
             color="green")
    plt.fill_between(param_range, test_mean - test_std,
                     test_mean + test_std, alpha=0.2, color="green")

    plt.legend(loc="best")
    plt.xlabel("Number of Components", fontsize=18)
    plt.ylabel("Score", fontsize=18)
    plt.xlim(param_range.min(),param_range.max())
    plt.show()

def plot_sample(x_true, y_true, x, y, sample, xdgmm):
    setup_text_plots(fontsize=16, usetex=True)
    plt.clf()
    fig = plt.figure(figsize=(12, 9))
    fig.subplots_adjust(left=0.1, right=0.95,
                        bottom=0.1, top=0.95,
                        wspace=0.02, hspace=0.02)

    ax1 = fig.add_subplot(221)
    ax1.scatter(x_true, y_true, s=4, lw=0, c='k')

    ax2 = fig.add_subplot(222)

    ax2.scatter(x, y, s=4, lw=0, c='k')

    ax3 = fig.add_subplot(223)
    ax3.scatter(sample[:, 0], sample[:, 1], s=4, lw=0, c='k')

    ax4 = fig.add_subplot(224)
    for i in range(xdgmm.n_components):
        draw_ellipse(xdgmm.mu[i], xdgmm.V[i], scales=[2], ax=ax4,
                     ec='k', fc='gray', alpha=0.2)

    titles = ["True Distribution", "Noisy Distribution",
              "Extreme Deconvolution\n  resampling",
            "Extreme Deconvolution\n  cluster locations"]

    ax = [ax1, ax2, ax3, ax4]

    for i in range(4):
        ax[i].set_xlim(-1, 13)
        ax[i].set_ylim(-6, 16)

        ax[i].xaxis.set_major_locator(plt.MultipleLocator(4))
        ax[i].yaxis.set_major_locator(plt.MultipleLocator(5))

        ax[i].text(0.05, 0.95, titles[i],
                   ha='left', va='top', transform=ax[i].transAxes)

        if i in (0, 1):
            ax[i].xaxis.set_major_formatter(plt.NullFormatter())
        else:
            ax[i].set_xlabel('$x$', fontsize = 18)

        if i in (1, 3):
            ax[i].yaxis.set_major_formatter(plt.NullFormatter())
        else:
            ax[i].set_ylabel('$y$', fontsize = 18)

    plt.show()

def plot_cond_model(xdgmm, cond_xdgmm, y):
    plt.clf()
    setup_text_plots(fontsize=16, usetex=True)
    fig = plt.figure(figsize=(12, 9))

    ax1 = fig.add_subplot(111)
    for i in range(xdgmm.n_components):
        draw_ellipse(xdgmm.mu[i], xdgmm.V[i], scales=[2], ax=ax1,
                     ec='k', fc='gray', alpha=0.2)

    ax1.plot([-2,15],[y,y],color='blue',linewidth=2)
    ax1.set_xlim(-1, 13)
    ax1.set_ylim(-6, 16)
    ax1.set_xlabel('$x$', fontsize = 18)
    ax1.set_ylabel('$y$', fontsize = 18)

    ax2 = ax1.twinx()
    x = np.array([np.linspace(-2,14,1000)]).T

    gmm=skl_GMM(n_components = cond_xdgmm.n_components,
                covariance_type = 'full')
    gmm.means_ = cond_xdgmm.mu
    gmm.weights_ = cond_xdgmm.weights
    gmm.covars_ = cond_xdgmm.V

    logprob, responsibilities = gmm.score_samples(x)

    pdf = np.exp(logprob)
    ax2.plot(x, pdf, color='red', linewidth = 2,
             label='Cond. dist. of $x$ given $y='+str(y)+'\pm 0.05$')
    ax2.legend()
    ax2.set_ylabel('Probability', fontsize= 18 )
    ax2.set_ylim(0, 0.52)
    ax1.set_xlim(-1, 13)
    plt.show()

def plot_cond_sample(x, y):
    plt.clf()
    setup_text_plots(fontsize=16, usetex=True)
    fig = plt.figure(figsize=(12, 9))

    plt.hist(x, 50, histtype='step', color='red',lw=2)

    plt.ylim(0,70)
    plt.xlim(-1,13)

    plt.xlabel('$x$', fontsize=18)
    plt.ylabel('Number of Points', fontsize=18)

    plt.show()

def plot_conditional_predictions(y, true_x, predicted_x):
    plt.clf()
    setup_text_plots(fontsize=16, usetex=True)
    fig = plt.figure(figsize=(12, 9))

    plt.scatter(true_x, y, color='red', s=4, marker='o',
                label="True Distribution")
    plt.scatter(predicted_x, y, color='blue', s=4, marker='o',
                label="Predicted Distribution")

    plt.xlim(-1, 13)
    plt.ylim(-6, 16)
    plt.legend(loc=2, scatterpoints=1)

    plt.xlabel('$x$', fontsize = 18)
    plt.ylabel('$y$', fontsize = 18)
    plt.show()
