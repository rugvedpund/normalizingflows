import lusee
import matplotlib.pyplot as plt
import numpy as np


def z2nu(z):
    return 1420.4 / (z + 1)


def nu2z(nu):
    return 1420.4 / nu - 1


freqs = np.arange(1, 101)
fDA = np.arange(1, 51)
fCD = np.arange(51, 101)
tDA = 1e3 * lusee.MonoSkyModels.T_DarkAges(freqs)

modelDA = 1e3 * lusee.MonoSkyModels.T_DarkAges_Scaled(
    fDA, nu_rms=14, nu_min=16.4, A=0.04
)
modelCD = 1e3 * lusee.MonoSkyModels.T_CosmicDawn_Scaled(
    fCD, nu_rms=20, nu_min=67.5, A=0.130
)

fig = plt.figure(figsize=(6, 5))
fig.subplots_adjust(top=0.9, bottom=0.25, left=0.1, right=0.9)

plt.plot(
    freqs, 1e3 * lusee.MonoSkyModels.T_DarkAges(freqs), c="k", label="True 21cm Signal"
)
plt.plot(fDA, modelDA, label="Dark Ages Model")
plt.plot(fCD, modelCD, label="Cosmic Dawn Model")



# fac = np.linspace(-0.1, 0.1, 10)
# for f in fac:
#     plt.plot(fDA, 1e3*lusee.MonoSkyModels.T_DarkAges_Scaled(fDA, nu_rms=14+10*f, nu_min=16.4+10*f, A=0.04+0.1*f), c="C0",alpha=0.5, lw=0.5)
#     plt.plot(fCD, 1e3*lusee.MonoSkyModels.T_CosmicDawn_Scaled(fCD, nu_rms=20+10*f, nu_min=67.5+10*f, A=0.130+0.1*f), c="C1",alpha=0.5, lw=0.5)

##---------------------------------------------------------------------------##
ymax,ymin = 50, -180

plt.axhline(0.0, c="k", ls="--")
plt.axvline(50.5, c="k", lw=0.5)


plt.fill_betweenx([ymin,ymax], 1, 50.5, color="C0", alpha=0.1)
plt.fill_betweenx([ymin,ymax], 50.5, 100, color="C1", alpha=0.1)



plt.plot([16.4, 16.4], [ymin, 0], c="C0", ls="--", lw=0.5)
plt.plot([67.5, 67.5], [ymin, 0], c="C1", ls="--", lw=0.5)
plt.plot([1, 16.4], [-41, -41], c="C0", ls="--", lw=0.5)
plt.plot([1, 67.5], [-130, -130], c="C1", ls="--", lw=0.5)

plt.ylabel(r"$A$ [mK]")
plt.xlabel(r"$\nu$ [MHz]")

plt.ylim(ymin, ymax)
plt.xlim(1, 100)

# plt.arrow(16.4-14/2,-20,14,0,head_width=1,head_length=5,fc='k',ec='k')
# plt.arrow(67.5-20/2,-110,20,0,head_width=1,head_length=5,fc='k',ec='k')
# plt.plot([16.4-14.0/2,16.4+14.0/2],[-30,-30],c='C0',lw=0.5)
# plt.plot([67.5-20.0/2,67.5+20.0/2],[-77,-77],c='C1',lw=0.5)

plt.text( 2, -45, r"$A_0 = -40$ [mK]", horizontalalignment="left", verticalalignment="top", color="C0", fontsize="small",)
plt.text( 51.5, -135, r"$A_0 = -130$ [mK]", horizontalalignment="left", verticalalignment="top", color="C1", fontsize="small",)

plt.text( 17.4, ymin+5, r"$\nu_{\rm min} = 16.4$ [MHz]", horizontalalignment="left", verticalalignment="bottom", color="C0", fontsize="small",)
plt.text( 68.5, ymin+5, r"$\nu_{\rm min} = 67.5$ [MHz]", horizontalalignment="left", verticalalignment="bottom", color="C1", fontsize="small",)

plt.text( 16.4, 0, r"$\nu_{\rm rms} = 14$ [MHz]", horizontalalignment="center", verticalalignment="bottom", color="C0", fontsize="small",)
plt.text( 67.5, 0, r"$\nu_{\rm rms} = 20$ [MHz]", horizontalalignment="center", verticalalignment="bottom", color="C1", fontsize="small",)

plt.title("21cm Signal Models for Dark Ages and Cosmic Dawn")
fig.legend(
    bbox_to_anchor=(0.5, 0.1), loc="center", borderaxespad=0, ncol=3, fontsize="small"
)

plt.savefig("SignalModels.pdf", bbox_inches="tight", dpi=300)
plt.show()
