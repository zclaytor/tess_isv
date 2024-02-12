import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
import lightkurve as lk
from astroquery.mast import Catalogs


def query_tic():
    cat1 = Catalogs.query_criteria(catalog="TIC", eclat=[75, 90], Tmag=[0, 4.5]).to_pandas(index="ID")
    cat2 = Catalogs.query_criteria(catalog="TIC", eclat=[-90, -75], Tmag=[0, 4.5]).to_pandas(index="ID")
    cat = pd.concat([cat1, cat2])
    cat.index = cat.index.astype(int)
    cat = cat.sort_index()
    return cat.reset_index()


def plot_tpfs_and_lcs(tic):
    """Download TPFs, plot TPFs, and return the sum of the whole TPF vs. time.
    """
    sr = lk.search_targetpixelfile(f"TIC {tic}", sector=range(1, 27), author="tess-spoc")
    tpfs = sr.download_all(flux_column="sap_flux")
    lcs = []
    
    for tpf in tpfs:
        # plot TPF
        fig, ax = plt.subplots()
        tpf.plot(ax=ax, aperture_mask=tpf.pipeline_mask, title=f"TIC {tic}, Sector {tpf.sector}", show_colorbar=False)
        fig.tight_layout()
        #ax.set_title(f"Sector {tpf.sector} | Npix: {tpf.shape[1]*tpf.shape[2]} | N_ap: {tpf.pipeline_mask.sum()}")
        plt.savefig(f"data/TIC{tic}-s{tpf.sector:04d}-tpf.png")
        plt.close()

        # sum TPF
        l = tpf.to_lightcurve(aperture_mask=np.ones_like(tpf.pipeline_mask, dtype=bool))
        lcs.append(l[l["quality"] == 0])

    return lk.LightCurveCollection(lcs)
    

def plot_lightcurves(row, sum_tpfs=None):
    tic = row["ID"]
    tmag = row["Tmag"]
    
    # Download SPOC light curves
    sr = lk.search_lightcurve(f"TIC {tic}", sector=range(1, 27), author="tess-spoc")
    lcs = sr.download_all(flux_column="sap_flux", quality_bitmask="hardest")

    # Compute raw light curves
    raw_lcs = lk.LightCurveCollection([])
    for l in lcs:
        raw_lcs.append(l + l["sap_bkg"])

    # plot light curves
    ax = lcs.plot()
    raw_lcs.plot(ax=ax, linestyle=":")
    if sum_tpfs is not None:
        sum_tpfs.plot(ax=ax, linewidth=2)
        
    ax.legend().remove()
    ax.set_title(f"Target ID: {tic}")
    plt.savefig(f"data/TIC{tic}-zlc.png", dpi=500)
    plt.close()
    
    
if __name__ == "__main__":
    stars = query_tic()
    for i, row in stars.iterrows():
        print(f"{i+1:2d}/{len(stars)}: TIC {row['ID']}")
        sum_tpfs = plot_tpfs_and_lcs(row["ID"])
        plot_lightcurves(row, sum_tpfs)
