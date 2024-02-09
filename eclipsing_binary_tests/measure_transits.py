import os 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightkurve as lk

   
def count_sectors(val):
    return len(val.split(","))

def load_ebs():
    ebs = pd.read_csv(
        "../data/hlsp_tess-ebs_tess_lcf-ffi_s0001-s0026_tess_v1.0_cat.csv", 
        index_col=0
        ).dropna(subset="sectors")
    
    ebs["n_sectors"] = ebs.sectors.apply(count_sectors)
    cvzs = ebs.query("n_sectors > 10").sort_index()
    return cvzs



if __name__ == "__main__":
    cvzs = load_ebs()

    plt.close("all")
    for tic, row in cvzs.iterrows():
        p0 = row["period"]
        dur = row["prim_width_2g"]
        
        sr = lk.search_lightcurve(f"TIC {tic}", author="spoc", cadence=120)
        lcs = sr.download_all(flux_column="sap_flux", quality_bitmask="hard")

        if len(lcs) == 0:
            continue
            
        sr = lk.search_targetpixelfile(f"TIC {tic}", author="spoc", cadence=120,
            sector=[l.sector for l in lcs])
        tpfs = sr.download_all(quality_bitmask="hard")
        
        path = f"ebs_astropy/{tic:010d}"
        os.makedirs(path, exist_ok=True)
        lcs = lcs[np.argsort([l.sector for l in lcs])]
        
        for lc in lcs:
            sector = lc.sector
            camera = lc.camera
            ccd = lc.ccd
            
            print(tic, sector)

            try:
                pg = lc.to_periodogram("bls", minimum_period=p0/1.5, maximum_period=3*p0)
            except ValueError:
                print(f"Period ({p0:.3f}) and Duration ({dur:.3f}) break the periodogram. Skipping.")
                continue
                
            ax = lc.plot()
            ax.legend().remove()
            
            ymin, ymax = ax.get_ylim()
            
            med_flux = np.median(lc.flux.value)
            
            stats = pg.compute_stats()
            ax.vlines(stats["transit_times"].value, ymin=ymin, ymax=ymax, color="k", linestyle=":", alpha=0.3)
            ax.axhline(med_flux - stats["depth"][0].value, color="k")
            ax.axhline(med_flux, color="k")
            plt.tight_layout()
            plt.savefig(os.path.join(path, f"TIC{tic:010d}-s{sector:04d}.png"))
            plt.close()
