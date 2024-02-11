import os 
import sys
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightkurve as lk
from lightkurve.interact import _get_corrected_coordinate as getcoord


root = "plots"
download_dir="../data"
   
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

def get_tpfs(tic, quality_bitmask="hardest"):
    files = sorted(glob(os.path.join(
        download_dir, "mastDownload", "TESS", f"*{tic}*", "*tp.fits")))
    if len(files) > 0:
        tpfs = lk.TargetPixelFileCollection(
            [lk.read(f, quality_bitmask=quality_bitmask) for f in files])
    else:
        sr = lk.search_targetpixelfile(f"TIC {tic}", author="spoc", cadence=120)
        tpfs = sr.download_all(quality_bitmask=quality_bitmask, download_dir=download_dir)
        if len(tpfs) == 0:
            return -1        
        tpfs = tpfs[np.argsort([t.sector for t in tpfs])]
    return tpfs

def get_pixels(tpf):
    target_ra, target_dec, pm_corrected = getcoord(tpf)
    pix_x, pix_y = tpf.wcs.all_world2pix([(target_ra, target_dec)], 0)[0]
    target_x, target_y = tpf.column + pix_x, tpf.row + pix_y
    return target_x, target_y

def plot_transits(lc, stats, save_path):
    ax = lc.plot()
    ax.legend().remove()
    
    ymin, ymax = ax.get_ylim()
    
    med_flux = np.median(lc.flux.value)
    
    ax.vlines(stats["transit_times"].value, ymin=ymin, ymax=ymax, color="k", linestyle=":", alpha=0.3)
    ax.axhline(med_flux - stats["depth"][0].value, color="k")
    ax.axhline(med_flux, color="k")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def initial_download():
    cvzs = load_ebs()

    print("TIC,sector,camera,ccd," 
        "target_x,target_y,crowdsap,flfrcsap,"
        "period,duration,btjd0,flux_median,e_flux_median,"
        "primary_depth,primary_std",
        #"secondary_depth,secondary_std,"
        sep=",", file=sys.stdout)

    for tic, row in cvzs.loc[382517745:].iterrows():
        p0 = row["period"]
        dur = row["prim_width_2g"]
        
        tpfs = get_tpfs(tic)
        if tpfs == -1:
            continue

        path = os.path.join(root, f"{tic:010d}")
        os.makedirs(path, exist_ok=True)

        for tpf in tpfs:
            sector = tpf.sector
            camera = tpf.camera
            ccd = tpf.ccd
            
            target_x, target_y = get_pixels(tpf)
            crowdsap = tpf.hdu[1].header["CROWDSAP"]
            flfrcsap = tpf.hdu[1].header["FLFRCSAP"]

            lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)
            lc = lc[lc["quality"] == 0].remove_nans()

            try:
                pg = lc.to_periodogram("bls", minimum_period=p0/1.5, maximum_period=1.5*p0)
            except ValueError:
                print(f"Period ({p0:.3f}) and Duration ({dur:.3f}) break the periodogram. Skipping.", 
                    file=sys.stderr)
                continue
            
            period = pg.period_at_max_power
            duration = pg.duration_at_max_power
            btjd0 = pg.transit_time_at_max_power

            mask = pg.get_transit_mask(period, duration, btjd0)
            flux = lc.flux.value[~mask]
            flux_median = np.median(flux)
            flux_std = np.std(flux, ddof=1)

            stats = pg.compute_stats(period, duration, btjd0)
            primary_depth, primary_std = stats["depth"]

            plot_transits(lc, stats, save_path=os.path.join(path, f"TIC{tic:010d}-s{sector:04d}.png"))
            print(tic, sector, camera, ccd, 
                target_x, target_y, crowdsap, flfrcsap,
                period.value, duration.value, btjd0.value,
                flux_median, flux_std,
                primary_depth.value, primary_std.value, 
                #secondary_depth, secondary_std, 
                sep=",", file=sys.stdout)


def reanalyze():
    tics = pd.read_csv("eb_transits.csv", index_col="TIC").index.unique()
    cvzs = load_ebs().loc[tics].sort_index()

    print("TIC,sector,camera,ccd," 
        "npix_tpf,npx_ap,"
        "target_x,target_y,crowdsap,flfrcsap,"
        "period,duration,btjd0,flux_median,e_flux_median,"
        "primary_depth,primary_std",
        #"secondary_depth,secondary_std,"
        sep=",", file=sys.stdout)

    for tic, row in cvzs.iterrows():
        print(f"Trying TIC {tic}...", file=sys.stderr)
        p0 = row["period"]
        dur = row["prim_width_2g"]

        tpfs = get_tpfs(tic)
        if tpfs == -1:
            print("No TPFs found.", file=sys.stderr)
            continue

        path = os.path.join(root, f"{tic:010d}")

        for tpf in tpfs:
            sector = tpf.sector
            camera = tpf.camera
            ccd = tpf.ccd
            aperture = tpf.pipeline_mask
            npix_tpf = np.product(aperture.shape)
            npix_ap = aperture.sum()

            target_x, target_y = get_pixels(tpf)
            crowdsap = tpf.hdu[1].header["CROWDSAP"]
            flfrcsap = tpf.hdu[1].header["FLFRCSAP"]

            lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)
            lc = lc[lc["quality"] == 0].remove_nans()

            try:
                pg = lc.to_periodogram("bls", minimum_period=p0/1.5, maximum_period=1.5*p0)
            except ValueError:
                print(f"Period ({p0:.3f}) and Duration ({dur:.3f}) break the periodogram. Skipping.", 
                    file=sys.stderr)
                continue
            
            period = pg.period_at_max_power
            duration = pg.duration_at_max_power
            btjd0 = pg.transit_time_at_max_power

            mask = pg.get_transit_mask(period, duration, btjd0)
            flux = lc.flux.value[~mask]
            flux_median = np.median(flux)
            flux_std = np.std(flux, ddof=1)

            stats = pg.compute_stats(period, duration, btjd0)
            primary_depth, primary_std = stats["depth"]

            #plot_transits(lc, stats, save_path=os.path.join(path, f"TIC{tic:010d}-s{sector:04d}.png"))
            print("Success!", file=sys.stderr)
            print(tic, sector, camera, ccd, 
                npix_tpf, npix_ap,
                target_x, target_y, crowdsap, flfrcsap,
                period.value, duration.value, btjd0.value,
                flux_median, flux_std,
                primary_depth.value, primary_std.value, 
                #secondary_depth, secondary_std, 
                sep=",", file=sys.stdout)


if __name__ == "__main__":
    reanalyze()
