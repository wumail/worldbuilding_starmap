"""Download a small, reproducible, publicly served CMD isochrone grid, with provenance."""
import hashlib
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request

REQUEST = {
    "cmd_version":"3.9", "track_omegai":"0.00", "track_parsec":"parsec_CAF09_v1.2S",
    "track_colibri":"parsec_CAF09_v1.2S_S_LMC_08_web", "track_postagb":"no",
    "n_inTPC":"0", "eta_reimers":"0.2", "kind_interp":"1", "kind_postagb":"-1",
    "photsys_file":"YBC_tab_mag_odfnew/tab_mag_ubvrijhk.dat", "photsys_version":"YBCnewVega",
    "dust_sourceM":"dpmod60alox40", "dust_sourceC":"AMCSIC15", "kind_mag":"2", "kind_dust":"0",
    "extinction_av":"0.0", "extinction_coeff":"constant", "extinction_curve":"cardelli",
    "kind_LPV":"4", "imf_file":"tab_imf/imf_kroupa_orig.dat", "isoc_isagelog":"1",
    "isoc_agelow":"1.0e9", "isoc_ageupp":"1.0e10", "isoc_dage":"0.0",
    "isoc_lagelow":"6.5", "isoc_lageupp":"10", "isoc_dlage":"0.5",
    "isoc_ismetlog":"1", "isoc_zlow":"0.0152", "isoc_zupp":"0.03", "isoc_dz":"0.0",
    "isoc_metlow":"-1", "isoc_metupp":"0", "isoc_dmet":"1", "output_kind":"0",
    "output_evstage":"1", "lf_maginf":"-15", "lf_magsup":"20", "lf_deltamag":"0.5",
    "sim_mtot":"1.0e4", "submit_form":"Submit"
}

if __name__ == "__main__":
    endpoint = "https://stev.oapd.inaf.it/cgi-bin/cmd_3.9"
    request = urllib.request.Request(endpoint, data=urllib.parse.urlencode(REQUEST).encode())
    result = urllib.request.urlopen(request, timeout=180).read().decode()
    match = re.search(r"(?:\.\./)?tmp/output[0-9]+\.dat", result)
    if not match:
        raise RuntimeError("CMD returned no downloadable isochrone grid")
    url = urllib.parse.urljoin(endpoint, match[0])
    raw = urllib.request.urlopen(url, timeout=90).read()
    if b"# Zini" not in raw or b"Vmag" not in raw:
        raise RuntimeError("Unexpected CMD output")
    from paths import DATA_DIR
    folder = DATA_DIR / "parsec"
    folder.mkdir(exist_ok=True)
    path = folder / "cmd39_population.dat"
    if path.exists():
        raise FileExistsError("Refusing to overwrite an existing scientific grid")
    path.write_bytes(raw)
    (folder / "population_source.json").write_text(json.dumps({
        "endpoint":endpoint, "request":REQUEST, "temporary_download_url":url,
        "sha256":hashlib.sha256(raw).hexdigest(), "bytes":len(raw),
        "version":"PARSEC v1.2S + COLIBRI, CMD 3.9", "interstellar_Av":0,
        "notes":"Coarse evolutionary stage labels; label 9 / logL=-9.999 is a terminator, not a white dwarf. Post-AGB disabled. Circumstellar TP-AGB dust retained.",
        "citation":"References in the original preserved header; Bressan et al. 2012; Marigo et al. 2017; Chen et al. 2019.",
        "terms":"Public scientific service with citation instructions; no explicit redistribution license located."
    }, indent=2)+"\n")
    print(path, len(raw), hashlib.sha256(raw).hexdigest(), flush=True)
