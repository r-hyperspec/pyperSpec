[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable release yet.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)


# Python Package pyperspec

This is a Python package designed to simplify the analysis and manipulation of hyperspectral datasets.
The package provides an object-oriented approach providing a user-friendly interface that feels familiar to Python users, i.e. close to libraries such as `numpy`, `pandas`, and `scikit-learn`.

This is heavily inspired by R package [hyperSpec](https://github.com/r-hyperspec/hyperSpec) and part of **`r-hyperspec`**.
The goal is to make the work with hyperspectral data sets, (i.e. spatially or time-resolved spectra, or spectra with any other kind of information associated with each of the spectra) more comfortable.
The spectra can be data obtained during 
[XRF](https://en.wikipedia.org/wiki/X-ray_fluorescence),
[UV/VIS](https://en.wikipedia.org/wiki/Ultraviolet%E2%80%93visible_spectroscopy), 
[Fluorescence](https://en.wikipedia.org/wiki/Fluorescence_spectroscopy),
[AES](https://en.wikipedia.org/wiki/Auger_electron_spectroscopy),
[NIR](https://en.wikipedia.org/wiki/Near-infrared_spectroscopy),
[IR](https://en.wikipedia.org/wiki/Infrared_spectroscopy), 
[Raman](https://en.wikipedia.org/wiki/Raman_spectroscopy), 
[NMR](https://en.wikipedia.org/wiki/Nuclear_magnetic_resonance_spectroscopy), 
[MS](https://en.wikipedia.org/wiki/Mass_spectrometry),
etc. spectroscopy measurements.

**NOTE:** The main focus is **not** on algroithms since there are already many other good packages implementing algorithms, e.g. 
(numpy)[https://numpy.org/],
(scipy)[https://scipy.org/],
(pybaselines)[https://github.com/derb12/pybaselines],
(and more can be found in [FOSS For Spectroscopy](https://bryanhanson.github.io/FOSS4Spectroscopy/) list).

Rather, it provides convinient interface for those algorithms and other routine tasks.

For detailed information and documentation, please visit [PyPerSpec Documentation](#TODO).

## Documentation

TODO

## Installation

Currently available only from GitHub:

```bash
pip install git+https://github.com/r-hyperspec/pyperspec.git
```

## Quick Demo

```python
from pyspc import SpectraFrame

sf = SpectraFrame(...,wl=wl, data=data)

sf.A
sf["A"]
sf["E"] = ...
sf[:,:,500:1000]
sf[:,:,500:1000].plot()
sf[:,:,500:1000].plot(colors="B")
```

## Acknowlegments

* This project was a continuation of [ibcp/pyspectra](https://github.com/ibcp/pyspectra). We acknowlege support and contribution of [Emanuel Institute of Biochemical Physics, RAS](https://biochemphysics.ru/)
* This project has received funding from the European Union’s [Horizon 2020](https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-2020_en) research and innovation programme under the [Marie Sklodowska-Curie Actions](https://marie-sklodowska-curie-actions.ec.europa.eu/) *(Grant Agreement 861122)* as part of [IMAGE-IN](https://image-in-itn.eu/) project.
* The project was developed as part of secondment at [Chemometrix GmbH](https://chemometrix.gmbh/)
* Supervision was from [Chemometrix GmbH](https://chemometrix.gmbh/), [Leibniz-IPHT](https://www.leibniz-ipht.de/en/), and [BMD Software](https://www.bmd-software.com/)

<img src="https://biochemphysics.ru/static/img/logo_en.png" alt="Emanuel Institute of Biochemical Physics, RAS" height="100"/>
<img src="https://bgsmath.cat/wp-content/uploads/2017/09/marie_curie1-300x160.jpg" alt="Horizon 2020" height="100"/>
<img src=".images/logo_imagein.png" alt="IMAGE-IN" height="100"/>

<img src="https://chemometrix.gmbh/assets/images/chemometrix-logo.png" alt="Chemometrix GmbH" height="50">
<img src="https://image-in-itn.eu/wp-content/uploads/2020/09/IPHTLogo_neu_rgb_01_de.png" alt="Leibniz-IPHT" height="50">
<img src="https://image-in-itn.eu/wp-content/uploads/2020/09/bmd-1024x213.png" alt="BMD Software" height="50" width="200">