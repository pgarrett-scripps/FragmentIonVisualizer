from dataclasses import dataclass
from typing import List, Union
import streamlit as st

from help_messages import *
from constants import *


@dataclass
class FragmentSpectraParams:
    ion_types: List[str]
    ion_types_intensities: List[float]
    precursor_H2O_intensity: float
    precursor_NH3_intensity: float
    isotope_model: str
    max_isotope: Union[int, None]
    max_isotope_probability: Union[float, None]
    losses: bool
    immonium_ions: bool
    precursor_peak: bool
    all_precursor_charges: Union[bool, None]
    first_prefix_ion: bool = True
    meta_info: bool = True


def get_fragment_spectra_params() -> FragmentSpectraParams:
    ion_types = st.multiselect('Ion Types:', ION_COLUMNS, default=DEFAULT_IONS_COLUMNS, help=ION_COLUMNS_HELP_MESSAGE)

    ion_intensity_input = lambda txt, msg: st.number_input(txt, min_value=0.0, value=1.0, step=0.1, help=msg)
    ion_types_intensities = []
    for ion_type in ion_types:
        ion_types_intensities.append(ion_intensity_input(f"{ion_type} ion intensity", ION_INTENSITY_HELP_MESSAGE))
    precursor_H2O_intensity = ion_intensity_input('precursor H2O intensity', PRECURSOR_H2O_INTENSITY)
    precursor_NH3_intensity = ion_intensity_input('precursor NH3 intensity', PRECURSOR_NH3_INTENSITY)

    isotope_model = st.radio("isotope model", ('none', 'coarse', 'fine'), index=0, help=ISOTOPE_MODEL_HELP_MESSAGE)
    max_isotope, max_isotope_probability = None, None
    if isotope_model == 'coarse':
        max_isotope = st.number_input('max isotopic peak', value=2, key='z_ion_intensity',
                                      help=MAX_ISOTOPE_HELP_MESSAGE)
        max_isotope_probability = st.number_input('max isotopic peak probability', value=0.05,
                                                  help=MAX_ISOTOPE_PROBABILITY_HELP_MESSAGE)
    losses = st.checkbox('include neutral loss peaks', value=False, help=ADD_LOSS_HELP_MESSAGE)
    immonium_ions = st.checkbox('add abundant immonium ions', value=False, help=ADD_ABUNDANT_IMMONIUM_IONS_HELP_MESSAGE)

    precursor_peaks = st.checkbox('add precursor peaks', value=False, help=ADD_PRECURSOR_PEAKS_HELP_MESSAGE)
    all_precursor_charges = None
    if precursor_peaks:
        all_precursor_charges = st.checkbox('add all precursor charges', value=False,
                                            help=ADD_ALL_PRECURSOR_CHARGES_HELP_MESSAGE)

    return FragmentSpectraParams(ion_types=ion_types,
                                 ion_types_intensities=ion_types_intensities,
                                 precursor_H2O_intensity=precursor_H2O_intensity,
                                 precursor_NH3_intensity=precursor_NH3_intensity,
                                 isotope_model=isotope_model,
                                 max_isotope=max_isotope,
                                 max_isotope_probability=max_isotope_probability,
                                 losses=losses,
                                 immonium_ions=immonium_ions,
                                 precursor_peak=precursor_peaks,
                                 all_precursor_charges=all_precursor_charges)
