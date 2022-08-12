from typing import List, Union

import streamlit as st
from dataclasses import dataclass
from pyopenms import *
import plotly.graph_objects as go
import numpy as np
import pandas as pd

ION_COLUMNS = ['a', 'b', 'c', 'x', 'y', 'z']
DEFAULT_IONS_COLUMNS = ['b', 'y']
ADD_A_ION_HELP_MESSAGE="Add peaks of a-ions to the spectrum"
ADD_B_ION_HELP_MESSAGE="Add peaks of b-ions to the spectrum"
ADD_C_ION_HELP_MESSAGE="Add peaks of c-ions to the spectrum"
ADD_X_ION_HELP_MESSAGE="Add peaks of x-ions to the spectrum"
ADD_Y_ION_HELP_MESSAGE="Add peaks of y-ions to the spectrum"
ADD_Z_ION_HELP_MESSAGE="Add peaks of z-ions to the spectrum"
A_ION_INTENSITY_HELP_MESSAGE="Intensity of the A-ions"
B_ION_INTENSITY_HELP_MESSAGE="Intensity of the B-ions"
C_ION_INTENSITY_HELP_MESSAGE="Intensity of the C-ions"
X_ION_INTENSITY_HELP_MESSAGE="Intensity of the X-ions"
Y_ION_INTENSITY_HELP_MESSAGE="Intensity of the Y-ions"
Z_ION_INTENSITY_HELP_MESSAGE="Intensity of the Z-ions"
ION_INTENSITY_HELP_MESSAGE="Intensity of the fragment ion"
ISOTOPE_MODEL_HELP_MESSAGE = "Model to use for isotopic peaks ('none' means no isotopic peaks are added, 'coarse' adds isotopic peaks in unit mass distance, 'fine' uses the hyperfine isotopic generator to add accurate isotopic peaks. Note that adding isotopic peaks is very slow."
MAX_ISOTOPE_HELP_MESSAGE = "Defines the maximal isotopic peak which is added if 'isotope_model' is 'coarse'"
MAX_ISOTOPE_PROBABILITY_HELP_MESSAGE="Defines the maximal isotopic probability to cover if 'isotope_model' is 'fine'"
ADD_LOSS_HELP_MESSAGE="Include neutral loss peaks in fragment spectra"
PEAK_META_INFO_HELP_MESSAGE="Adds the type of peaks as metainfo to the peaks, like y8+, [M-H2O+2H]++"
ADD_PRECURSOR_PEAKS_HELP_MESSAGE="Adds peaks of the unfragmented precursor ion to the spectrum"
ADD_ALL_PRECURSOR_CHARGES_HELP_MESSAGE="Adds precursor peaks with all charges in the given range"
ADD_ABUNDANT_IMMONIUM_IONS_HELP_MESSAGE="Add most abundant immonium ions (for Proline, Cystein, Iso/Leucine, Histidin, Phenylalanin, Tyrosine, Tryptophan)"
ADD_FIRST_PREFIX_ION_HELP_MESSAGE="If set to true e.g. b1 ions are added"
RELATIVE_LOSS_INTENSITY_HELP="Intensity of loss ions, in relation to the intact ion intensity"
PRECURSOR_INTENSITY_HELP="Intensity of the precursor peak"
PRECURSOR_H2O_INTENSITY="Intensity of the H2O loss peak of the precursor"
PRECURSOR_NH3_INTENSITY="Intensity of the NH3 loss peak of the precursor"
ION_COLUMNS_HELP_MESSAGE = 'ion types to use in fragmentation'

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

def convert_df(df_to_download):
    return df_to_download.to_csv(index=False).encode('ascii')

tsg_param_conv = lambda use_ion: 'true' if use_ion else 'false'


def get_theoretical_spectra_generator_from_params(fragment_spectra_params: FragmentSpectraParams):
    tsg = TheoreticalSpectrumGenerator()
    tsg_params = tsg.getParameters()
    tsg_params.setValue("add_a_ions", tsg_param_conv('a' in fragment_spectra_params.ion_types))
    tsg_params.setValue("add_b_ions", tsg_param_conv('b' in fragment_spectra_params.ion_types))
    tsg_params.setValue("add_c_ions", tsg_param_conv('c' in fragment_spectra_params.ion_types))
    tsg_params.setValue("add_x_ions", tsg_param_conv('x' in fragment_spectra_params.ion_types))
    tsg_params.setValue("add_y_ions", tsg_param_conv('y' in fragment_spectra_params.ion_types))
    tsg_params.setValue("add_z_ions", tsg_param_conv('z' in fragment_spectra_params.ion_types))

    for ion_type, intensity in zip(fragment_spectra_params.ion_types, fragment_spectra_params.ion_types_intensities):
        tsg_params.setValue(f"{ion_type}_intensity", intensity)

    tsg_params.setValue("precursor_H2O_intensity", fragment_spectra_params.precursor_H2O_intensity)
    tsg_params.setValue("precursor_NH3_intensity", fragment_spectra_params.precursor_NH3_intensity)

    tsg_params.setValue("isotope_model", fragment_spectra_params.isotope_model)
    if fragment_spectra_params.isotope_model == 'coarse':
        tsg_params.setValue("max_isotope", fragment_spectra_params.max_isotope)
        tsg_params.setValue("max_isotope_probability", fragment_spectra_params.max_isotope_probability)
    tsg_params.setValue("add_losses", tsg_param_conv(fragment_spectra_params.losses))
    tsg_params.setValue("add_abundant_immonium_ions", tsg_param_conv(fragment_spectra_params.immonium_ions))
    tsg_params.setValue("add_precursor_peaks", tsg_param_conv(fragment_spectra_params.precursor_peak))
    if fragment_spectra_params.precursor_peak:
        tsg_params.setValue("add_all_precursor_charges", tsg_param_conv(fragment_spectra_params.all_precursor_charges))
    tsg_params.setValue("add_metainfo", tsg_param_conv(fragment_spectra_params.meta_info))
    tsg_params.setValue("add_first_prefix_ion", tsg_param_conv(fragment_spectra_params.first_prefix_ion))
    tsg.setParameters(tsg_params)
    return tsg

def get_color_from_annotation(annot):
    import plotly.express as px
    if 'a' in annot:
        return px.colors.qualitative.Plotly[0]
    if 'b' in annot:
        return px.colors.qualitative.Plotly[1]
    if 'c' in annot:
        return px.colors.qualitative.Plotly[2]
    if 'x' in annot:
        return px.colors.qualitative.Plotly[3]
    if 'y' in annot:
        return px.colors.qualitative.Plotly[4]
    if 'z' in annot:
        return px.colors.qualitative.Plotly[5]
    return px.colors.qualitative.Plotly[6]


def flatten(l):
    return [item for sublist in l for item in sublist]


def add_prefix_postfix_sequences(df, sequence_to_protein_map):
    ser_peptides_rows = []
    for sequence in df.sequence.values:
        pp_sequences = []
        for protein in sequence_to_protein_map[sequence]:
            peptide_start_index = protein.sequence.find(sequence)
            peptide_end_index = peptide_start_index + len(sequence)
            prefix_aa = "-"
            if peptide_start_index != 0:
                prefix_aa = protein.sequence[peptide_start_index - 1]
            postfix_aa = "-"
            if peptide_end_index != len(protein.sequence):
                postfix_aa = protein.sequence[peptide_end_index]
            pp_sequences.append(f"{prefix_aa}.{sequence}.{postfix_aa}")
        ser_peptides = " ".join(list(set(pp_sequences)))
        ser_peptides_rows.append(ser_peptides)
    df['ser_peptides'] = ser_peptides_rows

def get_fragment_ions_df(peptide, charge, fragment_spectra_params):
    theo_spectrum = MSSpectrum()
    tsg = get_theoretical_spectra_generator_from_params(fragment_spectra_params)
    tsg.getSpectrum(theo_spectrum, AASequence.fromString(peptide), 1, charge)
    theo_mz, theo_int = theo_spectrum.get_peaks()
    annot = theo_spectrum.getStringDataArrays()[0]

    data = []
    for mz, intensity, an in zip(theo_mz, theo_int, annot):
        data.append({'ion': an.decode(), 'mz': mz, 'intensity': intensity,
                     'color': get_color_from_annotation(an.decode())})
    df = pd.DataFrame(data).sort_values(by=['mz'])
    return df

def get_fragment_spectra_plot(peptide, charge, fragment_spectra_params):
    frag_df = get_fragment_ions_df(peptide, charge, fragment_spectra_params)
    data = [go.Scatter(x=frag_df.mz, y=frag_df.intensity, text=frag_df.ion, mode='markers', marker=dict(color='red'))]

    # Use the 'shapes' attribute from the layout to draw the vertical lines
    layout = go.Layout(shapes=[
        dict(type='line', xref='x', yref='y', x0=frag_df.mz.values[i], y0=0, x1=frag_df.mz.values[i],
             y1=frag_df.intensity.values[i],
             line=dict(color=frag_df.color.values[i], width=1)) for i in range(len(frag_df))],
        title='Lollipop Chart'
    )
    fig = go.Figure(data, layout)
    return fig

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
        all_precursor_charges = st.checkbox('add all precursor charges', value=False, help=ADD_ALL_PRECURSOR_CHARGES_HELP_MESSAGE)

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


fragment_spectra_params = get_fragment_spectra_params()

st.header("Fragment Ion Calculator")
seq_col, charge_col = st.columns(2)
with seq_col:
    peptide = st.text_input('Peptide Sequence', 'PEPTIDE')
with charge_col:
    charge = st.number_input('Charge:', min_value=0, max_value=1000, value=1,
                             help="peptide charge state to use for fragmentation")

fig = get_fragment_spectra_plot(peptide, charge, fragment_spectra_params)
st.plotly_chart(fig)

frag_df = get_fragment_ions_df(peptide, charge, fragment_spectra_params)[['ion', 'mz', 'intensity']]
st.dataframe(frag_df)

st.download_button(label="Download Frag Ions",
                   data=convert_df(frag_df),
                   file_name='frag_ions.csv',
                   mime='text/csv', )