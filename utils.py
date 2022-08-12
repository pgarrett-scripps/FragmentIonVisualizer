import pandas as pd
from pyopenms import MSSpectrum, TheoreticalSpectrumGenerator, AASequence
import plotly.graph_objects as go

from params import FragmentSpectraParams


def convert_df(df_to_download):
    return df_to_download.to_csv(index=False).encode('ascii')


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
